from vllm.distributed.weight_transfer import (
        ModuleSource,
        HTTPVLLMWeightSyncClient,
        WeightTransferTrainerFactory,
)
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerInitInfo,
)

from transformers import AutoProcessor, AutoModelForMultimodalLM
from datasets import Dataset
import datasets
from openai import AsyncOpenAI
from argparse import ArgumentParser
from collections import defaultdict
from urllib.parse import urljoin

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

import os
import time
import wandb
import shutil
import logging
import requests
import asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def wait_for(path: str):
    while True:
        try:
            requests.get(path, timeout=5).raise_for_status()
            logger.info(f"READY: {path}")
            break
        except requests.exceptions.RequestException:
            logger.info(f"Waiting for {path}")
            time.sleep(2)


def initialize_engine(model, config):
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=NCCLTrainerInitInfo(
            master_address=get_ip(),
            master_port=get_open_port(),
            world_size=1+config.student_workers,
            rank=0,
            packed=True,
        ),
        client=HTTPVLLMWeightSyncClient(config.student_address),
        source=ModuleSource(model),
    )

    return engine


def load_dataset(config) -> Dataset:
    dataset = datasets.load_dataset(
        config.dataset, config.dataset_config, split=config.dataset_split)

    # we only need prompts; the completions all come from the student
    dataset = dataset.select_columns([config.prompt_column])
    if config.prompt_column != "prompt":
        dataset = dataset.rename_column(config.prompt_column, "prompt")

    dataset = dataset.shuffle(seed=config.seed)
    if config.max_prompts is not None:
        dataset = dataset.select(range(min(config.max_prompts, len(dataset))))

    return dataset


async def score(
        token_ids: list[int],
        teacher_client: AsyncOpenAI,
        config,
) -> list[float]:
    # "generating" 1 token with prompt_logprobs is how we get vllm to score an
    # existing sequence; prompt_logprobs=0 returns only the tokens we passed in
    response = await teacher_client.completions.create(
        model=config.teacher_model,
        prompt=token_ids,
        max_tokens=1,
        extra_body={"prompt_logprobs": 0},
    )

    # the first token has nothing to condition on, so vllm gives us None
    return [
        0. if token is None else next(iter(token.values()))["logprob"]
        for token in response.choices[0].prompt_logprobs
    ]


async def rollout(
        item: str,
        student_client: AsyncOpenAI,
        teacher_client: AsyncOpenAI,
        config,
) -> dict[str, list]:
    # we use the return_token_id's argument to avoid retokenization drift;
    # for more info, check out: 
    #    https://vllm.ai/blog/2025-10-22-agent-lightning
    #
    # we return the logprobs from vllm (even though we technically don't need 
    # to), to monitor any diff
    completion = await student_client.chat.completions.create(
        model=config.student_model,
        messages=[{"role": "user", "content": item}],
        max_tokens=config.max_tokens,
        logprobs=True,
        n=config.rollouts,
        extra_body={"return_token_ids": True, "top_k": -1},
        temperature=1.0,
        top_p=1.0
    ) 

    prompt_ids = completion.prompt_token_ids

    # for each rollout above, we want to score asynchronously
    teacher_logps = await asyncio.gather(*[
        score(prompt_ids + choice.token_ids, teacher_client, config)
        for choice in completion.choices
    ])

    samples = {
        "token_ids": [],
        "attention_mask": [],
        "mask": [],
        "student_logp": [],
        "teacher_logp": [],
    }
    for choice, teacher_logp in zip(completion.choices, teacher_logps):
        token_ids = prompt_ids + choice.token_ids
        student_logp = [token.logprob for token in choice.logprobs.content]
        assert len(student_logp) == len(choice.token_ids)
        assert len(teacher_logp) == len(token_ids)

        # everything is aligned to token_ids; prompt positions are masked out
        samples["token_ids"].append(token_ids)
        samples["attention_mask"].append([1]*len(token_ids))
        samples["mask"].append([0]*len(prompt_ids) + [1]*len(choice.token_ids))
        samples["student_logp"].append([0.]*len(prompt_ids) + student_logp)
        samples["teacher_logp"].append(teacher_logp)

    return samples


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum()


def compute_loss(
        student,
        sample: dict[str, torch.Tensor],
        is_clip: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = sample["token_ids"]

    logits = student(
            input_ids=input_ids,
            attention_mask=sample["attention_mask"]
    ).logits[:, :-1]

    # logits at position t predict token t+1, so everything else shifts by one
    logp = -F.cross_entropy(
        logits.float().transpose(1, 2),
        input_ids[:, 1:], reduction="none")
    sampled_logp = sample["student_logp"][:, 1:]
    teacher_logp = sample["teacher_logp"][:, 1:]
    mask = sample["mask"][:, 1:].float()

    # single-sample estimate of the negative reverse KL at each token
    advantage = teacher_logp - sampled_logp

    # the tokens were sampled from vllm's distribution, not the trainer's; even
    # with synced weights the two disagree numerically, so we reweight by the
    # (truncated) importance ratio. it's detached: a correction, not a gradient path
    ratio = torch.exp(logp.detach() - sampled_logp)
    is_weight = ratio.clamp(max=is_clip)

    loss = masked_mean(-logp * advantage * is_weight, mask)

    # drift is how far the trainer's logprobs are from the ones vllm sampled
    # with; right after a weight sync it should be ~0
    metrics = {
        "advantage": masked_mean(advantage, mask).item(),
        "drift": masked_mean((logp.detach() - sampled_logp).abs(), mask).item(),
        "is_ratio": masked_mean(ratio, mask).item(),
        "is_clip_frac": masked_mean((ratio > is_clip).float(), mask).item(),
    }

    return loss, metrics


def collate_fn(data: dict[str, list], device: str = "cuda") -> dict[str, torch.Tensor]:
    # torch infers int64 for the ids/masks and float32 for the logprobs
    return {
        k: pad_sequence(
            [torch.tensor(seq) for seq in v],
            batch_first=True, padding_value=0).to(device)
        for k, v in data.items()
    }


async def train_one_step(
    model,
    batch,
    config,
    student_client,
    teacher_client,
    optimizer,
) -> dict[str, float]:

    # rollouts for the whole batch are generated + scored concurrently
    samples = await asyncio.gather(*[
        rollout(item, student_client, teacher_client, config)
        for item in batch["prompt"]
    ])
    samples = [collate_fn(sample) for sample in samples]
    total_tokens = sum(sample["mask"].sum().item() for sample in samples)

    # each prompt's rollouts are one microbatch; weighting by its share of the
    # completion tokens makes the accumulated gradient (and the metrics) a
    # per-token mean over the full batch, without holding batch_size * rollouts
    # logits in memory
    optimizer.zero_grad()
    metrics = defaultdict(float)
    for sample in samples:
        weight = sample["mask"].sum().item() / total_tokens
        loss, micro_metrics = compute_loss(model, sample, config.is_clip)
        (loss * weight).backward()

        micro_metrics["loss"] = loss.item()
        for k, v in micro_metrics.items():
            metrics[k] += v * weight

    # clip_grad_norm_ returns the norm from *before* clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.max_grad_norm)
    optimizer.step()

    return {**metrics, "grad_norm": grad_norm.item(), "tokens": total_tokens}


def save_checkpoint(model, processor, config, step: int):
    # the processor goes alongside the weights so that the directory can be
    # served directly: `vllm serve checkpoints/step-000010`
    path = os.path.join(config.output_dir, f"step-{step:06d}")
    model.save_pretrained(path)
    processor.save_pretrained(path)
    logger.info(f"saved checkpoint to {path}")

    # checkpoints are ~2GB each, so by default only the most recent are kept
    if config.keep_checkpoints > 0:
        checkpoints = sorted(
            d for d in os.listdir(config.output_dir) if d.startswith("step-"))
        for stale in checkpoints[:-config.keep_checkpoints]:
            shutil.rmtree(os.path.join(config.output_dir, stale))


async def train(config):
    wait_for(urljoin(config.student_address, "health"))
    wait_for(urljoin(config.teacher_address, "health"))

    processor = AutoProcessor.from_pretrained(
            config.student_model)
    model = AutoModelForMultimodalLM.from_pretrained(
            config.student_model, device_map="auto")

    dataset = load_dataset(config)

    engine = initialize_engine(model, config)

    student_client = AsyncOpenAI(
        base_url=urljoin(config.student_address, "v1"),
        api_key="(empty)")
    
    teacher_client = AsyncOpenAI(
        base_url=urljoin(config.teacher_address, "v1"),
        api_key="(empty)")
    
    # reset the weights, so you're not using the first training run
    engine.send_weights()

    wandb.init(
        project=config.wandb_project,
        mode=config.wandb_mode,
        config=vars(config))

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    step = 0
    for epoch in range(config.epochs):
        for batch in dataset.iter(batch_size=config.batch_size):
            start = time.perf_counter()
            metrics = await train_one_step(
                model, batch, config,
                student_client, teacher_client, optimizer
            )
            engine.send_weights()

            metrics["epoch"] = epoch
            metrics["step_time"] = time.perf_counter() - start
            wandb.log(metrics, step=step)
            logger.info(f"step {step}: {metrics}")
            step += 1

            if step % config.save_every == 0:
                save_checkpoint(model, processor, config, step)

    # always keep the final weights, even if we stopped between saves
    if step % config.save_every != 0:
        save_checkpoint(model, processor, config, step)

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--student-workers", type=int, default=1)
    parser.add_argument("--student-address", type=str, default="http://localhost:8000")
    parser.add_argument("--teacher-address", type=str, default="http://localhost:8001")
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset-config", type=str, default="main")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--prompt-column", type=str, default="question")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--is-clip", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                        help="number of most recent checkpoints to keep; 0 keeps all")
    parser.add_argument("--wandb-project", type=str, default="nanopd")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    config = parser.parse_args()

    asyncio.run(train(config))

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
from openai import AsyncOpenAI
from argparse import ArgumentParser
from urllib.parse import urljoin

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

import time
import logging
import requests
import asyncio


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def wait_for(path: str):
    while True:
        try: requests.get(path)
        except:
            logger.info(f"Waiting for {path}")
            time.sleep(2)
        finally: 
            logger.info(f"READY: {path}")
            break


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


def load_dataset(*args, **kwargs):
    dataset = Dataset.from_dict({
        "prompt": [
            "Explain how monte-carlo estimators for on-policy distillation work.",
            "Give the equation for on-policy distillation loss in PyTorch.",
        ]
    })

    return dataset


async def rollout(
        item: str,
        student_client: AsyncOpenAI,
        teacher_client: AsyncOpenAI,
        config,
        processor
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
        extra_body={"return_token_ids": True},
    ) 

    prompt_ids = completion.prompt_token_ids

    # for each rollout above, we want to score asynchronously
    teacher_score_tasks = [
        teacher_client.completions.create(
            model=config.teacher_model,
            prompt=prompt_ids + choice.token_ids,
            max_tokens=1,
            extra_body={"prompt_logprobs": 0, "return_token_ids": True},
        )
        for choice in completion.choices
    ]

    teacher_scores = await asyncio.gather(*teacher_score_tasks)
    
    samples = {
        "token_ids": [],
        "mask": [],
        "student_logp": [],
        "teacher_logp": [],
    }
    for completion, teacher_score in zip(completion.choices, teacher_scores):
        samples["student_logp"].append(
            torch.tensor(
                [0.]*len(prompt_ids) + [token.logprob for token in completion.logprobs.content],
                device=torch.device("cuda")))
        samples["teacher_logp"].append(
            torch.tensor([
            0. if token is None else next(iter(token.values()))["logprob"]
            for token in teacher_score.choices[0].prompt_logprobs], device=torch.device("cuda")))
        samples["token_ids"].append(
                torch.tensor(
                prompt_ids + completion.token_ids, device=torch.device("cuda"), dtype=torch.int64))
        samples["mask"].append(
                torch.tensor(
                [0,]*len(prompt_ids) + [1,]*len(completion.token_ids), device=torch.device("cuda"), dtype=torch.int64))

    return samples


def compute_loss(student, sample: list[dict]):
    input_ids = sample["token_ids"]
    attention_mask = torch.tensor(torch.ones_like(input_ids), device=torch.device("cuda"), dtype=torch.int64)

    logits = student(
            input_ids=input_ids,
            attention_mask=attention_mask
    ).logits[:, :-1]

    student_logp = -F.cross_entropy(
        logits.float().transpose(1, 2),
        input_ids[:, 1:], reduction="none")

    advantage = -(sample["student_logp"][:, 1:] - sample["teacher_logp"][:, 1:])

    mask = sample["mask"][:, 1:].float()

    per_tok = (student_logp * advantage * mask)

    return per_tok.sum() / mask.sum()


def collate_fn(data: dict[str, list]):
    return {
        k: pad_sequence(v, batch_first=True, padding_value=0)
        for k, v in data.items()
    }


async def train_one_step(
    model,
    processor,
    batch,
    config,
    student_client,
    teacher_client,
):
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)

    for item in batch["prompt"]:
        sample = await rollout(item, student_client, teacher_client, config, processor)
        sample = collate_fn(sample)
        loss = compute_loss(model, sample)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


async def train(config):
    wait_for(urljoin(config.student_address, "health"))
    wait_for(urljoin(config.teacher_address, "health"))

    processor = AutoProcessor.from_pretrained(
            config.student_model)
    model = AutoModelForMultimodalLM.from_pretrained(
            config.student_model, device_map="auto")

    dataset = load_dataset(config.dataset)

    engine = initialize_engine(model, config)

    student_client = AsyncOpenAI(
        base_url=urljoin(config.student_address, "v1"),
        api_key="(empty)")
    
    teacher_client = AsyncOpenAI(
        base_url=urljoin(config.teacher_address, "v1"),
        api_key="(empty)")
    
    # reset the weights, so you're not using the first training run
    engine.send_weights()
    for step in range(config.epochs):
        for batch in dataset.iter(batch_size=config.batch_size):
            await train_one_step(
                model, processor, batch, config,
                student_client, teacher_client,
            )
            engine.send_weights()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--student-workers", type=int, default=1)
    parser.add_argument("--student-address", type=str, default="http://localhost:8000")
    parser.add_argument("--teacher-address", type=str, default="http://localhost:8001")
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", type=str, default="test_data")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    config = parser.parse_args()

    asyncio.run(train(config))

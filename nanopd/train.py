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
from collections import namedtuple

import time
import requests
import asyncio


Sample = namedtuple('Sample', ['student_completion', 'teacher_logp'])


def wait_for(path: str):
    while True:
        try: requests.get(path)
        except:
            print(f"Waiting for {path}")
            time.sleep(2)
        finally: 
            print(f"READY: {path}")
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
) -> Sample:
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

    # for each rollout above, we want to score asynchronously
    teacher_score_tasks = [
        teacher_client.chat.completions.create(
            model=config.teacher_model,
            messages=[
                {"role": "user", "content": item},
                {"role": "assistant", "content": choice.message.content},
            ],
            max_tokens=1,
            extra_body={"prompt_logprobs": 1},
        )
        for choice in completion.choices
    ]

    teacher_scores = await asyncio.gather(*teacher_score_tasks)

    prompt_ids = completion.prompt_token_ids
    
    for completion, teacher_score in zip(completion.choices, teacher_scores)
        completion_ids = completion.token_ids
        student_logp = ...
        teacher_logp = ...
        mask = []

    return Sample(
        token_ids,
        mask,
        completion.choices[0].logprobs,
        teacher_scores.prompt_logprobs
    )


def compute_reverse_kl(sample: Sample):
    pass


async def train_one_step(
    model,
    processor,
    batch,
    config,
    student_client,
    teacher_client,
):
    for item in batch["prompt"]:
        sample = await rollout(item, student_client, teacher_client, config)
        advantage = -1 * compute_reverse_kl(sample)
        loss = -(advantage * student_logp * mask).sum() / mask.sum()
        loss.backward()


async def train(config):
    wait_for(config.student_address + "/health")
    wait_for(config.teacher_address + "/health")

    processor = AutoProcessor.from_pretrained(
            config.student_model)
    model = AutoModelForMultimodalLM.from_pretrained(
            config.student_model, device_map="auto")

    dataset = load_dataset(config.dataset)

    engine = initialize_engine(model, config)

    student_client = AsyncOpenAI(
        base_url=config.student_address + "/v1",
        api_key="(empty)")
    
    teacher_client = AsyncOpenAI(
        base_url=config.teacher_address + "/v1",
        api_key="(empty)")
    
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    config = parser.parse_args()

    asyncio.run(train(config))

from openai import AsyncOpenAI
from argparse import ArgumentParser
from urllib.parse import urljoin

import re
import json
import asyncio
import logging
import datasets


logging.basicConfig(level=logging.INFO)
# one INFO line per request is a lot of noise at 1.3k questions
for name in ("httpx", "httpx2"):
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    numbers = NUMBER.findall(text)
    if not numbers:
        return None
    return float(numbers[-1].replace(",", ""))


def extract_answer(response: str) -> float | None:
    # prefer the last \boxed{...}; otherwise fall back to the last number in
    # the response (lm-eval's "flexible-extract")
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", response)
    if boxed and (answer := parse_number(boxed[-1])) is not None:
        return answer
    return parse_number(response)


def load_dataset(config) -> datasets.Dataset:
    dataset = datasets.load_dataset("openai/gsm8k", "main", split=config.split)
    if config.limit is not None:
        dataset = dataset.select(range(min(config.limit, len(dataset))))
    return dataset


async def evaluate_one(item: dict, client: AsyncOpenAI, semaphore, config) -> dict:
    async with semaphore:
        completion = await client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": item["question"] + config.prompt_suffix}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    choice = completion.choices[0]
    response = choice.message.content or ""

    # gsm8k references look like "<reasoning> #### 1,234"
    target = parse_number(item["answer"].split("####")[-1])
    predicted = extract_answer(response)

    return {
        "question": item["question"],
        "response": response,
        "target": target,
        "predicted": predicted,
        "correct": predicted is not None and abs(predicted - target) < 1e-6,
        "truncated": choice.finish_reason == "length",
    }


async def evaluate(config) -> dict[str, float]:
    dataset = load_dataset(config)

    client = AsyncOpenAI(
        base_url=urljoin(config.address, "v1"),
        api_key="(empty)")

    # vllm batches for us; the semaphore just keeps the request queue sane
    semaphore = asyncio.Semaphore(config.concurrency)
    results = await asyncio.gather(*[
        evaluate_one(item, client, semaphore, config)
        for item in dataset
    ])

    metrics = {
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "truncated": sum(r["truncated"] for r in results) / len(results),
        "no_answer": sum(r["predicted"] is None for r in results) / len(results),
        "n": len(results),
    }
    logger.info(f"{config.model} @ {config.address}: {metrics}")

    if config.output is not None:
        with open(config.output, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        logger.info(f"wrote {len(results)} responses to {config.output}")

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    # the trained weights only live in the student vllm server, so by default
    # we evaluate whatever it's currently serving
    parser.add_argument("--address", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--prompt-suffix", type=str,
                        default="\nPlease reason step by step, and put your final answer within \\boxed{}.")
    parser.add_argument("--output", type=str, default=None)
    config = parser.parse_args()

    asyncio.run(evaluate(config))

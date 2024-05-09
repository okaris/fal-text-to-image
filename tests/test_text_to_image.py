import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from tempfile import TemporaryDirectory
from typing import Any

import fal_client
import httpx
import pytest
from fal import flags
from fal.sdk import get_default_credentials
from image_similarity_measures.evaluate import evaluation as image_similarity

import text_to_image.model

SIMILARITY_METHOD = "fsim"
SIMILARITY_THRESHOLD = 0.6


MAX_TIME_TO_HEALTHY = 60


@pytest.fixture(scope="session")
def ephemeral_server() -> Iterator[str]:
    ready_event = threading.Event()
    url = None

    with subprocess.Popen(
        ["fal", "fn", "run", text_to_image.model.__file__, "MegaPipeline"],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    ) as process:

        def proxy_logs_thread() -> None:
            nonlocal url
            while process.poll() is None:
                try:
                    line = process.stdout.readline()  # type: ignore
                except ValueError:
                    break

                if "https://fal.run" in line:
                    ready_event.set()
                    url = line.strip().removesuffix("\x1b[0m").split()[-1]

                print(line, end="")

        thread = threading.Thread(target=proxy_logs_thread, daemon=True)
        thread.start()
        while process.poll() is None:
            if not thread.is_alive():
                thread.join()
                raise RuntimeError("Proxy logs thread died")

            if ready_event.wait(timeout=0.1):
                break
        else:
            raise RuntimeError("Server failed to start")

        if not url:
            raise RuntimeError("Server did not provide a URL")

        app_id = url.removeprefix("https://fal.run/")
        print(f"Connecting to app: {app_id}/health")
        for _ in range(MAX_TIME_TO_HEALTHY * 5):
            time.sleep(0.1)

            if process.poll() is not None:
                raise RuntimeError("Server died while waiting for app to be healthy")

            try:
                response = fal_client.run(f"{app_id}/health", arguments={})
            except httpx.HTTPError as exc:
                if exc.response.status_code == 500:
                    continue
                raise

            if response["status"] == "ok":
                break

        try:
            yield app_id
        finally:
            process.terminate()
            thread.join()


TEST_CASES = [
    {
        "name": "Stable Diffusion XL",
        "input": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "prompt": "Albert Einstein, wojak",
            "seed": 1463963878,
            "num_inference_steps": 10,
            "guidance_scale": 7.5,
            "scheduler": "Euler A",
            "num_images": 1,
        },
        "output": {"images": ["fal_file_storage/5207243b0f56408b9a332e0a48968210.png"]},
    },
    {
        "name": "Playground Isometric Astronaut Dog",
        "input": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "prompt": "single isometric black bernedoodle dressed as an American astronaut",
            "seed": 1477142,
            "image_size": {"width": 1024, "height": 1024},
            "clip_skip": 0,
            "loras": [
                {"path": "https://civitai.com/api/download/models/130580", "scale": 1}
            ],
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), (watermark), immature, child, (big nose:1.4)",
            "scheduler": "DPM++ 2M",
            "num_images": 1,
        },
        "output": {
            "images": ["fal_file_storage/bffa13e17f174984907cf9295e9adf5b.png"],
        },
    },
    {
        "name": "Playground Queen",
        "input": {
            "model_name": "emilianJR/epiCRealism",
            "prompt": "photo of a european medieval 40 year old queen, silver hair, highly detailed face, detailed eyes, head shot, intricate crown, age spots, wrinkles",
            "seed": 3088576288,
            "image_size": {"width": 544, "height": 960},
            "clip_skip": 0,
            "num_inference_steps": 40,
            "guidance_scale": 4,
            "negative_prompt": "ccartoon, painting, illustration, (worst quality, low quality, normal quality:2), (watermark), immature, child, (big nose:1.4)",
            "scheduler": "DPM++ 2M",
            "num_images": 1,
        },
        "output": {"images": ["fal_file_storage/7ae9d422aab44cc7bff9afbe32d154f3.png"]},
    },
    {
        "name": "Playground Red Mustang",
        "input": {
            "model_name": "emilianJR/epiCRealism",
            "prompt": "Photo of a classic red mustang car parked in las vegas strip at night",
            "seed": 2767081012,
            "image_size": {"width": 720, "height": 960},
            "clip_skip": 0,
            "num_inference_steps": 80,
            "guidance_scale": 5,
            "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
            "scheduler": "DPM++ 2M",
            "num_images": 1,
        },
        "output": {"images": ["fal_file_storage/6b4e820f33d74dd6ab758c11cd2cefa6.png"]},
    },
    {
        "name": "Playground Congolese Woman",
        "input": {
            "model_name": "emilianJR/epiCRealism",
            "prompt": "Photo of a congolese woman, wrinkles, aged, necklace, crowded, closeup, neutral colors, barren land",
            "seed": 803613448,
            "image_size": {"width": 512, "height": 768},
            "clip_skip": 0,
            "num_inference_steps": 60,
            "guidance_scale": 5,
            "negative_prompt": "asian, chinese, busty, (epicnegative:0.9)",
            "scheduler": "DPM++ 2M",
            "num_images": 1,
        },
        "output": {"images": ["fal_file_storage/60db9268799542bc987bef0ac56c894e.png"]},
    },
    {
        "name": "LCM x SDXL lora",
        "input": {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "prompt": "Self-portrait of a colorpoint british shorthair, 8k",
            "seed": 13077596251214904743,
            "num_inference_steps": 4,
            "guidance_scale": 0,
            "scheduler": "LCM",
            "loras": [
                {
                    "path": "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
                    "scale": 1,
                }
            ],
        },
        "output": {"images": ["fal_file_storage/1e57c30cedc04c9d89e3d4990b36c506.png"]},
    },
    {
        "name": "CN x IP Adapter x Lora",
        "input": {
            "loras": [
                {"path": "https://civitai.com/api/download/models/135931", "scale": 1}
            ],
            "prompt": "Photo of a european medieval 40 year old queen, silver hair, highly detailed face, detailed eyes, head shot, intricate crown, age spots, wrinkles",
            "embeddings": [
                {
                    "path": "https://storage.googleapis.com/falserverless/style_lora/emb_our_test_1.safetensors",
                    "tokens": ["<s0>", "<s1>"],
                }
            ],
            "image_size": "square_hd",
            "ip_adapter": [
                {
                    "path": "h94/IP-Adapter",
                    "scale": 1,
                    "weight_name": "ip-adapter-plus_sdxl_vit-h.safetensors",
                    "model_subfolder": "sdxl_models",
                    "ip_adapter_image_url": "https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/robot.jpeg",
                }
            ],
            "image_encoder_path": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "num_images": 1,
            "controlnets": [
                {
                    "path": "diffusers/controlnet-canny-sdxl-1.0",
                    "image_url": "https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/canny-edge.resized.jpg",
                    "end_percentage": 1,
                    "conditioning_scale": 1,
                }
            ],
            "image_format": "jpeg",
            "guidance_scale": 7.5,
            "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
            "num_inference_steps": 30,
            "enable_safety_checker": True,
            "seed": 6175561901514984000,
        },
        "output": {"images": ["fal_file_storage/test_8.png"]},
    },
    {
        "name": "CN x Lora",
        "input": {
            "loras": [
                {"path": "https://civitai.com/api/download/models/135931", "scale": 1}
            ],
            "prompt": "Photo of a european medieval 40 year old queen, silver hair, highly detailed face, detailed eyes, head shot, intricate crown, age spots, wrinkles",
            "embeddings": [
                {
                    "path": "https://storage.googleapis.com/falserverless/style_lora/emb_our_test_1.safetensors",
                    "tokens": ["<s0>", "<s1>"],
                }
            ],
            "image_size": "square_hd",
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "num_images": 1,
            "controlnets": [
                {
                    "path": "diffusers/controlnet-canny-sdxl-1.0",
                    "image_url": "https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/canny-edge.resized.jpg",
                    "end_percentage": 1,
                    "conditioning_scale": 1,
                }
            ],
            "image_format": "jpeg",
            "guidance_scale": 7.5,
            "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
            "num_inference_steps": 30,
            "enable_safety_checker": True,
            "seed": 6175561901514984000,
        },
        "output": {"images": ["fal_file_storage/test_9.png"]},
    },
    # {
    #     "name": "CN x Simple IP Adapter x Lora",
    #     "input": {
    #         "loras": [
    #             {"path": "https://civitai.com/api/download/models/135931", "scale": 1}
    #         ],
    #         "prompt": "Photo of a european medieval 40 year old queen, silver hair, highly detailed face, detailed eyes, head shot, intricate crown, age spots, wrinkles",
    #         "embeddings": [
    #             {
    #                 "path": "https://storage.googleapis.com/falserverless/style_lora/emb_our_test_1.safetensors",
    #                 "tokens": ["<s0>", "<s1>"],
    #             }
    #         ],
    #         "image_size": "square_hd",
    #         "ip_adapter": [
    #             {
    #                 "path": "h94/IP-Adapter",
    #                 "scale": 1,
    #                 "weight_name": "ip-adapter_sdxl.bin",
    #                 "model_subfolder": "sdxl_models",
    #                 "ip_adapter_image_url": "https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/robot.jpeg",
    #             }
    #         ],
    #         "num_images": 1,
    #         "image_format": "jpeg",
    #         "guidance_scale": 7.5,
    #         "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
    #         "num_inference_steps": 30,
    #         "enable_safety_checker": True,
    #         "seed": 6175561901514984000,
    #     },
    #     "output": {"images": ["fal_file_storage/test_12.jpg"]},
    # },
]


@pytest.fixture(scope="session")
def rest_client() -> httpx.Client:
    credentials = get_default_credentials()
    with httpx.Client(
        base_url=flags.REST_URL, headers=credentials.to_headers()
    ) as client:
        yield client


@pytest.fixture(scope="session")
def image_client() -> httpx.Client:
    with httpx.Client() as client:
        yield client


@pytest.mark.parametrize(
    "name, input, output",
    [
        (
            test["name"],
            test["input"],
            test["output"],
        )
        for test in TEST_CASES
    ],
)
def test_text_to_image(
    ephemeral_server: str,
    name: str,
    input: dict[str, Any],
    output: dict[str, Any],
    rest_client: httpx.Client,
    image_client: httpx.Client,
) -> None:
    print("started", name)
    print("output", output)
    result = fal_client.run(ephemeral_server, arguments=input)
    print("results", result)

    for generated_image, expected_image in zip(result["images"], output["images"]):
        response = rest_client.get(f"/storage/link/{expected_image}")
        response.raise_for_status()

        expected_image_url = response.json()["url"]
        similarity = compare_images(
            image_client,
            generated_image["url"],
            expected_image_url,
        )
        assert similarity >= SIMILARITY_THRESHOLD


def compare_images(
    client: httpx.Client, baseline_image: str, target_image: str
) -> float:
    with TemporaryDirectory() as directory:
        files = [f"{directory}/baseline.png", f"{directory}/target.png"]
        print("Comparing: ", baseline_image, target_image)
        for image_url, filename in zip([baseline_image, target_image], files):
            response = client.get(image_url)
            response.raise_for_status()
            with open(filename, "wb") as file:
                file.write(response.content)

        results = image_similarity(
            *files,
            metrics=[SIMILARITY_METHOD],
        )
        return results[SIMILARITY_METHOD]

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import fal_client
import httpx
import pytest

import text_to_image.model

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


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "prompt": "A cat",
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        },
        {
            "prompt": "A cat",
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "num_images": 2,
        },
        {
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "prompt": "Photo of a european medieval 40 year old queen, silver hair, highly detailed face, detailed eyes, head shot, intricate crown, age spots, wrinkles",
            "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
            "loras": [
                {"path": "https://civitai.com/api/download/models/135931", "scale": 1}
            ],
            "embeddings": [
                {
                    "path": "https://storage.googleapis.com/falserverless/style_lora/emb_our_test_1.safetensors",
                    "tokens": ["<s0>", "<s1>"],
                }
            ],
            "controlnets": [
                {
                    "path": "diffusers/controlnet-canny-sdxl-1.0",
                    "image_url": "https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/canny-edge.resized.jpg",
                    "conditioning_scale": 1,
                    "end_percentage": 1,
                }
            ],
            "ip_adapter": [
                {
                    "ip_adapter_image_url": "https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/robot.jpeg",
                    "path": "h94/IP-Adapter",
                    "model_subfolder": "sdxl_models",
                    "weight_name": "ip-adapter-plus_sdxl_vit-h.safetensors",
                    "scale": 1,
                }
            ],
            "image_encoder_path": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "image_encoder_weight_name": "pytorch_model.bin",
            "image_size": "square_hd",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "image_format": "jpeg",
            "num_images": 1,
            "tile_width": 4096,
            "tile_height": 4096,
            "tile_stride_width": 2048,
            "tile_stride_height": 2048,
        },
    ],
)
def test_text_to_image(ephemeral_server: str, inputs: dict[str, Any]) -> None:
    # TODO: we should run image similarity tests here
    result = fal_client.run(ephemeral_server, arguments=inputs)
    assert "images" in result

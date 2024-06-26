import gc
import os
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

from fal.toolkit import Image, download_model_weights
from fal.toolkit.file import FileRepository
from fal.toolkit.file.providers.gcp import GoogleStorageRepository
from fastapi import HTTPException
from pydantic import BaseModel, Field

from text_to_image.pipeline import create_pipeline


def download_or_hf_key(path: str) -> str:
    if path.startswith("https://") or path.startswith("http://"):
        return str(download_model_weights(path))
    return path


DeviceType = Literal["cpu", "cuda"]

TEMP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0"
)
ONE_MB = 1024**2
CHUNK_SIZE = 32 * ONE_MB
CACHE_PREFIX = ""

SUPPORTED_SCHEDULERS = {
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {}),
    "DPM++ 2M Karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "DPM++ 2M SDE": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++"},
    ),
    "DPM++ 2M SDE Karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "Euler": ("EulerDiscreteScheduler", {}),
    "Euler A": ("EulerAncestralDiscreteScheduler", {}),
    "LCM": ("LCMScheduler", {}),
}

# Amount of RAM to use as buffer, in percentages.
RAM_BUFFER_PERCENTAGE = 1 - 0.75


@dataclass
class Model:
    pipeline: object
    last_cache_hit: float = 0

    def as_base(self) -> object:
        self.last_cache_hit = time.monotonic()

        pipe = self.pipeline
        return pipe

    def device(self) -> DeviceType:
        return self.pipeline.device.type


class LoraWeight(BaseModel):
    path: str = Field(
        description="URL or the path to the LoRA weights.",
        examples=[
            "https://civitai.com/api/download/models/135931",
            "https://filebin.net/3chfqasxpqu21y8n/my-custom-lora-v1.safetensors",
        ],
    )
    scale: float = Field(
        default=1.0,
        description="""
            The scale of the LoRA weight. This is used to scale the LoRA weight
            before merging it with the base model.
        """,
        ge=0.0,
        le=1.0,
    )


class Embedding(BaseModel):
    path: str = Field(
        description="URL or the path to the embedding weights.",
        examples=[
            "https://storage.googleapis.com/falserverless/style_lora/emb_our_test_1.safetensors",
        ],
    )
    tokens: list[str] = Field(
        default=["<s0>", "<s1>"],
        description="""
            The tokens to map the embedding weights to. Use these tokens in your prompts.
        """,
    )


class ControlNet(BaseModel):
    path: str = Field(
        description="URL or the path to the control net weights.",
        examples=[
            "diffusers/controlnet-canny-sdxl-1.0",
        ],
    )
    image_url: str = Field(
        description="URL of the image to be used as the control net.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/canny-edge.resized.jpg",
        ],
    )
    conditioning_scale: float = Field(
        default=1.0,
        description="""
            The scale of the control net weight. This is used to scale the control net weight
            before merging it with the base model.
        """,
        ge=0.0,
        le=2.0,
    )
    start_percentage: float = Field(
        default=0.0,
        description="""
            The percentage of the image to start applying the controlnet in terms of the total timesteps.
        """,
        ge=0.0,
        le=1.0,
    )
    end_percentage: float = Field(
        default=1.0,
        description="""
            The percentage of the image to end applying the controlnet in terms of the total timesteps.
        """,
        ge=0.0,
        le=1.0,
    )


# make the ip adapter weight loader class
class IPAdapter(BaseModel):
    ip_adapter_image_url: str | None = Field(
        description="URL of the image to be used as the IP adapter.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/robot.jpeg",
        ],
    )
    path: str | None = Field(
        description="URL or the path to the IP adapter weights.",
        examples=[
            "h94/IP-Adapter",
        ],
    )
    model_subfolder: str | None = Field(
        description="Subfolder in the model directory where the IP adapter weights are stored.",
        examples=[
            "sdxl_models",
        ],
    )
    weight_name: str | None = Field(
        description="Name of the weight file.",
        examples=[
            "ip-adapter-plus_sdxl_vit-h.safetensors",
        ],
    )
    image_encoder_path: str | None = Field(
        description="URL or the path to the image encoder weights.",
        examples=[
            "h94/IP-Adapter",
        ],
    )
    image_encoder_subpath: str | None = Field(
        description="Subpath to the image encoder weights.",
        examples=[
            "models/image_encoder",
        ],
    )
    scale: float = Field(
        default=1.0,
        description="""
            The scale of the IP adapter weight. This is used to scale the IP adapter weight
            before merging it with the base model.
        """,
        ge=0.0,
    )


@dataclass
class GlobalRuntime:
    models: dict[tuple[str, ...], Model] = field(default_factory=dict)
    executor: ThreadPoolExecutor = field(default_factory=ThreadPoolExecutor)
    repository: str | FileRepository = "fal"

    def __post_init__(self):
        import torch
        from diffusers.pipelines.stable_diffusion.safety_checker import (
            StableDiffusionSafetyChecker,
        )
        from transformers import AutoFeatureExtractor

        if os.getenv("GCLOUD_SA_JSON"):
            self.repository = GoogleStorageRepository(
                url_expiration=2 * 24 * 60,  # 2 days, same as fal,
                bucket_name=os.getenv("GCS_BUCKET_NAME", "fal_file_storage"),
            )

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "CompVis/stable-diffusion-safety-checker",
            torch_dtype="float16",
        )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker",
            torch_dtype=torch.float16,
        ).to("cuda")

    def merge_and_apply_loras(
        self,
        pipe: object,
        loras: list[LoraWeight],
    ):
        print(f"LoRAs: {loras}")
        lora_paths = [download_or_hf_key(lora.path) for lora in loras]
        adapter_names = [
            Path(lora_path).name.replace(".", "_") for lora_path in lora_paths
        ]
        lora_scales = [lora_weight.scale for lora_weight in loras]

        for lora_path, lora_scale, adapter_name in zip(
            lora_paths, lora_scales, adapter_names
        ):
            print(f"Applying LoRA {lora_path} with scale {lora_scale}.")
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)

        pipe.set_adapters(adapter_names=adapter_names, adapter_weights=lora_scales)
        pipe.fuse_lora()

    def get_model(self, model_name: str, arch: str) -> Model:
        import torch
        from diffusers.pipelines.controlnet import MultiControlNetModel

        regular_pipeline_cls, sdxl_pipeline_cls = create_pipeline()

        model_key = (model_name, arch)
        if model_key not in self.models:
            if arch == "sdxl":
                pipeline_cls = sdxl_pipeline_cls
            else:
                pipeline_cls = regular_pipeline_cls

            if model_name.endswith(".ckpt") or model_name.endswith(".safetensors"):
                pipe = pipeline_cls.from_single_file(
                    model_name,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                )
            else:
                pipe = pipeline_cls.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    controlnet=MultiControlNetModel([]),
                )

            if hasattr(pipe, "watermark"):
                pipe.watermark = None

            if hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None

            self.models[model_key] = Model(pipe)

        return self.models[model_key]

    @contextmanager
    def add_ip_adapter(self, ip_adapter: IPAdapter | None, pipe) -> Iterator[None]:
        import torch
        from transformers import CLIPVisionModelWithProjection

        if not ip_adapter or not ip_adapter.path:
            yield
            return

        ip_adapter_huggingface_key = None
        ip_adapter_directory = None
        ip_adapter_name = None
        try:
            if ip_adapter.path.startswith("https://"):
                print("Assuming IP adapter path is a URL")
                ip_adapter_path = download_or_hf_key(ip_adapter.path)
                ip_adapter_directory = ip_adapter_path.parent
                ip_adapter_name = ip_adapter_path.name
            elif ip_adapter.path.startswith("http://"):
                # raise an error if the path is an http link
                raise HTTPException(
                    422,
                    detail="HTTP links are not supported for IP adapter weights. Please use HTTPS links or local paths.",
                )
            else:
                print("Assuming IP adapter path is a huggingface model")
                ip_adapter_huggingface_key = ip_adapter.path  # type: ignore
        except Exception as e:
            raise HTTPException(
                422,
                detail=f"Failed to download IP adapter: {e}",
            )

        # try downloading the image encoder weights if they are provided
        image_encoder_huggingface_key = None
        image_encoder_path = None
        if ip_adapter.image_encoder_path:
            try:
                if ip_adapter.image_encoder_path.startswith("https://"):
                    print("Assuming image encoder path is a URL")
                    image_encoder_path = download_model_weights(
                        ip_adapter.image_encoder_path,
                    )

                elif ip_adapter.image_encoder_path.startswith("http://"):
                    # raise an error if the path is an http link
                    raise HTTPException(
                        422,
                        detail="HTTP links are not supported for image encoder weights. Please use HTTPS links or local paths.",
                    )
                else:
                    print("Assuming image encoder path is a huggingface model")
                    image_encoder_huggingface_key = ip_adapter.image_encoder_path  # type: ignore
            except Exception as e:
                print(e)
                raise HTTPException(
                    422,
                    detail=f"Failed to load IP adapter: {e}",
                )

        old_image_encoder = pipe.image_encoder

        try:
            print("adding IP adapter to the pipe")
            if ip_adapter_huggingface_key:
                pipe.load_ip_adapter(
                    ip_adapter_huggingface_key,
                    subfolder=ip_adapter.model_subfolder,
                    weight_name=ip_adapter.weight_name,
                )
            elif ip_adapter_directory:
                pipe.load_ip_adapter(ip_adapter_directory, weight_name=ip_adapter_name)
            else:
                raise HTTPException(
                    500,
                    detail="IP adapter path or name was not found. This should be impossible?!",
                )

            if image_encoder_huggingface_key:
                encoder = CLIPVisionModelWithProjection.from_pretrained(
                    image_encoder_huggingface_key,
                    subfolder=ip_adapter.image_encoder_subpath,
                    torch_dtype=torch.float16,
                )
                pipe.image_encoder = self.execute_on_cuda(
                    partial(encoder.to, "cuda"), ignored_models=[pipe]
                )
            elif image_encoder_path:
                raise NotImplementedError(
                    "Loading image encoder weights from a local path is not supported yet."
                )

            pipe.set_ip_adapter_scale(ip_adapter.scale)

            yield

        finally:
            pipe.unload_ip_adapter()
            pipe.image_encoder = old_image_encoder

    @contextmanager
    def add_controlnets(self, controlnets: list[ControlNet], pipe) -> Iterator[None]:
        import torch
        from diffusers import ControlNetModel
        from diffusers.pipelines.controlnet import MultiControlNetModel

        if not controlnets:
            yield
            return

        controlnet_paths = []
        for controlnet in controlnets:
            try:
                # see if you can parse the controlnet path as a URL
                if controlnet.path.startswith("https://"):
                    print("Assuming controlnet path is a URL")
                    controlnet_path = download_model_weights(
                        controlnet.path,
                    )
                    controlnet_paths.append(Path(controlnet_path))
                elif controlnet.path.startswith("http://"):
                    # raise an error it needs to be https
                    raise HTTPException(
                        422,
                        detail="Controlnet path needs to be an HTTPS URL",
                    )
                else:
                    print("Assuming controlnet path is a huggingface model")
                    controlnet_paths.append(controlnet.path)  # type: ignore
            except Exception as e:
                raise HTTPException(
                    422,
                    detail=f"Failed to download controlnet: {e}",
                )

        controlnet_models = []
        try:
            for controlnet_path in controlnet_paths:
                if isinstance(controlnet_path, Path):
                    print("Loading controlnet from path", controlnet_path)
                    controlnet_model = ControlNetModel.from_single_file(
                        controlnet_path,
                        torch_dtype=torch.float16,
                    )
                else:
                    print("loading from huggingface model", controlnet_path)
                    controlnet_model = ControlNetModel.from_pretrained(
                        controlnet_path,
                        torch_dtype=torch.float16,
                    )

                controlnet_models.append(
                    self.execute_on_cuda(
                        partial(controlnet_model.to, "cuda"), ignored_models=[pipe]
                    )
                )

            print(
                "adding controlnets to the pipe, controlnet_models len",
                len(controlnet_models),
            )

            pipe.controlnet = MultiControlNetModel(controlnet_models)

            yield
        except Exception as e:
            print(e)
            raise HTTPException(
                422,
                detail=f"Failed to load controlnet: {e}",
            )

        finally:
            pipe.controlnet = MultiControlNetModel([])
            for controlnet_model in controlnet_models:
                if controlnet_model is not None:
                    controlnet_model.cpu()
                    del controlnet_model

            del controlnet_models
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    @contextmanager
    def add_embeddings(self, embeddings: list[Embedding], pipe, arch) -> Iterator[None]:
        from safetensors.torch import load_file

        if not embeddings:
            yield
            return
        elif len(embeddings) > 1:
            raise HTTPException(
                422,
                detail="Only one embedding is supported at the moment.",
            )

        [embedding] = embeddings
        try:
            embedding_path = download_or_hf_key(embedding.path)
        except Exception as e:
            raise HTTPException(
                422,
                detail=f"Failed to download embedding: {e}",
            )

        try:
            embedding_state_dict = load_file(embedding_path)
        except Exception as e:
            raise HTTPException(
                422,
                detail=f"Failed to load LoRA weight: {e}",
            )

        textual_inversions = []

        try:
            print(embedding_state_dict.keys())

            if "clip_l" in embedding_state_dict:
                text_encoder_key = "clip_l"
            elif "text_encoders_0" in embedding_state_dict:
                text_encoder_key = "text_encoders_0"
            elif "text_encoder" in embedding_state_dict:
                text_encoder_key = "text_encoder"
            else:
                raise HTTPException(
                    422,
                    detail="Invalid embedding state dict. Needs to have a key that is either 'clip_l', 'text_encoders_0', or 'text_encoder'",
                )

            if arch == "sdxl":
                if "clip_g" in embedding_state_dict:
                    text_encoder_2_key = "clip_g"
                elif "text_encoders_1" in embedding_state_dict:
                    text_encoder_2_key = "text_encoders_1"
                else:
                    raise HTTPException(
                        422,
                        detail="Invalid embedding state dict. Needs to have a key that is either 'clip_g' or 'text_encoders_1'",
                    )

            if arch == "sdxl":
                encoders = [
                    (text_encoder_key, pipe.text_encoder, pipe.tokenizer),
                    (text_encoder_2_key, pipe.text_encoder_2, pipe.tokenizer_2),
                ]
            else:
                encoders = [
                    (text_encoder_key, pipe.text_encoder, pipe.tokenizer),
                ]

            for te_key, text_encoder_ref, tokenizer in encoders:
                if te_key in embedding_state_dict:
                    print(
                        f"Loading textual inversion for {te_key} with {embedding.tokens}"
                    )
                    textual_inversions.append(
                        (embedding.tokens, text_encoder_ref, tokenizer)
                    )
                    pipe.load_textual_inversion(
                        embedding_state_dict[te_key],
                        token=embedding.tokens,
                        text_encoder=text_encoder_ref,
                        tokenizer=tokenizer,
                    )

            yield
        finally:
            for tokens, text_encoder, tokenizer in textual_inversions:
                pipe.unload_textual_inversion(
                    tokens=tokens, text_encoder=text_encoder, tokenizer=tokenizer
                )

    @contextmanager
    def load_model(
        self,
        model_name: str,
        loras: list[LoraWeight],
        embeddings: list[Embedding],
        controlnets: list[ControlNet],
        ip_adapter: IPAdapter | None,
        clip_skip: int = 0,
        scheduler: str | None = None,
        model_architecture: str | None = None,
    ) -> Iterator[object | None]:
        model_name = download_or_hf_key(model_name)

        if model_architecture is None:
            if "xl" in model_name.lower():
                arch = "sdxl"
            else:
                arch = "sd"
            print(
                f"Guessing {arch} architecture for {model_name}. If this is wrong, "
                "please specify it as part of the model_architecture parameter."
            )
        else:
            arch = model_architecture

        model = self.get_model(str(model_name), arch=arch)
        pipe = model.as_base()
        pipe = self.execute_on_cuda(partial(pipe.to, "cuda"))

        if clip_skip:
            print(f"Ignoring clip_skip={clip_skip} for now, it's not supported yet!")

        with self.change_scheduler(pipe, scheduler):
            with self.add_embeddings(embeddings, pipe, arch):
                with self.add_controlnets(controlnets, pipe):
                    with self.add_ip_adapter(ip_adapter, pipe):
                        try:
                            if loras:
                                self.merge_and_apply_loras(pipe, loras)

                            yield pipe
                        finally:
                            if loras:
                                try:
                                    pipe.unfuse_lora()
                                    pipe.set_adapters(adapter_names=[])
                                except Exception:
                                    print(
                                        "Failed to unfuse LoRAs from the pipe, clearing it out of memory."
                                    )
                                    traceback.print_exc()
                                    self.models.pop((model_name, arch), None)
                                else:
                                    pipe.unload_lora_weights()

    @contextmanager
    def change_scheduler(
        self, pipe: object, scheduler_name: str | None = None
    ) -> Iterator[None]:
        import diffusers

        if scheduler_name is None:
            yield
            return

        scheduler_cls_name, scheduler_kwargs = SUPPORTED_SCHEDULERS[scheduler_name]
        scheduler_cls = getattr(diffusers, scheduler_cls_name)
        if (
            scheduler_cls not in pipe.scheduler.compatibles
            # Apparently LCM doesn't get reported as a compatible scheduler even
            # though it works.
            and scheduler_cls_name != "LCMScheduler"
        ):
            compatibles = ", ".join(cls.__name__ for cls in pipe.scheduler.compatibles)
            raise ValueError(
                f"The scheduler {scheduler_name} is not compatible with this model.\n"
                f"Compatible schedulers: {compatibles}"
            )

        original_scheduler = pipe.scheduler
        try:
            pipe.scheduler = scheduler_cls.from_config(
                pipe.scheduler.config,
                **scheduler_kwargs,
            )
            yield
        finally:
            pipe.scheduler = original_scheduler

    def upload_images(self, images: list[object]) -> list[Image]:
        print("Uploading images...")
        image_uploader = partial(Image.from_pil, repository=self.repository)
        res = list(self.executor.map(image_uploader, images))
        print("Done uploading images.")
        return res

    def execute_on_cuda(
        self,
        function: Callable[..., Any],
        *,
        ignored_models: list[object] | None = None,
    ):
        cached_models = self.get_loaded_models_by_device(
            "cuda",
            ignored_models=ignored_models or [],
        )

        first_try = True
        while first_try or cached_models:
            first_try = False

            try:
                return function()
            except RuntimeError as error:
                # Only retry if the error is a CUDA OOM error.
                # https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py#L19
                __cuda_oom_errors = [
                    "CUDA out of memory",
                    "INTERNAL ASSERT FAILED",
                ]
                if (
                    not any(error_str in str(error) for error_str in __cuda_oom_errors)
                    or not cached_models
                ):
                    raise

                # Since cached_models is sorted by last cache hit, we'll pop the the
                # model with the oldest cache hit and try again.
                target_model_id = cached_models.pop()
                self.offload_model_to_cpu(target_model_id)
                self.empty_cache()

        self.empty_cache()
        raise RuntimeError("Not enough CUDA memory to complete the operation.")

    def get_loaded_models_by_device(
        self,
        device: DeviceType,
        ignored_models: list[object],
    ):
        models = [
            model_id
            for model_id in self.models
            if self.models[model_id].device() == device
            if self.models[model_id].pipeline not in ignored_models
        ]
        models.sort(key=lambda model_id: self.models[model_id].last_cache_hit)
        return models

    def offload_model_to_cpu(self, model_id: tuple[str, ...]):
        print(f"Offloading model={model_id} to CPU.")

        def is_ram_buffer_full():
            import psutil

            memory = psutil.virtual_memory()
            percent_available = memory.available / memory.total
            return percent_available < RAM_BUFFER_PERCENTAGE

        models = self.get_loaded_models_by_device("cpu", ignored_models=[])
        while is_ram_buffer_full():
            if not models:
                print(
                    f"Not enough RAM to offload the model to CPU, evicting {model_id}"
                    "it directly."
                )
                del self.models[model_id]
                gc.collect()
                return

            lru_model_id = models.pop()
            print(f"Offloading model={lru_model_id} back to disk.")
            del self.models[lru_model_id]
            gc.collect()

        model = self.models[model_id]
        model.pipeline = model.pipeline.to("cpu")

    def empty_cache(self):
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def run_safety_checker(self, images: list[object], enable_safety_checker: bool):
        if not enable_safety_checker:
            return [False] * len(images)

        import numpy as np
        import torch

        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(
            "cuda"
        )

        np_image = [np.array(val) for val in images]

        _, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )

        return has_nsfw_concept


def filter_by(
    has_nsfw_concepts: list[bool],
    images: list[object],
) -> list[object]:
    from PIL import Image as PILImage

    return [
        (
            PILImage.new("RGB", (image.width, image.height), (0, 0, 0))
            if has_nsfw
            else image
        )
        for image, has_nsfw in zip(images, has_nsfw_concepts)
    ]

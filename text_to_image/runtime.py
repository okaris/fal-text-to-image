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


# make the ip adapter weight loader class
class IPAdapter(BaseModel):
    ip_adapter_image_url: str | list[str] = Field(
        description="URL of the image to be used as the IP adapter.",
        examples=[
            "https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/robot.jpeg",
        ],
    )
    ip_adapter_mask_url: str | None = Field(
        default=None,
        description="""
            The mask to use for the IP adapter. When using a mask, the ip-adapter image size and the mask size must be the same
        """,
    )
    path: str = Field(
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
    insight_face_model_path: str | None = Field(
        description="URL or the path to the InsightFace model weights.",
        examples=[
            "h94/IP-Adapter",
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
    scale_json: dict | None = Field(
        description="""
            The scale of the IP adapter weight. This is used to scale the IP adapter weight
            before merging it with the base model.
        """,
    )
    unconditional_noising_factor: float = Field(
        default=0.0,
        description="""The factor to apply to the unconditional noising of the IP adapter.""",
        ge=0.0,
        le=1.0,
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
    mask_url: str | None = Field(
        default=None,
        description="""
            The mask to use for the controlnet. When using a mask, the control image size and the mask size must be the same and divisible by 32.
        """,
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
    ip_adapter_index: int = Field(
        default=None,
        description="""
            The index of the IP adapter to be applied to the controlnet. This is only needed for InstantID ControlNets.
        """,
    )


def download_or_hf_key(path: str) -> str:
    if path.startswith("https://") or path.startswith("http://"):
        return str(download_model_weights(path))
    return path


INSIGHTFACE_MODEL_CACHE_PATH = Path("/data")
INSIGHTFACE_MODEL_DOWNLOAD_PATH = "/data/models"

DeviceType = Literal["cpu", "cuda"]

TEMP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0"
)
ONE_MB = 1024**2
CHUNK_SIZE = 32 * ONE_MB
CACHE_PREFIX = ""

SUPPORTED_SCHEDULERS = {
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {"final_sigmas_type": "zero"}),
    "DPM++ 2M Karras": (
        "DPMSolverMultistepScheduler",
        {"use_karras_sigmas": True, "final_sigmas_type": "zero"},
    ),
    "DPM++ 2M SDE": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "final_sigmas_type": "zero"},
    ),
    "DPM++ 2M SDE Karras": (
        "DPMSolverMultistepScheduler",
        {
            "algorithm_type": "sde-dpmsolver++",
            "use_karras_sigmas": True,
            "final_sigmas_type": "zero",
        },
    ),
    "Euler": ("EulerDiscreteScheduler", {}),
    "Euler A": ("EulerAncestralDiscreteScheduler", {}),
    "LCM": ("LCMScheduler", {}),
    "DDIM": ("DDIMScheduler", {}),
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


def patch_onnx_runtime(
    inter_op_num_threads: int = 16,
    intra_op_num_threads: int = 16,
    omp_num_threads: int = 16,
):
    import os

    import onnxruntime as ort

    os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    _default_session_options = ort.capi._pybind_state.get_default_session_options()

    def get_default_session_options_new():
        _default_session_options.inter_op_num_threads = inter_op_num_threads
        _default_session_options.intra_op_num_threads = intra_op_num_threads
        return _default_session_options

    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new


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

        patch_onnx_runtime()

        if os.getenv("GCLOUD_SA_JSON"):
            self.repository = GoogleStorageRepository(
                url_expiration=2 * 24 * 60,  # 2 days, same as fal,
                bucket_name=os.getenv("GCS_BUCKET_NAME", "fal_file_storage"),
            )

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "CompVis/stable-diffusion-safety-checker",
            torch_dtype=torch.float16,
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

    def get_model(self, model_name: str, arch: str, variant: str | None) -> Model:
        import torch
        from diffusers.pipelines.controlnet import MultiControlNetModel

        regular_pipeline_cls, sdxl_pipeline_cls = create_pipeline()

        model_key = (model_name, arch)
        if model_key not in self.models:
            print(f"Loading model {model_name}...")
            if arch == "sdxl":
                pipeline_cls = sdxl_pipeline_cls
            else:
                pipeline_cls = regular_pipeline_cls

            if model_name.endswith(".ckpt") or model_name.endswith(".safetensors"):
                pipe = pipeline_cls.from_single_file(
                    model_name,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                    variant=variant,
                )
            else:
                pipe = pipeline_cls.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    controlnet=MultiControlNetModel([]),
                    variant=variant,
                )

            if hasattr(pipe, "watermark"):
                pipe.watermark = None

            if hasattr(pipe, "safety_checker"):
                pipe.safety_checker = None

            self.models[model_key] = Model(pipe)
        else:
            print(f"Model {model_name} is already loaded, reusing it.")

        return self.models[model_key]

    def load_ip_adapter(
        self,
        pipeline,
        directory,
        weight_name,
        scales,
    ):
        args = []
        kwargs = {}
        load_fn = pipeline.load_ip_adapter

        # assert all the files exist
        for path, w in zip(directory, weight_name):
            local_path = Path(path) / Path(w)
            print(f"Checking if {local_path} exists")
            assert Path(local_path).exists(), f"Path {path} does not exist."

        args = [directory]
        kwargs = {
            "weight_name": weight_name,
            "scale": scales,
            "subfolder": ["./"] * len(directory),
            "image_encoder_folder": None,
        }

        # `load_ip_adapter` loads it to the pipeline's device
        # https://github.com/huggingface/diffusers/commit/065f251766d5fa307f18814bfbb862f180755fe1
        self.execute_on_cuda(
            partial(load_fn, *args, **kwargs),
            ignored_models=[pipeline],
        )

    @contextmanager
    def add_ip_adapter(
        self,
        ip_adapters: list[IPAdapter] | None,
        image_encoder_path,
        image_encoder_subfolder,
        image_encoder_weight_name,
        pipe,
    ) -> Iterator[None]:
        import torch
        from huggingface_hub import snapshot_download
        from insightface.app import FaceAnalysis
        from transformers import CLIPVisionModelWithProjection

        if not ip_adapters or len(ip_adapters) == 0:
            yield
            return

        ip_adapter_local_folders: list[str | None] = []
        ip_adapter_local_names: list[str | None] = []
        ip_adapter_scales: list[float | dict] = []
        image_encoder_local_folders: list[str | None] = []
        image_encoder_local_names: list[str | None] = []

        image_encoder_local_folder = None
        image_encoder_local_name = None

        if image_encoder_path:
            try:
                if image_encoder_path.startswith("https://"):
                    print("Assuming image encoder path is a URL")
                    image_encoder_local_path = download_model_weights(
                        image_encoder_path,
                    )
                    image_encoder_local_folder = image_encoder_local_path.parent
                    image_encoder_local_name = image_encoder_local_path.name

                elif image_encoder_path.startswith("http://"):
                    # raise an error if the path is an http link
                    raise HTTPException(
                        422,
                        detail="HTTP links are not supported for image encoder weights. Please use HTTPS links or local paths.",
                    )
                else:
                    print("Assuming image encoder path is a huggingface model")
                    image_encoder_local_folder = snapshot_download(
                        image_encoder_path,
                    )

                    if image_encoder_subfolder:
                        image_encoder_local_folder = image_encoder_local_folder / Path(
                            image_encoder_subfolder
                        )

                    image_encoder_local_name = image_encoder_weight_name
            except Exception as e:
                print(e)
                raise HTTPException(
                    422,
                    detail=f"Failed to load IP adapter: {e}",
                )

        for ip_adapter in ip_adapters:
            ip_adapter_local_folder = None
            ip_adapter_local_name = None

            if ip_adapter.path is None:
                raise HTTPException(
                    422,
                    detail="IP adapter path is required.",
                )

            try:
                if ip_adapter.path.startswith("https://"):
                    print("Assuming IP adapter path is a URL")
                    ip_adapter_path = download_model_weights(ip_adapter.path)
                    ip_adapter_local_folder = ip_adapter_path.parent
                    ip_adapter_local_name = ip_adapter_path.name
                elif ip_adapter.path.startswith("http://"):
                    # raise an error if the path is an http link
                    raise HTTPException(
                        422,
                        detail="HTTP links are not supported for IP adapter weights. Please use HTTPS links or local paths.",
                    )
                else:
                    print("Assuming IP adapter path is a huggingface model")
                    ip_adapter_local_folder = snapshot_download(ip_adapter.path)

                    if ip_adapter.model_subfolder:
                        ip_adapter_local_folder = ip_adapter_local_folder / Path(
                            ip_adapter.model_subfolder
                        )

                    ip_adapter_local_name = ip_adapter.weight_name
            except Exception as e:
                raise HTTPException(
                    422,
                    detail=f"Failed to download IP adapter: {e}",
                )

            # try to download the insightface model if specified

            insightface_model_name = None
            try:
                if ip_adapter.insight_face_model_path:
                    # check if it is a url
                    if ip_adapter.insight_face_model_path.startswith("https://"):
                        print("Assuming insightface model path is a URL")
                        insightface_path = download_model_weights(
                            ip_adapter.insight_face_model_path,
                        )
                        insightface_model_dir = Path(insightface_path).parent
                        insightface_model_name = Path(insightface_path).name
                    elif ip_adapter.insight_face_model_path.startswith("http://"):
                        # raise an error if the path is an http link
                        raise HTTPException(
                            422,
                            detail="HTTP links are not supported for insightface model weights. Please use HTTPS links or local paths.",
                        )
                    # see if there is a single forward slash in the path
                    elif ip_adapter.insight_face_model_path.count("/") == 1:
                        insightface_model_name = (
                            ip_adapter.insight_face_model_path.split("/")[-1]
                        )
                        insightface_download_dir = f"{INSIGHTFACE_MODEL_DOWNLOAD_PATH}/{insightface_model_name}"
                        snapshot_download(
                            ip_adapter.insight_face_model_path,
                            local_dir=insightface_download_dir,
                        )

                        insightface_model_dir = INSIGHTFACE_MODEL_CACHE_PATH
                    else:
                        # assume it is a model name
                        insightface_model_name = ip_adapter.insight_face_model_path
                        insightface_model_dir = INSIGHTFACE_MODEL_CACHE_PATH

            except Exception as e:
                raise HTTPException(
                    422,
                    detail=f"Failed to download insightface model: {e}",
                )

            if insightface_model_name:
                app = FaceAnalysis(
                    name=insightface_model_name,
                    root=insightface_model_dir,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(640, 640))
                pipe.face_analysis = app

            ip_adapter_scale: float | dict | None = None
            if ip_adapter.scale_json is not None:
                ip_adapter_scale = ip_adapter.scale_json
            elif ip_adapter.scale:
                ip_adapter_scale = ip_adapter.scale
            else:
                raise HTTPException(
                    422,
                    detail="IP adapter scale or scale_json is required.",
                )

            ip_adapter_local_folders.append(
                ip_adapter_local_folder,
            )

            ip_adapter_local_names.append(ip_adapter_local_name)
            ip_adapter_scales.append(ip_adapter_scale)

        image_encoder_local_folders.append(image_encoder_local_folder)
        image_encoder_local_names.append(image_encoder_local_name)

        ip_adapter_data = [
            ip_adapter_local_folders,
            ip_adapter_local_names,
            ip_adapter_scales,
        ]

        if not all(len(arr) == len(ip_adapters) for arr in ip_adapter_data):  # type: ignore # mypy bug
            error_details = (
                f"IP Adapter count and array counts must match. Found {len(ip_adapters)} IP Adapters, "
                f"{len(ip_adapter_local_folders)} huggingface keys or paths, "
                f"{len(ip_adapter_local_names)} names, {len(ip_adapter_scales)} scales, "
                f"{len(image_encoder_local_folders)} image encoder paths, "
                f"and {len(image_encoder_local_names)} image encoder names."
            )
            raise HTTPException(422, error_details)

        old_image_encoder = pipe.image_encoder

        try:
            print("adding IP adapter to the pipe")
            self.load_ip_adapter(
                pipe,
                ip_adapter_local_folders,
                ip_adapter_local_names,
                ip_adapter_scales,
            )

            image_encoder_local_path = Path(str(image_encoder_local_folder)) / Path(
                str(image_encoder_local_name)
            )  # type: ignore
            print("Loading image encoder from path", image_encoder_local_path)
            assert (
                image_encoder_local_path.exists()
            ), f"Image encoder path {image_encoder_local_path} does not exist."

            encoder = CLIPVisionModelWithProjection.from_pretrained(
                image_encoder_local_path.parent,
                torch_dtype=torch.float16,
                local_files_only=True,
                force_download=False,
            )

            pipe.image_encoder = self.execute_on_cuda(
                partial(encoder.to, "cuda"), ignored_models=[pipe]
            )

            pipe.set_ip_adapter_scale(ip_adapter_scales)

            yield

        finally:
            import gc

            new_image_encoder = pipe.image_encoder
            if new_image_encoder is not None:
                new_image_encoder.cpu()
            pipe.unload_ip_adapter()

            pipe.image_encoder = old_image_encoder
            if new_image_encoder is not None:
                del new_image_encoder

            gc.collect()
            torch.cuda.empty_cache()

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
        ip_adapter: list[IPAdapter] | None = None,
        image_encoder_path: str | None = None,
        image_encoder_subfolder: str | None = None,
        image_encoder_weight_name: str | None = None,
        clip_skip: int = 0,
        scheduler: str | None = None,
        model_architecture: str | None = None,
        variant: str | None = None,
    ) -> Iterator[object | None]:
        import torch

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

        model = self.get_model(str(model_name), arch=arch, variant=variant)
        pipe = model.as_base()
        pipe = self.execute_on_cuda(partial(pipe.to, "cuda", dtype=torch.float16))

        if clip_skip:
            print(f"Ignoring clip_skip={clip_skip} for now, it's not supported yet!")

        with self.change_scheduler(pipe, scheduler):
            with self.add_embeddings(embeddings, pipe, arch):
                with self.add_controlnets(controlnets, pipe):
                    with self.add_ip_adapter(
                        ip_adapter,
                        image_encoder_path=image_encoder_path,
                        image_encoder_subfolder=image_encoder_subfolder,
                        image_encoder_weight_name=image_encoder_weight_name,
                        pipe=pipe,
                    ):
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
                    print(f"Not a CUDA OOM error, re-raising the error: {error}")
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
        models.sort(
            key=lambda model_id: self.models[model_id].last_cache_hit, reverse=True
        )
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
        # do the parts individually to avoid the safety checker
        # from being offloaded to CPU
        model.pipeline.unet = model.pipeline.unet.to("cpu")
        model.pipeline.vae = model.pipeline.vae.to("cpu")
        model.pipeline.text_encoder = model.pipeline.text_encoder.to("cpu")
        model.pipeline.text_encoder_2 = model.pipeline.text_encoder_2.to("cpu")

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

        image_tensors = self.feature_extractor(images, return_tensors="pt")
        safety_checker_input = self.execute_on_cuda(partial(image_tensors.to, "cuda"))

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

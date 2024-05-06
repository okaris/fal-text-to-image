from contextlib import contextmanager
from functools import lru_cache, partial
from typing import Any, ClassVar, Literal
from urllib.request import Request, urlopen

import fal
import PIL.Image
from fal import cached
from fal.toolkit import File, Image, ImageSizeInput, get_image_size
from fal.toolkit.image import ImageSize
from pydantic import BaseModel, Field, root_validator

from text_to_image.runtime import (
    SUPPORTED_SCHEDULERS,
    ControlNet,
    Embedding,
    GlobalRuntime,
    IPAdapter,
    LoraWeight,
    filter_by,
)


class OrderedBaseModel(BaseModel):
    SCHEMA_IGNORES: ClassVar[set[str]] = set()
    FIELD_ORDERS: ClassVar[list[str]] = []

    class Config:
        @staticmethod
        def schema_extra(schema: dict[str, Any], model: type) -> None:
            # Remove the model_name and scheduler fields from the schema
            for key in model.SCHEMA_IGNORES:
                schema["properties"].pop(key, None)

            # Reorder the fields to make sure FIELD_ORDERS are accurate,
            # any missing fields will be appearing at the end of the schema.
            properties = {}
            for field in model.FIELD_ORDERS:
                if props := schema["properties"].pop(field, None):
                    properties[field] = props

            schema["properties"] = {**properties, **schema["properties"]}


@cached
def load_session():
    return GlobalRuntime()


def invalid_data_error(field_name: list, message: str):
    from fastapi import HTTPException

    details = [{"loc": ["body"] + field_name, "msg": message, "type": "value_error"}]

    return HTTPException(status_code=422, detail=details)


@lru_cache(maxsize=64)
def read_image_from_url(url: str):
    import PIL.Image
    from fastapi import HTTPException

    try:
        with urlopen(
            Request(
                url,
                headers={
                    "User-Agent": "fal.ai/1.0",
                },
            )
        ) as response:
            image = PIL.Image.open(response).convert("RGB")
    except:
        import traceback

        traceback.print_exc()
        raise HTTPException(422, f"Could not load image from url: {url}")

    return image


def create_empty_mask_for_image(image: PIL.Image.Image) -> PIL.Image.Image:
    import numpy as np

    return PIL.Image.fromarray(np.ones_like(np.array(image)) * 255)


class InputParameters(OrderedBaseModel):
    model_name: str = Field(
        description="URL or HuggingFace ID of the base model to generate the image.",
        examples=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "SG161222/Realistic_Vision_V2.0",
        ],
    )
    variant: str | None = Field(
        default=None,
        description="The variant of the model to use for huggingface models, e.g. 'fp16'.",
    )
    prompt: str = Field(
        description="The prompt to use for generating the image. Be as descriptive as possible for best results.",
        examples=[
            "Photo of a european medieval 40 year old queen, silver hair, highly detailed face, detailed eyes, head shot, intricate crown, age spots, wrinkles",
            "Photo of a classic red mustang car parked in las vegas strip at night",
        ],
    )
    negative_prompt: str = Field(
        default="",
        description="""
            The negative prompt to use.Use it to address details that you don't want
            in the image. This could be colors, objects, scenery and even the small details
            (e.g. moustache, blurry, low resolution).
        """,
        examples=[
            "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
            "nsfw, cartoon, (epicnegative:0.9)",
        ],
    )
    image_url: str | None = Field(
        default=None,
        description="URL of image to use for image to image.",
    )
    noise_strength: float = Field(
        default=0.5,
        description="The amount of noise to add to noise image for image. Only used if the image_url is provided. 1.0 is complete noise and 0 is no noise.",
        ge=0.0,
        le=1.0,
    )
    loras: list[LoraWeight] = Field(
        default_factory=list,
        description="""
            The LoRAs to use for the image generation. You can use any number of LoRAs
            and they will be merged together to generate the final image.
        """,
    )
    embeddings: list[Embedding] = Field(
        default_factory=list,
        description="""
            The embeddings to use for the image generation. Only a single embedding is supported at the moment.
            The embeddings will be used to map the tokens in the prompt to the embedding weights.
        """,
    )
    controlnets: list[ControlNet] = Field(
        default_factory=list,
        description="""
            The control nets to use for the image generation. You can use any number of control nets
            and they will be applied to the image at the specified timesteps.
        """,
    )
    controlnet_guess_mode: bool = Field(
        default=False,
        description="""
            If set to true, the controlnet will be applied to only the conditional predictions.
        """,
    )
    ip_adapter: list[IPAdapter] = Field(
        default_factory=list,
        description="""
            The IP adapter to use for the image generation.
        """,
    )
    image_encoder_path: str | None = Field(
        description="""
            The path to the image encoder model to use for the image generation.
        """,
        default=None,
        examples=[
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        ],
    )
    image_encoder_subfolder: str | None = Field(
        description="""
            The subfolder of the image encoder model to use for the image generation.
        """,
        default=None,
        examples=[],
    )
    image_encoder_weight_name: str | None = Field(
        description="""
            The weight name of the image encoder model to use for the image generation.
        """,
        default="pytorch_model.bin",
        examples=[
            "pytorch_model.bin",
        ],
    )
    seed: int | None = Field(
        default=None,
        description="""
            The same seed and the same prompt given to the same version of Stable Diffusion
            will output the same image every time.
        """,
    )
    image_size: ImageSizeInput | None = Field(
        default="square_hd",
        description="""
            The size of the generated image. You can choose between some presets or custom height and width
            that **must be multiples of 8**.
        """,
    )
    num_inference_steps: int = Field(
        default=30,
        description="""
            Increasing the amount of steps tells Stable Diffusion that it should take more steps
            to generate your final result which can increase the amount of detail in your image.
        """,
        ge=0,
        le=150,
        title="Number of inference steps",
    )
    guidance_scale: float = Field(
        default=7.5,
        description="""
            The CFG (Classifier Free Guidance) scale is a measure of how close you want
            the model to stick to your prompt when looking for a related image to show you.
        """,
        ge=0.0,
        le=20.0,
        title="Guidance scale (CFG)",
    )
    clip_skip: int = Field(
        default=0,
        description="""
            Skips part of the image generation process, leading to slightly different results.
            This means the image renders faster, too.
        """,
        ge=0,
        le=2,
    )
    model_architecture: Literal["sd", "sdxl"] | None = Field(
        default=None,
        description=(
            "The architecture of the model to use. If an HF model is used, it will be automatically detected. Otherwise will assume depending on "
            "the model name (whether XL is in the name or not)."
        ),
    )
    scheduler: Literal._getitem(Literal, *SUPPORTED_SCHEDULERS) | None = Field(  # type: ignore
        default=None,
        description="Scheduler / sampler to use for the image denoising process.",
    )
    image_format: Literal["jpeg", "png"] = Field(
        default="png",
        description="The format of the generated image.",
        examples=["jpeg"],
    )
    num_images: int = Field(
        default=1,
        description="""
            Number of images to generate in one request. Note that the higher the batch size,
            the longer it will take to generate the images.
        """,
        ge=1,
        le=8,
        title="Number of images",
    )
    enable_safety_checker: bool = Field(
        default=False,
        description="If set to true, the safety checker will be enabled.",
    )
    tile_width: int = Field(
        default=4096,
        description="The size of the tiles to be used for the image generation.",
        ge=128,
        le=4096,
    )
    tile_height: int = Field(
        default=4096,
        description="The size of the tiles to be used for the image generation.",
        ge=128,
        le=4096,
    )
    tile_stride_width: int = Field(
        default=2048,
        description="The stride of the tiles to be used for the image generation.",
        ge=64,
        le=2048,
    )
    tile_stride_height: int = Field(
        default=2048,
        description="The stride of the tiles to be used for the image generation.",
        ge=64,
        le=2048,
    )
    debug_latents: bool = Field(
        default=False,
        description="If set to true, the latents will be saved for debugging.",
    )
    debug_per_pass_latents: bool = Field(
        default=False,
        description="If set to true, the latents will be saved for debugging per pass.",
    )

    @root_validator
    def the_validator(cls, values):
        for i, controlnet in enumerate(values.get("controlnets", [])):
            if controlnet.start_percentage >= controlnet.end_percentage:
                raise invalid_data_error(
                    ["controlnet", i, "start_percentage"],
                    "'controlnet.start_percentage' must be smaller than 'controlnet.end_percentage'.",
                )

        # get the ip adapter
        ip_adapters = values.get("ip_adapter", [])

        image_encoder_path = values.get("image_encoder_path", [])
        # get the image encoder subpath
        image_encoder_subfolder = values.get("image_encoder_subfolder", [])
        image_encoder_weight_name = values.get("image_encoder_weight_name", [])

        # ensure that the if the image_encoder_subfolder is provided, the image_encoder_subfolder is provided
        if image_encoder_subfolder is not None and image_encoder_path is None:
            raise invalid_data_error(
                ["image_encoder_subfolder"],
                "'image_encoder_subfolder' must be provided if 'image_encoder_path' is provided.",
            )

        for i, ip_adapter in enumerate(ip_adapters):
            # get the ip adapter path
            weight_name = ip_adapter.weight_name

            # if the ip adapter path is known and a plus model we can check if the image encoder path is provided
            known_plus_models = [
                "ip-adapter-plus_sdxl_vit-h.safetensors",
                "ip-adapter-plus-face_sdxl_vit-h.safetensors",
                "ip-adapter-plus-sdxl_vit-h.bin",
                "ip-adapter-plus-face_sdxl_vit-h.bin",
                "ip-adapter-plus_sd15.bin",
                "ip-adapter-plus_sd15.safetensors",
            ]
            if weight_name in known_plus_models:
                if image_encoder_path is None:
                    raise invalid_data_error(
                        ["image_encoder_path"],
                        """
                        'image_encoder_path' must be provided for plus models. Try using 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
                        and 'models/image_encoder' for the 'image_encoder_path' and 'image_encoder_subfolder' respectively.
                        """,
                    )

        return values


class OutputParameters(BaseModel):
    images: list[Image] = Field(description="The generated image files info.")
    seed: int = Field(
        description="""
            Seed of the generated Image. It will be the same value of the one passed in the
            input or the randomly generated that was used in case none was passed.
        """
    )
    has_nsfw_concepts: list[bool] = Field(
        description="Whether the generated images contain NSFW concepts."
    )
    debug_latents: File | None = Field(
        description="The latents saved for debugging.",
    )
    debug_per_pass_latents: File | None = Field(
        description="The latents saved for debugging per pass.",
    )


@contextmanager
def wrap_excs():
    from fastapi import HTTPException

    try:
        yield
    except (ValueError, TypeError) as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(422, detail=str(exc))


def generate_image(input: InputParameters) -> OutputParameters:
    """
    A single API for text-to-image, built on [fal](https://fal.ai) that supports
    all Stable Diffusion variants, checkpoints and LoRAs from HuggingFace (🤗) and CivitAI.
    """
    import os
    import shutil
    import tempfile

    import cv2
    import numpy as np
    import torch

    session = load_session()

    image_size = None
    if input.image_size is not None:
        image_size = get_image_size(input.image_size)

    with wrap_excs():
        with session.load_model(
            input.model_name,
            loras=input.loras,
            embeddings=input.embeddings,
            controlnets=input.controlnets,
            ip_adapter=input.ip_adapter,
            clip_skip=input.clip_skip,
            scheduler=input.scheduler,
            model_architecture=input.model_architecture,
            image_encoder_path=input.image_encoder_path,
            image_encoder_subfolder=input.image_encoder_subfolder,
            image_encoder_weight_name=input.image_encoder_weight_name,
            variant=input.variant,
        ) as pipe:
            seed = input.seed or torch.seed()

            kwargs = {
                "prompt": input.prompt,
                "negative_prompt": input.negative_prompt,
                "num_images_per_prompt": input.num_images,
                "num_inference_steps": input.num_inference_steps,
                "guidance_scale": input.guidance_scale,
                "generator": torch.Generator("cuda").manual_seed(seed),
            }

            if image_size is not None and input.image_url is None:
                kwargs["width"] = image_size.width
                kwargs["height"] = image_size.height

            if input.controlnets:
                kwargs["controlnet_guess_mode"] = input.controlnet_guess_mode
                kwargs["controlnet_conditioning_scale"] = [
                    x.conditioning_scale for x in input.controlnets
                ]
                kwargs["control_guidance_start"] = [
                    x.start_percentage for x in input.controlnets
                ]
                kwargs["control_guidance_end"] = [
                    x.end_percentage for x in input.controlnets
                ]
                kwargs["ip_adapter_index"] = [
                    x.ip_adapter_index for x in input.controlnets
                ]

                # download all the controlnet images
                controlnet_images = []
                controlnet_masks = []

                # if any controlnet has a mask we need to make an empty one
                any_controlnet_has_mask = any(
                    controlnet.mask_url is not None for controlnet in input.controlnets
                )

                for controlnet in input.controlnets:
                    # TODO replace with something that doesn't download the image every time
                    controlnet_image = read_image_from_url(controlnet.image_url)
                    controlnet_images.append(controlnet_image)
                    if controlnet.mask_url is not None:
                        controlnet_mask = read_image_from_url(controlnet.mask_url)
                        controlnet_masks.append(controlnet_mask)
                    elif any_controlnet_has_mask:
                        controlnet_masks.append(
                            create_empty_mask_for_image(controlnet_image)
                        )

                if len(controlnet_images) > 0:
                    kwargs["image"] = controlnet_images

                if len(controlnet_masks) > 0:
                    kwargs["control_mask"] = controlnet_masks

            kwargs["tile_window_height"] = input.tile_height
            kwargs["tile_window_width"] = input.tile_width
            kwargs["tile_stride_height"] = input.tile_stride_height
            kwargs["tile_stride_width"] = input.tile_stride_width
            kwargs["debug_latents"] = input.debug_latents
            kwargs["debug_per_pass_latents"] = input.debug_per_pass_latents

            if input.image_url is not None:
                print("reading noise image", input.image_url)
                kwargs["image_for_noise"] = read_image_from_url(input.image_url)
                kwargs["strength"] = input.noise_strength

            ip_adapter_images: list[PIL.Image.Image | torch.Tensor] = []
            ip_adapter_masks = []
            ip_adapter_uncond_noise_factors = []
            # if any ip adapter has a mask we need to make an empty one
            any_ip_adapter_has_mask = any(
                ip_adapter.ip_adapter_mask_url is not None
                for ip_adapter in input.ip_adapter
            )

            for i, ip_adapter in enumerate(input.ip_adapter):
                ip_adapter = input.ip_adapter[i]
                ip_adapter_image_url = ip_adapter.ip_adapter_image_url
                if ip_adapter_image_url is not None:
                    if not isinstance(ip_adapter_image_url, list):
                        ip_adapter_image_url = [ip_adapter_image_url]
                    ip_adapter_uncond_noise_factors.append(ip_adapter.unconditional_noising_factor)
                    ip_adapter_face_image = None
                    if ip_adapter.insight_face_model_path is not None:

                        # We only support a single image for identity extraction for now
                        if len(ip_adapter_image_url) > 1:
                            raise ValueError(
                                "Only a single image is supported for identity extraction."
                            )
                        image = read_image_from_url(ip_adapter_image_url[0])
                        ip_adapter_face_image = image
                        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
                        faces = pipe.face_analysis.get(image)
                        largest_face = sorted(
                            faces,
                            key=lambda x: x["bbox"][2] * x["bbox"][3],
                            reverse=True,
                        )[0]
                        face_embedding = torch.tensor(largest_face["embedding"]).to(
                            "cuda"
                        )
                        ip_adapter_images.append(face_embedding)
                    else:
                        ip_adapter_images.append(
                            [read_image_from_url(url) for url in ip_adapter_image_url]
                        )

                    if ip_adapter.ip_adapter_mask_url is not None:
                        print("reading mask image", ip_adapter.ip_adapter_mask_url)
                        ip_adapter_masks.append(
                            [read_image_from_url(ip_adapter.ip_adapter_mask_url)]
                        )
                    elif any_ip_adapter_has_mask:
                        ip_adapter_masks.append(
                            create_empty_mask_for_image(ip_adapter_images[-1])
                        )
                    else:
                        current_image = (
                            ip_adapter_images[-1][0]
                            if ip_adapter_face_image is None
                            else ip_adapter_face_image
                        )
                        if isinstance(current_image, PIL.Image.Image):
                            ip_adapter_masks.append(
                                create_empty_mask_for_image(current_image)
                            )
                        else:
                            raise ValueError(
                                "The image for the IP Adapter must be a PIL Image."
                            )

            if len(ip_adapter_images) > 0:
                kwargs["ip_adapter_image"] = ip_adapter_images

            if len(ip_adapter_masks) > 0:
                kwargs["ip_adapter_mask"] = ip_adapter_masks

            if len(ip_adapter_uncond_noise_factors) > 0:
                kwargs["ip_adapter_unconditional_noising_factor"] = ip_adapter_uncond_noise_factors

            print(f"Generating {input.num_images} images...")
            make_inference = partial(pipe, **kwargs)

            print("Active adapters", pipe.get_active_adapters())
            result = session.execute_on_cuda(make_inference, ignored_models=[pipe])

            has_nsfw_concepts = session.run_safety_checker(
                images=result.images,
                enable_safety_checker=input.enable_safety_checker,
            )

            images = session.upload_images(filter_by(has_nsfw_concepts, result.images))

            with tempfile.TemporaryDirectory() as temp_dir:
                debug_latents = None
                if input.debug_latents:
                    # get the debug latents from the result
                    # save them to files and tar them
                    debug_latents = result.debug_latents

                    # make an images directory
                    images_dir = os.path.join(temp_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    for i, latent in enumerate(debug_latents):
                        for batch_index, latent_for_batch in enumerate(latent):
                            file_path = os.path.join(
                                images_dir,
                                f"{str(batch_index).zfill(4)}_{str(i).zfill(4)}.png",
                            )
                            latent_for_batch.save(file_path)

                    # create a tar file using shutil
                    tar_file_path = os.path.join(temp_dir, "debug_latents.tar")
                    shutil.make_archive(tar_file_path[:-4], "tar", images_dir)
                    debug_latents = File.from_path(tar_file_path)

                debug_per_pass_latents = None
                if input.debug_per_pass_latents:
                    # get the debug latents from the result
                    # save them to files and tar them
                    debug_per_pass_latents = result.debug_per_pass_latents
                    # make an images directory
                    images_dir = os.path.join(temp_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    for pass_index, latent_per_pass in enumerate(
                        debug_per_pass_latents
                    ):
                        for timestep_index, (dims, latents, weights) in enumerate(
                            latent_per_pass
                        ):

                            for batch_index, latent_for_batch in enumerate(latents):
                                file_path = os.path.join(
                                    images_dir,
                                    f"{str(batch_index).zfill(4)}_{str(pass_index).zfill(4)}_{str(timestep_index).zfill(4)}_{dims}.png",
                                )
                                latent_for_batch.save(file_path)

                            for batch_index, weight_for_batch in enumerate(weights):
                                file_path = os.path.join(
                                    images_dir,
                                    f"weight_{str(batch_index).zfill(4)}_{str(pass_index).zfill(4)}_{str(timestep_index).zfill(4)}_{dims}.png",
                                )
                                weight_for_batch.save(file_path)

                    # create a tar file using shutil
                    tar_file_path = os.path.join(temp_dir, "debug_per_pass_latents.tar")
                    shutil.make_archive(tar_file_path[:-4], "tar", images_dir)
                    debug_per_pass_latents = File.from_path(tar_file_path)

                return OutputParameters(
                    images=images,
                    seed=seed,
                    has_nsfw_concepts=has_nsfw_concepts,
                    debug_latents=debug_latents,
                    debug_per_pass_latents=debug_per_pass_latents,
                )


class TextToImageInputOverwrites(BaseModel):
    SCHEMA_IGNORES: ClassVar[set[str]] = {
        "image_url",
        "noise_strength",
    }


class TextToImageInput(TextToImageInputOverwrites, InputParameters):
    pass


class ImageToImageInputOverwrites(BaseModel):
    SCHEMA_IGNORES: ClassVar[set[str]] = {
        "image_size",
    }


class ImageToImageInput(ImageToImageInputOverwrites, InputParameters):
    pass


class MegaPipeline(
    fal.App,
    _scheduler="nomad",
    # _scheduler="kubernetes",
    max_concurrency=2,
    keep_alive=300,
    resolver="uv",  # type: ignore
):
    machine_type = "GPU"

    requirements = [
        "git+https://github.com/huggingface/diffusers.git@0d7c4790235ac00b4524b492bc2a680dcc5cf6b0",
        "transformers",
        "accelerate",
        "torch>=2.1",
        "torchvision",
        "safetensors",
        "pytorch-lightning",
        "omegaconf",
        "invisible-watermark",
        "google-cloud-storage",
        "psutil",
        "peft",
        "huggingface-hub==0.20.2",
        "git+https://github.com/badayvedat/insightface.git@1ffa3405eedcfe4193c3113affcbfc294d0e684f#subdirectory=python-package",
        "opencv-python",
        "numpy",
        "onnxruntime-gpu",
    ]

    def setup(self) -> None:
        initial_input = InputParameters(
            model_name="frankjoshua/albedobaseXL_v13",
            prompt="A person",
            negative_prompt="blurry, out of focus",
            image_url="https://github.com/okaris/omni-zero/assets/1448702/b97d2b29-d35b-4e73-8d86-e3cbe9953e8c",
            noise_strength=0.85,
            controlnets=[
                ControlNet(
                    path="okaris/face-controlnet-xl",
                    image_url="https://github.com/okaris/omni-zero/assets/1448702/da96c849-5e75-4b23-9d5a-a44311509626",
                    conditioning_scale=1.0,
                    start_percentage=0.0,
                    end_percentage=1.0,
                    ip_adapter_index=0,
                ),
                ControlNet(
                    path="okaris/zoe-depth-controlnet-xl",
                    image_url="https://github.com/okaris/omni-zero/assets/1448702/977b7ab0-8b88-430d-8b7c-a415d47fd1c0",
                    conditioning_scale=0.5,
                    start_percentage=0.0,
                    end_percentage=1.0,
                ),
            ],
            image_encoder_path="h94/IP-Adapter",
            image_encoder_subfolder="models/image_encoder",
            ip_adapter=[
                IPAdapter(
                    ip_adapter_image_url="https://github.com/okaris/omni-zero/assets/1448702/ba193a3a-f90e-4461-848a-560454531c58",
                    path="okaris/ip-adapter-instantid",
                    model_subfolder=None,
                    weight_name="ip-adapter-instantid.bin",
                    insight_face_model_path="okaris/antelopev2",
                    scale=1.0,
                    unconditional_noising_factor=0.5,
                ),
                IPAdapter(
                    ip_adapter_image_url="https://github.com/okaris/omni-zero/assets/1448702/64dc150b-f683-41b1-be23-b6a52c771584",
                    path="h94/IP-Adapter",
                    model_subfolder="sdxl_models",
                    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                    scale_json={
                        "down": {"block_2": [0.0, 0.0]},  # Composition
                        "up": {"block_0": [0.0, 1.0, 0.0]},  # Style
                    },
                    unconditional_noising_factor=0.5,
                ),
                IPAdapter(
                    ip_adapter_image_url="https://github.com/okaris/omni-zero/assets/1448702/b97d2b29-d35b-4e73-8d86-e3cbe9953e8c",
                    path="h94/IP-Adapter",
                    model_subfolder="sdxl_models",
                    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                    scale_json={
                        "down": {"block_2": [0.0, 1.0]},  # Composition
                        "up": {"block_0": [0.0, 0.0, 0.0]},  # Style
                    },
                    unconditional_noising_factor=0.5,
                ),
            ],
            guidance_scale=3.5,
            num_inference_steps=20,
            num_images=1,
            seed=42,
            model_architecture="sdxl",
            scheduler="DPM++ 2M SDE Karras",
            image_size=ImageSize(width=1024, height=1024),
        )

        result = generate_image(initial_input)
        print(result)

        self.ready = True
        return

    @fal.endpoint("/health")
    def health(self):
        if hasattr(self, "ready") and self.ready:
            return {"status": "ok"}
        else:
            return {"status": "not ready"}

    @fal.endpoint("/")
    def text_to_image(self, input: TextToImageInput) -> OutputParameters:
        return generate_image(input)

    @fal.endpoint("/image-to-image")
    def image_to_image(self, input: ImageToImageInput) -> OutputParameters:
        return generate_image(input)


if __name__ == "__main__":
    # curl $URL -H 'content-type: application/json' -H 'accept: application/json, */*;q=0.5' -d '{"image_url":"https://storage.googleapis.com/falserverless/model_tests/supir/GGsAolHXsAA58vn.jpeg"}'
    app_fn = fal.wrap_app(MegaPipeline)
    app_fn()

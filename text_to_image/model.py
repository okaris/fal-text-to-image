from contextlib import contextmanager
from functools import lru_cache, partial
from typing import Any, ClassVar, Literal
from urllib.request import Request, urlopen

import fal
from fal import cached
from fal.toolkit import Image, ImageSizeInput, get_image_size
from fal.toolkit.image import ImageSize
from pydantic import BaseModel, Field, root_validator

from text_to_image.runtime import SUPPORTED_SCHEDULERS, GlobalRuntime, filter_by


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
    import PIL
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


class InputParameters(OrderedBaseModel):
    model_name: str = Field(
        description="URL or HuggingFace ID of the base model to generate the image.",
        examples=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "SG161222/Realistic_Vision_V2.0",
        ],
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
        default=1024,
        description="The size of the tiles to be used for the image generation.",
        ge=128,
        le=4096,
    )
    tile_height: int = Field(
        default=1024,
        description="The size of the tiles to be used for the image generation.",
        ge=128,
        le=4096,
    )
    tile_stride_width: int = Field(
        default=512,
        description="The stride of the tiles to be used for the image generation.",
        ge=64,
        le=2048,
    )
    tile_stride_height: int = Field(
        default=512,
        description="The stride of the tiles to be used for the image generation.",
        ge=64,
        le=2048,
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

        if ip_adapters:
            if len(ip_adapters) > 1:
                raise invalid_data_error(
                    ["ip_adapter", 1], "Only one IP adapter is supported at the moment."
                )
            else:
                ip_adapter = ip_adapters[0]
            # get the ip adapter path
            ip_adapter_path = ip_adapter.path
            # get the ip adapter image url
            ip_adapter_image_url = ip_adapter.ip_adapter_image_url
            # get the image encoder path
            image_encoder_path = ip_adapter.image_encoder_path
            # get the image encoder subpath
            image_encoder_subpath = ip_adapter.image_encoder_subpath
            # get the weight name
            weight_name = ip_adapter.weight_name
            # get the model subfolder
            model_subfolder = ip_adapter.model_subfolder

            # make sure that if the ip adapter path is not None, then the ip adapter image url is not None
            if ip_adapter_path is not None and ip_adapter_image_url is None:
                raise invalid_data_error(
                    ["ip_adapter", 0, "ip_adapter_image_url"],
                    "'ip_adapter.ip_adapter_image_url' must be provided if 'ip_adapter.path' is provided.",
                )

            # make sure that if the image encoder path is not None, then the image encoder subpath is not None
            if image_encoder_path is not None and image_encoder_subpath is None:
                raise invalid_data_error(
                    ["ip_adapter", 0, "image_encoder_subpath"],
                    "'ip_adapter.image_encoder_subpath' must be provided if 'ip_adapter.image_encoder_path' is provided.",
                )

            # make sure that if the weight name is not None, the path is not None
            if weight_name is not None and ip_adapter_path is None:
                raise invalid_data_error(
                    ["ip_adapter", 0, "weight_name"],
                    "'ip_adapter.path' must be provided if 'ip_adapter.weight_name' is provided.",
                )

            # make sure that if the model subfolder is not None, the path is not None
            if model_subfolder is not None and ip_adapter_path is None:
                raise invalid_data_error(
                    ["ip_adapter", 0, "model_subfolder"],
                    "'ip_adapter.path' must be provided if 'ip_adapter.model_subfolder' is provided.",
                )

            # make sure that if the encoder path is not None, the path is not None
            if image_encoder_path is not None and ip_adapter_path is None:
                raise invalid_data_error(
                    ["ip_adapter", 0, "image_encoder_path"],
                    "'ip_adapter.path' must be provided if 'ip_adapter.image_encoder_path' is provided.",
                )

            # make sure that if the encoder subpath is not None, the image_encoder_path is not None
            if image_encoder_subpath is not None and image_encoder_path is None:
                raise invalid_data_error(
                    ["ip_adapter", 0, "image_encoder_subpath"],
                    "'ip_adapter.image_encoder_path' must be provided if 'ip_adapter.image_encoder_subpath' is provided.",
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
    import torch

    session = load_session()

    image_size = None
    if input.image_size is not None:
        image_size = get_image_size(input.image_size)

    # we already validate there is max 1 ip adapter
    if len(input.ip_adapter) > 0:
        ip_adapter = input.ip_adapter[0]
    else:
        ip_adapter = None

    with wrap_excs():
        with session.load_model(
            input.model_name,
            loras=input.loras,
            embeddings=input.embeddings,
            controlnets=input.controlnets,
            ip_adapter=ip_adapter,
            clip_skip=input.clip_skip,
            scheduler=input.scheduler,
            model_architecture=input.model_architecture,
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

                # download all the controlnet images
                controlnet_images = []
                for controlnet in input.controlnets:
                    # TODO replace with something that doesn't download the image every time
                    controlnet_image = read_image_from_url(controlnet.image_url)

                    controlnet_images.append(controlnet_image)

                kwargs["image"] = controlnet_images

            kwargs["tile_window_height"] = input.tile_height
            kwargs["tile_window_width"] = input.tile_width
            kwargs["tile_stride_height"] = input.tile_stride_height
            kwargs["tile_stride_width"] = input.tile_stride_width

            if input.image_url is not None:
                print("reading noise image", input.image_url)
                kwargs["image_for_noise"] = read_image_from_url(input.image_url)
                kwargs["strength"] = input.noise_strength

            if ip_adapter is not None and ip_adapter.path is not None:
                kwargs["ip_adapter_image"] = read_image_from_url(
                    ip_adapter.ip_adapter_image_url
                )

            print(f"Generating {input.num_images} images...")
            make_inference = partial(pipe, **kwargs)

            print("Active adapters", pipe.get_active_adapters())
            result = session.execute_on_cuda(make_inference, ignored_models=[pipe])

            has_nsfw_concepts = session.run_safety_checker(
                images=result.images,
                enable_safety_checker=input.enable_safety_checker,
            )

            images = session.upload_images(filter_by(has_nsfw_concepts, result.images))

            print("images", images)

            return OutputParameters(
                images=images, seed=seed, has_nsfw_concepts=has_nsfw_concepts
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
    max_concurrency=2,
    keep_alive=300,
    resolver="uv",  # type: ignore
):
    machine_type = "GPU"

    requirements = [
        "diffusers==0.27.2",
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
    ]

    def setup(self) -> None:
        initial_input = InputParameters(
            model_name=f"stabilityai/stable-diffusion-xl-base-1.0",
            prompt="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
            image_url="https://storage.googleapis.com/falserverless/lora/1665_Girl_with_a_Pearl_Earring.jpg",
            noise_strength=0.5,
            loras=[
                LoraWeight(
                    path="https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
                    scale=1,
                )
            ],
            embeddings=[
                Embedding(
                    path="https://storage.googleapis.com/falserverless/style_lora/pimento_embeddings.pti",
                    tokens=["<s0>", "<s1>"],
                )
            ],
            controlnets=[
                ControlNet(
                    path="diffusers/controlnet-canny-sdxl-1.0",
                    image_url="https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/canny-edge.resized.jpg",
                    conditioning_scale=1.0,
                    start_percentage=0.0,
                    end_percentage=1.0,
                )
            ],
            ip_adapter=[
                IPAdapter(
                    ip_adapter_image_url="https://storage.googleapis.com/falserverless/model_tests/controlnet_sdxl/robot.jpeg",
                    path="h94/IP-Adapter",
                    model_subfolder="sdxl_models",
                    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                    image_encoder_path="h94/IP-Adapter",
                    image_encoder_subpath="models/image_encoder",
                )
            ],
            guidance_scale=7.5,
            num_inference_steps=20,
            num_images=1,
            seed=42,
            model_architecture="sdxl",
            scheduler="Euler A",
            image_size=ImageSize(width=256, height=256),
        )
        _ = generate_image(initial_input)
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

# type: ignore
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def create_pipeline():
    import copy
    import inspect
    import math
    from collections.abc import Callable
    from typing import Any

    import numpy as np
    import PIL.Image
    import torch
    import torch.nn.functional as F
    from diffusers.image_processor import (
        IPAdapterMaskProcessor,
        PipelineImageInput,
        VaeImageProcessor,
    )
    from diffusers.loaders import (
        FromSingleFileMixin,
        IPAdapterMixin,
        LoraLoaderMixin,
        StableDiffusionXLLoraLoaderMixin,
        TextualInversionLoaderMixin,
    )
    from diffusers.models import (
        AutoencoderKL,
        ControlNetModel,
        ImageProjection,
        UNet2DConditionModel,
    )
    from diffusers.models.attention_processor import (
        AttnProcessor2_0,
        LoRAAttnProcessor2_0,
        LoRAXFormersAttnProcessor,
        XFormersAttnProcessor,
    )
    from diffusers.models.lora import adjust_lora_scale_text_encoder
    from diffusers.pipelines.pipeline_utils import (
        DiffusionPipeline,
        StableDiffusionMixin,
    )
    from diffusers.pipelines.stable_diffusion.pipeline_output import (
        StableDiffusionPipelineOutput,
    )
    from diffusers.pipelines.stable_diffusion.safety_checker import (
        StableDiffusionSafetyChecker,
    )
    from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
        StableDiffusionXLPipelineOutput,
    )
    from diffusers.schedulers import KarrasDiffusionSchedulers
    from diffusers.utils import (
        USE_PEFT_BACKEND,
        deprecate,
        logging,
        replace_example_docstring,
        scale_lora_layers,
        unscale_lora_layers,
    )
    from diffusers.utils.import_utils import is_invisible_watermark_available
    from diffusers.utils.torch_utils import (
        is_compiled_module,
        is_torch_version,
        randn_tensor,
    )
    from transformers import (
        CLIPImageProcessor,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPVisionModelWithProjection,
    )

    if is_invisible_watermark_available():
        from diffusers.pipelines.stable_diffusion_xl.watermark import (
            StableDiffusionXLWatermarker,
        )

    from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

    logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

    def get_views(
        panorama_height,
        panorama_width,
        window_height=1024,
        window_width=1024,
        stride_height=256,
        stride_width=256,
        circular_padding=False,
    ):
        # Adjust calculations for separate window height and width
        num_blocks_height = max(
            math.ceil((panorama_height - window_height) / stride_height) + 1, 1
        )
        num_blocks_width = max(
            math.ceil((panorama_width - window_width) / stride_width) + 1, 1
        )

        if circular_padding:
            num_blocks_width = max(panorama_width // stride_width, 1)

        views = []
        for h_block in range(num_blocks_height):
            h_start = h_block * stride_height
            # Adjust h_start for the last block to ensure it's of window_height
            if h_start + window_height > panorama_height:
                h_start = panorama_height - window_height
            h_end = h_start + window_height

            for w_block in range(num_blocks_width):
                w_start = w_block * stride_width
                # Adjust w_start for the last block to ensure it's of window_width
                if w_start + window_width > panorama_width:
                    w_start = panorama_width - window_width
                w_end = w_start + window_width

                views.append((h_start, h_end, w_start, w_end))

        return views

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
    def retrieve_latents(
        encoder_output: torch.Tensor,
        generator: torch.Generator | None = None,
        sample_mode: str = "sample",
    ):
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    EXAMPLE_DOC_STRING = """
        Examples:
            ```py
            >>> # !pip install opencv-python transformers accelerate
            >>> from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
            >>> from diffusers.utils import load_image
            >>> import numpy as np
            >>> import torch

            >>> import cv2
            >>> from PIL import Image

            >>> prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
            >>> negative_prompt = "low quality, bad quality, sketches"

            >>> # download an image
            >>> image = load_image(
            ...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
            ... )

            >>> # initialize the models and pipeline
            >>> controlnet_conditioning_scale = 0.5  # recommended for good generalization
            >>> controlnet = ControlNetModel.from_pretrained(
            ...     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
            ... )
            >>> vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            >>> pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
            ... )
            >>> pipe.enable_model_cpu_offload()

            >>> # get canny image
            >>> image = np.array(image)
            >>> image = cv2.Canny(image, 100, 200)
            >>> image = image[:, :, None]
            >>> image = np.concatenate([image, image, image], axis=2)
            >>> canny_image = Image.fromarray(image)

            >>> # generate image
            >>> image = pipe(
            ...     prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image
            ... ).images[0]
            ```
    """

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
    def retrieve_timesteps(
        scheduler,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[int] | None = None,
        **kwargs,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                    Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                    timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                    must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    def prepare_noise_image(image):
        if isinstance(image, torch.Tensor):
            # Batch single image
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image = image.to(dtype=torch.float32)
        else:
            # preprocess image
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        return image

    class StableDiffusionXLControlNetPipeline(
        DiffusionPipeline,
        StableDiffusionMixin,
        TextualInversionLoaderMixin,
        StableDiffusionXLLoraLoaderMixin,
        IPAdapterMixin,
        FromSingleFileMixin,
    ):
        r"""
        Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet guidance.

        This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
        implemented for all pipelines (downloading, saving, running on a particular device, etc.).

        The pipeline also inherits the following loading methods:
            - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
            - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
            - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
            - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
            - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

        Args:
            vae ([`AutoencoderKL`]):
                Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
            text_encoder ([`~transformers.CLIPTextModel`]):
                Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
            text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):
                Second frozen text-encoder
                ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
            tokenizer ([`~transformers.CLIPTokenizer`]):
                A `CLIPTokenizer` to tokenize text.
            tokenizer_2 ([`~transformers.CLIPTokenizer`]):
                A `CLIPTokenizer` to tokenize text.
            unet ([`UNet2DConditionModel`]):
                A `UNet2DConditionModel` to denoise the encoded image latents.
            controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
                Provides additional conditioning to the `unet` during the denoising process. If you set multiple
                ControlNets as a list, the outputs from each ControlNet are added together to create one combined
                additional conditioning.
            scheduler ([`SchedulerMixin`]):
                A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
                [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
            force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
                Whether the negative prompt embeddings should always be set to 0. Also see the config of
                `stabilityai/stable-diffusion-xl-base-1-0`.
            add_watermarker (`bool`, *optional*):
                Whether to use the [invisible_watermark](https://github.com/ShieldMnt/invisible-watermark/) library to
                watermark output images. If not defined, it defaults to `True` if the package is installed; otherwise no
                watermarker is used.
        """

        # leave controlnet out on purpose because it iterates with unet
        model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
        _optional_components = [
            "tokenizer",
            "tokenizer_2",
            "text_encoder",
            "text_encoder_2",
            "feature_extractor",
            "image_encoder",
        ]
        _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            text_encoder_2: CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            tokenizer_2: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            controlnet: ControlNetModel
            | list[ControlNetModel]
            | tuple[ControlNetModel]
            | MultiControlNetModel
            | None = None,
            force_zeros_for_empty_prompt: bool = True,
            add_watermarker: bool | None = None,
            feature_extractor: CLIPImageProcessor = None,
            image_encoder: CLIPVisionModelWithProjection = None,
        ):
            super().__init__()

            if controlnet is None:
                controlnet = MultiControlNetModel([])

            if isinstance(controlnet, (list, tuple)):
                controlnet = MultiControlNetModel(controlnet)

            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
            )
            self.ip_adapter_mask_processor = IPAdapterMaskProcessor()
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor,
                do_convert_rgb=True,
                do_normalize=False,
            )
            add_watermarker = (
                add_watermarker
                if add_watermarker is not None
                else is_invisible_watermark_available()
            )

            if add_watermarker:
                self.watermark = StableDiffusionXLWatermarker()
            else:
                self.watermark = None

            self.register_to_config(
                force_zeros_for_empty_prompt=force_zeros_for_empty_prompt
            )

        def get_timesteps(self, num_inference_steps, strength, device):
            # get the original timestep using init_timestep
            init_timestep = min(
                int(num_inference_steps * strength), num_inference_steps
            )

            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)

            return timesteps, num_inference_steps - t_start

        # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt
        def encode_prompt(
            self,
            prompt: str,
            prompt_2: str | None = None,
            device: torch.device | None = None,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = True,
            negative_prompt: str | None = None,
            negative_prompt_2: str | None = None,
            prompt_embeds: torch.FloatTensor | None = None,
            negative_prompt_embeds: torch.FloatTensor | None = None,
            pooled_prompt_embeds: torch.FloatTensor | None = None,
            negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
            lora_scale: float | None = None,
            clip_skip: int | None = None,
        ):
            r"""
            Encodes the prompt into text encoder hidden states.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    prompt to be encoded
                prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                    used in both text-encoders
                device: (`torch.device`):
                    torch device
                num_images_per_prompt (`int`):
                    number of images that should be generated per prompt
                do_classifier_free_guidance (`bool`):
                    whether to use classifier free guidance or not
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                negative_prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                    `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
                negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                    argument.
                pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                    If not provided, pooled text embeddings will be generated from `prompt` input argument.
                negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                    input argument.
                lora_scale (`float`, *optional*):
                    A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
            """
            device = device or self._execution_device

            # set lora scale so that monkey patched LoRA
            # function of text encoder can correctly access it
            if lora_scale is not None and isinstance(
                self, StableDiffusionXLLoraLoaderMixin
            ):
                self._lora_scale = lora_scale

                # dynamically adjust the LoRA scale
                if self.text_encoder is not None:
                    if not USE_PEFT_BACKEND:
                        adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                    else:
                        scale_lora_layers(self.text_encoder, lora_scale)

                if self.text_encoder_2 is not None:
                    if not USE_PEFT_BACKEND:
                        adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                    else:
                        scale_lora_layers(self.text_encoder_2, lora_scale)

            prompt = [prompt] if isinstance(prompt, str) else prompt

            if prompt is not None:
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            # Define tokenizers and text encoders
            tokenizers = (
                [self.tokenizer, self.tokenizer_2]
                if self.tokenizer is not None
                else [self.tokenizer_2]
            )
            text_encoders = (
                [self.text_encoder, self.text_encoder_2]
                if self.text_encoder is not None
                else [self.text_encoder_2]
            )

            if prompt_embeds is None:
                prompt_2 = prompt_2 or prompt
                prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

                # textual inversion: process multi-vector tokens if necessary
                prompt_embeds_list = []
                prompts = [prompt, prompt_2]
                for prompt, tokenizer, text_encoder in zip(
                    prompts, tokenizers, text_encoders
                ):
                    if isinstance(self, TextualInversionLoaderMixin):
                        prompt = self.maybe_convert_prompt(prompt, tokenizer)

                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    text_input_ids = text_inputs.input_ids
                    untruncated_ids = tokenizer(
                        prompt, padding="longest", return_tensors="pt"
                    ).input_ids

                    if untruncated_ids.shape[-1] >= text_input_ids.shape[
                        -1
                    ] and not torch.equal(text_input_ids, untruncated_ids):
                        removed_text = tokenizer.batch_decode(
                            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                        )
                        logger.warning(
                            "The following part of your input was truncated because CLIP can only handle sequences up to"
                            f" {tokenizer.model_max_length} tokens: {removed_text}"
                        )

                    prompt_embeds = text_encoder(
                        text_input_ids.to(device), output_hidden_states=True
                    )

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds[0]
                    if clip_skip is None:
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                    else:
                        # "2" because SDXL always indexes from the penultimate layer.
                        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                    prompt_embeds_list.append(prompt_embeds)

                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

            # get unconditional embeddings for classifier free guidance
            zero_out_negative_prompt = (
                negative_prompt is None and self.config.force_zeros_for_empty_prompt
            )
            if (
                do_classifier_free_guidance
                and negative_prompt_embeds is None
                and zero_out_negative_prompt
            ):
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            elif do_classifier_free_guidance and negative_prompt_embeds is None:
                negative_prompt = negative_prompt or ""
                negative_prompt_2 = negative_prompt_2 or negative_prompt

                # normalize str to list
                negative_prompt = (
                    batch_size * [negative_prompt]
                    if isinstance(negative_prompt, str)
                    else negative_prompt
                )
                negative_prompt_2 = (
                    batch_size * [negative_prompt_2]
                    if isinstance(negative_prompt_2, str)
                    else negative_prompt_2
                )

                uncond_tokens: list[str]
                if prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = [negative_prompt, negative_prompt_2]

                negative_prompt_embeds_list = []
                for negative_prompt, tokenizer, text_encoder in zip(
                    uncond_tokens, tokenizers, text_encoders
                ):
                    if isinstance(self, TextualInversionLoaderMixin):
                        negative_prompt = self.maybe_convert_prompt(
                            negative_prompt, tokenizer
                        )

                    max_length = prompt_embeds.shape[1]
                    uncond_input = tokenizer(
                        negative_prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    negative_prompt_embeds = text_encoder(
                        uncond_input.input_ids.to(device),
                        output_hidden_states=True,
                    )
                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                    negative_prompt_embeds_list.append(negative_prompt_embeds)

                negative_prompt_embeds = torch.concat(
                    negative_prompt_embeds_list, dim=-1
                )

            if self.text_encoder_2 is not None:
                prompt_embeds = prompt_embeds.to(
                    dtype=self.text_encoder_2.dtype, device=device
                )
            else:
                prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                if self.text_encoder_2 is not None:
                    negative_prompt_embeds = negative_prompt_embeds.to(
                        dtype=self.text_encoder_2.dtype, device=device
                    )
                else:
                    negative_prompt_embeds = negative_prompt_embeds.to(
                        dtype=self.unet.dtype, device=device
                    )

                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_images_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )

            pooled_prompt_embeds = pooled_prompt_embeds.repeat(
                1, num_images_per_prompt
            ).view(bs_embed * num_images_per_prompt, -1)
            if do_classifier_free_guidance:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
                    1, num_images_per_prompt
                ).view(bs_embed * num_images_per_prompt, -1)

            if self.text_encoder is not None:
                if (
                    isinstance(self, StableDiffusionXLLoraLoaderMixin)
                    and USE_PEFT_BACKEND
                ):
                    # Retrieve the original scale by scaling back the LoRA layers
                    unscale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if (
                    isinstance(self, StableDiffusionXLLoraLoaderMixin)
                    and USE_PEFT_BACKEND
                ):
                    # Retrieve the original scale by scaling back the LoRA layers
                    unscale_lora_layers(self.text_encoder_2, lora_scale)

            return (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            )

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
        def encode_image(
            self,
            image,
            device,
            num_images_per_prompt,
            output_hidden_states=None,
            unconditional_noising_factor=None,
        ):
            dtype = next(self.image_encoder.parameters()).dtype

            needs_encoding = not isinstance(image, torch.Tensor)
            if needs_encoding:
                image = self.feature_extractor(image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            additional_noise_for_uncond = (
                torch.rand_like(image) * unconditional_noising_factor
            )

            if output_hidden_states:
                if needs_encoding:
                    image_encoded = self.image_encoder(image, output_hidden_states=True)
                    image_enc_hidden_states = image_encoded.hidden_states[-2]
                else:
                    image_enc_hidden_states = image.unsqueeze(0).unsqueeze(0)
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )

                if needs_encoding:
                    uncond_image_encoded = self.image_encoder(
                        additional_noise_for_uncond, output_hidden_states=True
                    )
                    uncond_image_enc_hidden_states = uncond_image_encoded.hidden_states[
                        -2
                    ]
                else:
                    uncond_image_enc_hidden_states = (
                        additional_noise_for_uncond.unsqueeze(0).unsqueeze(0)
                    )
                uncond_image_enc_hidden_states = (
                    uncond_image_enc_hidden_states.repeat_interleave(
                        num_images_per_prompt, dim=0
                    )
                )
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                if needs_encoding:
                    image_encoded = self.image_encoder(image)
                    image_embeds = image_encoded.image_embeds
                else:
                    image_embeds = image.unsqueeze(0).unsqueeze(0)
                if needs_encoding:
                    uncond_image_encoded = self.image_encoder(
                        additional_noise_for_uncond
                    )
                    uncond_image_embeds = uncond_image_encoded.image_embeds
                else:
                    uncond_image_embeds = additional_noise_for_uncond.unsqueeze(
                        0
                    ).unsqueeze(0)

                image_embeds = image_embeds.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                uncond_image_embeds = uncond_image_embeds.repeat_interleave(
                    num_images_per_prompt, dim=0
                )

                image_embeds = image_embeds.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                uncond_image_embeds = uncond_image_embeds.repeat_interleave(
                    num_images_per_prompt, dim=0
                )

                return image_embeds, uncond_image_embeds

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
        def prepare_ip_adapter_image_embeds(
            self,
            ip_adapter_image,
            ip_adapter_image_embeds,
            ip_adapter_mask,
            width,
            height,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            unconditional_noising_factors=None,
        ):
            if ip_adapter_image_embeds is None:
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]

                if len(ip_adapter_image) != len(
                    self.unet.encoder_hid_proj.image_projection_layers
                ):
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )

                image_embeds = []

                if unconditional_noising_factors is None:
                    unconditional_noising_factors = [0.0] * len(ip_adapter_image)
                else:
                    if not isinstance(unconditional_noising_factors, list):
                        unconditional_noising_factors = [unconditional_noising_factors]

                    if len(unconditional_noising_factors) != len(ip_adapter_image):
                        raise ValueError(
                            f"`unconditional_noising_factors` must have same length as the number of IP Adapters. Got {len(unconditional_noising_factors)} values and {len(ip_adapter_image)} IP Adapters."
                        )

                for (
                    single_ip_adapter_image,
                    unconditional_noising_factor,
                    image_proj_layer,
                ) in zip(
                    ip_adapter_image,
                    unconditional_noising_factors,
                    self.unet.encoder_hid_proj.image_projection_layers,
                ):
                    output_hidden_state = not isinstance(
                        image_proj_layer, ImageProjection
                    )
                    (
                        single_image_embeds,
                        single_negative_image_embeds,
                    ) = self.encode_image(
                        single_ip_adapter_image,
                        device,
                        1,
                        output_hidden_state,
                        unconditional_noising_factor,
                    )
                    single_image_embeds = torch.stack(
                        [single_image_embeds] * num_images_per_prompt, dim=0
                    )
                    single_negative_image_embeds = torch.stack(
                        [single_negative_image_embeds] * num_images_per_prompt, dim=0
                    )

                    if do_classifier_free_guidance:
                        single_image_embeds = torch.cat(
                            [single_negative_image_embeds, single_image_embeds]
                        )
                        single_image_embeds = single_image_embeds.to(device)

                    image_embeds.append(single_image_embeds)
            else:
                repeat_dims = [1]
                image_embeds = []
                for single_image_embeds in ip_adapter_image_embeds:
                    if do_classifier_free_guidance:
                        (
                            single_negative_image_embeds,
                            single_image_embeds,
                        ) = single_image_embeds.chunk(2)
                        single_image_embeds = single_image_embeds.repeat(
                            num_images_per_prompt,
                            *(repeat_dims * len(single_image_embeds.shape[1:])),
                        )
                        single_negative_image_embeds = (
                            single_negative_image_embeds.repeat(
                                num_images_per_prompt,
                                *(
                                    repeat_dims
                                    * len(single_negative_image_embeds.shape[1:])
                                ),
                            )
                        )
                        single_image_embeds = torch.cat(
                            [single_negative_image_embeds, single_image_embeds]
                        )
                    else:
                        single_image_embeds = single_image_embeds.repeat(
                            num_images_per_prompt,
                            *(repeat_dims * len(single_image_embeds.shape[1:])),
                        )
                    image_embeds.append(single_image_embeds)

            ip_adapter_masks = None
            if ip_adapter_mask is not None:
                ip_adapter_masks = self.ip_adapter_mask_processor.preprocess(
                    ip_adapter_mask, height=height, width=width
                )
                ip_adapter_masks = [mask.unsqueeze(0) for mask in ip_adapter_masks]

                reshaped_ip_adapter_masks = []
                for ip_img, mask in zip(image_embeds, ip_adapter_masks):
                    if isinstance(ip_img, list):
                        num_images = len(ip_img)
                        mask = mask.repeat(1, num_images, 1, 1)

                    reshaped_ip_adapter_masks.append(mask)

                ip_adapter_masks = reshaped_ip_adapter_masks

            return image_embeds, ip_adapter_masks

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
        def prepare_extra_step_kwargs(self, generator, eta):
            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]

            accepts_eta = "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()
            )
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            # check if the scheduler accepts generator
            accepts_generator = "generator" in set(
                inspect.signature(self.scheduler.step).parameters.keys()
            )
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            return extra_step_kwargs

        def prepare_image_latents(
            self,
            image,
            timestep,
            batch_size,
            num_images_per_prompt,
            dtype,
            device,
            generator=None,
            add_noise=True,
        ):
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )

            # Offload text encoder if `enable_model_cpu_offload` was enabled
            if (
                hasattr(self, "final_offload_hook")
                and self.final_offload_hook is not None
            ):
                self.text_encoder_2.to("cpu")
                torch.cuda.empty_cache()

            image = image.to(device=device, dtype=dtype)

            batch_size = batch_size * num_images_per_prompt

            if image.shape[1] == 4:
                init_latents = image

            else:
                # make sure the VAE is in float32 mode, as it overflows in float16
                try:
                    if self.vae.config.force_upcast:
                        image = image.float()
                        self.vae.to(dtype=torch.float32)

                    if isinstance(generator, list) and len(generator) != batch_size:
                        raise ValueError(
                            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                        )

                    elif isinstance(generator, list):
                        init_latents = [
                            retrieve_latents(
                                self.vae.encode(image[i : i + 1]),
                                generator=generator[i],
                            )
                            for i in range(batch_size)
                        ]
                        init_latents = torch.cat(init_latents, dim=0)
                    else:
                        init_latents = retrieve_latents(
                            self.vae.encode(image), generator=generator
                        )

                    init_latents = init_latents.to(dtype)
                    init_latents = self.vae.config.scaling_factor * init_latents
                finally:
                    if self.vae.config.force_upcast:
                        self.vae.to(dtype)

            if (
                batch_size > init_latents.shape[0]
                and batch_size % init_latents.shape[0] == 0
            ):
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // init_latents.shape[0]
                init_latents = torch.cat(
                    [init_latents] * additional_image_per_prompt, dim=0
                )
            elif (
                batch_size > init_latents.shape[0]
                and batch_size % init_latents.shape[0] != 0
            ):
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                init_latents = torch.cat([init_latents], dim=0)

            if add_noise:
                shape = init_latents.shape
                noise = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
                # get latents
                init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

            latents = init_latents

            return latents

        def check_inputs(
            self,
            prompt,
            prompt_2,
            image,
            callback_steps,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            negative_pooled_prompt_embeds=None,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            callback_on_step_end_tensor_inputs=None,
        ):
            if callback_steps is not None and (
                not isinstance(callback_steps, int) or callback_steps <= 0
            ):
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )

            if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs
                for k in callback_on_step_end_tensor_inputs
            ):
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
                )

            if prompt is not None and prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            elif prompt_2 is not None and prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            elif prompt is None and prompt_embeds is None:
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            elif prompt is not None and (
                not isinstance(prompt, str) and not isinstance(prompt, list)
            ):
                raise ValueError(
                    f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
                )
            elif prompt_2 is not None and (
                not isinstance(prompt_2, str) and not isinstance(prompt_2, list)
            ):
                raise ValueError(
                    f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}"
                )

            if negative_prompt is not None and negative_prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )
            elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )

            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )

            if prompt_embeds is not None and pooled_prompt_embeds is None:
                raise ValueError(
                    "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
                )

            if (
                negative_prompt_embeds is not None
                and negative_pooled_prompt_embeds is None
            ):
                raise ValueError(
                    "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
                )

            # `prompt` needs more sophisticated handling when there are multiple
            # conditionings.
            if isinstance(self.controlnet, MultiControlNetModel):
                if isinstance(prompt, list):
                    logger.warning(
                        f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                        " prompts. The conditionings will be fixed across the prompts."
                    )

            # Check `image`
            is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
                self.controlnet, torch._dynamo.eval_frame.OptimizedModule
            )
            if (
                isinstance(self.controlnet, ControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)
            ):
                self.check_image(image, prompt, prompt_embeds)
            elif (
                isinstance(self.controlnet, MultiControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
            ):
                if not isinstance(image, list):
                    raise TypeError(
                        "For multiple controlnets: `image` must be type `list`"
                    )

                # When `image` is a nested list:
                # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
                elif any(isinstance(i, list) for i in image):
                    raise ValueError(
                        "A single batch of multiple conditionings are supported at the moment."
                    )
                elif len(image) != len(self.controlnet.nets):
                    raise ValueError(
                        f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
                    )

                for image_ in image:
                    self.check_image(image_, prompt, prompt_embeds)
            else:
                assert False

            # Check `controlnet_conditioning_scale`
            if (
                isinstance(self.controlnet, ControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)
            ):
                if not isinstance(controlnet_conditioning_scale, float):
                    raise TypeError(
                        "For single controlnet: `controlnet_conditioning_scale` must be type `float`."
                    )
            elif (
                isinstance(self.controlnet, MultiControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
            ):
                if isinstance(controlnet_conditioning_scale, list):
                    if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                        raise ValueError(
                            "A single batch of multiple conditionings are supported at the moment."
                        )
                elif isinstance(controlnet_conditioning_scale, list) and len(
                    controlnet_conditioning_scale
                ) != len(self.controlnet.nets):
                    raise ValueError(
                        "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                        " the same length as the number of controlnets"
                    )
            else:
                assert False

            if not isinstance(control_guidance_start, (tuple, list)):
                control_guidance_start = [control_guidance_start]

            if not isinstance(control_guidance_end, (tuple, list)):
                control_guidance_end = [control_guidance_end]

            if len(control_guidance_start) != len(control_guidance_end):
                raise ValueError(
                    f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
                )

            if isinstance(self.controlnet, MultiControlNetModel):
                if len(control_guidance_start) != len(self.controlnet.nets):
                    raise ValueError(
                        f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                    )

            for start, end in zip(control_guidance_start, control_guidance_end):
                if start >= end:
                    raise ValueError(
                        f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                    )
                if start < 0.0:
                    raise ValueError(
                        f"control guidance start: {start} can't be smaller than 0."
                    )
                if end > 1.0:
                    raise ValueError(
                        f"control guidance end: {end} can't be larger than 1.0."
                    )

            if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
                raise ValueError(
                    "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
                )

            if ip_adapter_image_embeds is not None:
                if not isinstance(ip_adapter_image_embeds, list):
                    raise ValueError(
                        f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                    )
                elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                    raise ValueError(
                        f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                    )

        # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image
        def check_image(self, image, prompt, prompt_embeds):
            image_is_pil = isinstance(image, PIL.Image.Image)
            image_is_tensor = isinstance(image, torch.Tensor)
            image_is_np = isinstance(image, np.ndarray)
            image_is_pil_list = isinstance(image, list) and isinstance(
                image[0], PIL.Image.Image
            )
            image_is_tensor_list = isinstance(image, list) and isinstance(
                image[0], torch.Tensor
            )
            image_is_np_list = isinstance(image, list) and isinstance(
                image[0], np.ndarray
            )

            if (
                not image_is_pil
                and not image_is_tensor
                and not image_is_np
                and not image_is_pil_list
                and not image_is_tensor_list
                and not image_is_np_list
            ):
                raise TypeError(
                    f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
                )

            if image_is_pil:
                image_batch_size = 1
            else:
                image_batch_size = len(image)

            if prompt is not None and isinstance(prompt, str):
                prompt_batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                prompt_batch_size = len(prompt)
            elif prompt_embeds is not None:
                prompt_batch_size = prompt_embeds.shape[0]

            if image_batch_size != 1 and image_batch_size != prompt_batch_size:
                raise ValueError(
                    f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
                )

        # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
        def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        ):
            image = self.control_image_processor.preprocess(
                image, height=height, width=width
            ).to(dtype=torch.float32)
            image_batch_size = image.shape[0]

            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # image batch size is the same as prompt batch size
                repeat_by = num_images_per_prompt

            image = image.repeat_interleave(repeat_by, dim=0)

            image = image.to(device=device, dtype=dtype)

            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)

            return image

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
        def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
        ):
            shape = (
                batch_size,
                num_channels_latents,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if latents is None:
                latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
            else:
                latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents

        # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
        def _get_add_time_ids(
            self,
            original_size,
            crops_coords_top_left,
            target_size,
            dtype,
            text_encoder_projection_dim=None,
        ):
            add_time_ids = list(original_size + crops_coords_top_left + target_size)

            passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids)
                + text_encoder_projection_dim
            )
            expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

            if expected_add_embed_dim != passed_add_embed_dim:
                raise ValueError(
                    f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
                )

            add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
            return add_time_ids

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
        def upcast_vae(self):
            dtype = self.vae.dtype
            self.vae.to(dtype=torch.float32)
            use_torch_2_0_or_xformers = isinstance(
                self.vae.decoder.mid_block.attentions[0].processor,
                (
                    AttnProcessor2_0,
                    XFormersAttnProcessor,
                    LoRAXFormersAttnProcessor,
                    LoRAAttnProcessor2_0,
                ),
            )
            # if xformers or torch_2_0 is used attention block does not need
            # to be in float32 which can save lots of memory
            if use_torch_2_0_or_xformers:
                self.vae.post_quant_conv.to(dtype)
                self.vae.decoder.conv_in.to(dtype)
                self.vae.decoder.mid_block.to(dtype)

        # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
        def get_guidance_scale_embedding(
            self,
            w: torch.Tensor,
            embedding_dim: int = 512,
            dtype: torch.dtype = torch.float32,
        ) -> torch.FloatTensor:
            """
            See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

            Args:
                w (`torch.Tensor`):
                    Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
                embedding_dim (`int`, *optional*, defaults to 512):
                    Dimension of the embeddings to generate.
                dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                    Data type of the generated embeddings.

            Returns:
                `torch.FloatTensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
            """
            assert len(w.shape) == 1
            w = w * 1000.0

            half_dim = embedding_dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            emb = w.to(dtype)[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            assert emb.shape == (w.shape[0], embedding_dim)
            return emb

        @property
        def guidance_scale(self):
            return self._guidance_scale

        @property
        def clip_skip(self):
            return self._clip_skip

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        @property
        def do_classifier_free_guidance(self):
            return (
                self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
            )

        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs

        @property
        def denoising_end(self):
            return self._denoising_end

        @property
        def num_timesteps(self):
            return self._num_timesteps

        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            self,
            prompt: str | list[str] = None,
            prompt_2: str | list[str] | None = None,
            image: PipelineImageInput = None,
            control_mask: PipelineImageInput | None = None,
            ip_adapter_index: int | list[int] | None = None,
            height: int | None = None,
            width: int | None = None,
            num_inference_steps: int = 50,
            denoising_end: float | None = None,
            guidance_scale: float = 5.0,
            negative_prompt: str | list[str] | None = None,
            negative_prompt_2: str | list[str] | None = None,
            num_images_per_prompt: int | None = 1,
            eta: float = 0.0,
            generator: torch.Generator | list[torch.Generator] | None = None,
            latents: torch.FloatTensor | None = None,
            image_for_noise: PipelineImageInput | None = None,
            strength: float = 1.0,
            prompt_embeds: torch.FloatTensor | None = None,
            negative_prompt_embeds: torch.FloatTensor | None = None,
            pooled_prompt_embeds: torch.FloatTensor | None = None,
            negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
            ip_adapter_image: PipelineImageInput | None = None,
            ip_adapter_image_embeds: list[torch.FloatTensor] | None = None,
            ip_adapter_mask: PipelineImageInput | None = None,
            ip_adapter_unconditional_noising_factor: list[float] = [0.0],
            output_type: str | None = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: dict[str, Any] | None = None,
            controlnet_conditioning_scale: float | list[float] = 1.0,
            guess_mode: bool = False,
            control_guidance_start: float | list[float] = 0.0,
            control_guidance_end: float | list[float] = 1.0,
            original_size: tuple[int, int] = None,
            crops_coords_top_left: tuple[int, int] = (0, 0),
            target_size: tuple[int, int] = None,
            negative_original_size: tuple[int, int] | None = None,
            negative_crops_coords_top_left: tuple[int, int] = (0, 0),
            negative_target_size: tuple[int, int] | None = None,
            clip_skip: int | None = None,
            callback_on_step_end: Callable[[int, int, dict], None] | None = None,
            callback_on_step_end_tensor_inputs: list[str] = ["latents"],
            tile_window_height: int = 1024,
            tile_window_width: int = 1024,
            tile_stride_height: int = 512,
            tile_stride_width: int = 512,
            debug_latents: bool = False,
            debug_per_pass_latents: bool = False,
            **kwargs,
        ):
            r"""
            The call function to the pipeline for generation.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                    used in both text-encoders.
                image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                        `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                    The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                    specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                    accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                    and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                    `init`, images must be passed as a list such that each element of the list can be correctly batched for
                    input to a single ControlNet.
                height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The height in pixels of the generated image. Anything below 512 pixels won't work well for
                    [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                    and checkpoints that are not specifically fine-tuned on low resolutions.
                width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The width in pixels of the generated image. Anything below 512 pixels won't work well for
                    [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                    and checkpoints that are not specifically fine-tuned on low resolutions.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                denoising_end (`float`, *optional*):
                    When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                    completed before it is intentionally prematurely terminated. As a result, the returned sample will
                    still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                    scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                    "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                    Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
                guidance_scale (`float`, *optional*, defaults to 5.0):
                    A higher guidance scale value encourages the model to generate images closely linked to the text
                    `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                    pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
                negative_prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
                    and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                eta (`float`, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                    to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                    generation deterministic.
                latents (`torch.FloatTensor`, *optional*):
                    Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor is generated by sampling using the supplied random `generator`.
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                    provided, text embeddings are generated from the `prompt` input argument.
                negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                    not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                    not provided, pooled text embeddings are generated from `prompt` input argument.
                negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                    weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
                    argument.
                ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
                ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                    Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
                    Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
                    if `do_classifier_free_guidance` is set to `True`.
                    If not provided, embeddings are computed from the `ip_adapter_image` input argument.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                    [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                    The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                    to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                    the corresponding scale as a list.
                guess_mode (`bool`, *optional*, defaults to `False`):
                    The ControlNet encoder tries to recognize the content of the input image even if you remove all
                    prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
                control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                    The percentage of total steps at which the ControlNet starts applying.
                control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                    The percentage of total steps at which the ControlNet stops applying.
                original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                    `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                    explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                    `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                    `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                    `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    For most cases, `target_size` should be set to the desired height and width of the generated image. If
                    not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                    section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                    micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
                negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                    To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                    micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
                negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    To negatively condition the generation process based on a target image resolution. It should be as same
                    as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
                callback_on_step_end (`Callable`, *optional*):
                    A function that calls at the end of each denoising steps during the inference. The function is called
                    with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                    callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                    `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeline class.

            Examples:

            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                    If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                    otherwise a `tuple` is returned containing the output images.
            """

            if image is None:
                image = []

            ip_adapter_masks = None

            # make sure only one of the either latents of image_for_noise is passed
            if latents is not None and image_for_noise is not None:
                raise ValueError(
                    "Only one of `latents` or `image_for_noise` can be passed."
                )

            # make sure that strength is between 0 and 1
            if not 0 <= strength <= 1:
                raise ValueError("`strength` should be between 0 and 1.")

            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )

            controlnet = (
                self.controlnet._orig_mod
                if is_compiled_module(self.controlnet)
                else self.controlnet
            )

            # align format for control guidance
            if not isinstance(control_guidance_start, list) and isinstance(
                control_guidance_end, list
            ):
                control_guidance_start = len(control_guidance_end) * [
                    control_guidance_start
                ]
            elif not isinstance(control_guidance_end, list) and isinstance(
                control_guidance_start, list
            ):
                control_guidance_end = len(control_guidance_start) * [
                    control_guidance_end
                ]
            elif not isinstance(control_guidance_start, list) and not isinstance(
                control_guidance_end, list
            ):
                mult = (
                    len(controlnet.nets)
                    if isinstance(controlnet, MultiControlNetModel)
                    else 1
                )
                control_guidance_start, control_guidance_end = (
                    mult * [control_guidance_start],
                    mult * [control_guidance_end],
                )

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                prompt_2,
                image,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                negative_pooled_prompt_embeds,
                controlnet_conditioning_scale,
                control_guidance_start,
                control_guidance_end,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            self._denoising_end = denoising_end

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            if isinstance(controlnet, MultiControlNetModel) and isinstance(
                controlnet_conditioning_scale, float
            ):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                    controlnet.nets
                )

            if isinstance(controlnet, ControlNetModel):
                global_pool_conditions = controlnet.config.global_pool_conditions
            elif isinstance(controlnet, MultiControlNetModel):
                if len(controlnet.nets) != 0:
                    global_pool_conditions = controlnet.nets[
                        0
                    ].config.global_pool_conditions
                else:
                    global_pool_conditions = guess_mode

            guess_mode = guess_mode or global_pool_conditions

            # 3.1 Encode input prompt
            text_encoder_lora_scale = (
                self.cross_attention_kwargs.get("scale", None)
                if self.cross_attention_kwargs is not None
                else None
            )
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt,
                prompt_2,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )

            # 3.2 Encode ip_adapter_image
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds, ip_adapter_masks = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    ip_adapter_mask,
                    width,
                    height,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                    ip_adapter_unconditional_noising_factor,
                )
            if isinstance(ip_adapter_index, int):
                ip_adapter_index = [ip_adapter_index]

            # 4. Prepare image
            if isinstance(controlnet, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
            elif isinstance(controlnet, MultiControlNetModel):
                images = []

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                if len(image) > 0:
                    height, width = image[0].shape[-2:]
            else:
                assert False

            # 4.1 Prepare image for noise

            if image_for_noise is not None:
                image_for_noise = prepare_noise_image(
                    image_for_noise,
                )

                # if the height and width is set
                # resize the image_for_noise to the same size
                if height is not None and width is not None:
                    image_for_noise = F.interpolate(
                        image_for_noise, size=(height, width), mode="bilinear"
                    )
                else:
                    height, width = image_for_noise.shape[-2:]

            # 4.2 Prepare control mask
            controlnet_masks = []
            if control_mask is not None:
                for mask in control_mask:
                    mask = np.array(mask)
                    mask_tensor = torch.from_numpy(mask).to(
                        device=device, dtype=prompt_embeds.dtype
                    )
                    mask_tensor = mask_tensor[:, :, 0] / 255.0
                    mask_tensor = mask_tensor[None, None]
                    h, w = mask_tensor.shape[-2:]
                    control_mask_list = []
                    for scale in [8, 8, 8, 16, 16, 16, 32, 32, 32]:
                        # Python uses IEEE 754 rounding rules, we need to add a small value to round like the unet model
                        w_n = round((w + 0.01) / 8)
                        h_n = round((h + 0.01) / 8)
                        if scale in [16, 32]:
                            w_n = round((w_n + 0.01) / 2)
                            h_n = round((h_n + 0.01) / 2)
                        if scale == 32:
                            w_n = round((w_n + 0.01) / 2)
                            h_n = round((h_n + 0.01) / 2)
                        scale_mask_weight_image_tensor = F.interpolate(
                            mask_tensor, (h_n, w_n), mode="bilinear"
                        )
                        control_mask_list.append(scale_mask_weight_image_tensor)
                    controlnet_masks.append(control_mask_list)

            # 5. Prepare timesteps
            if image_for_noise is not None:
                # reworks for automatic1111/comfyui style timesteps, e.g. always the same
                # regardless of strength
                new_num_inference_steps = int(num_inference_steps * (1 / strength))
                self.scheduler.set_timesteps(new_num_inference_steps, device=device)
                timesteps, num_inference_steps = self.get_timesteps(
                    new_num_inference_steps, strength, device
                )
                latent_timestep = timesteps[:1].repeat(
                    batch_size * num_images_per_prompt
                )
            else:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps

            self._num_timesteps = len(timesteps)

            # 6. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            if image_for_noise is not None:
                latents = self.prepare_image_latents(
                    image_for_noise,
                    latent_timestep,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                )
            else:
                latents = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
                )

            view_batch_size = 1
            circular_padding = False
            latent_height = latents.shape[2]
            latent_width = latents.shape[3]

            window_height = tile_window_height // 8
            window_width = tile_window_width // 8
            stride_height = tile_stride_height // 8
            stride_width = tile_stride_width // 8

            window_height = min(window_height, latent_height)
            window_width = min(window_width, latent_width)

            views = get_views(
                latent_height,
                latent_width,
                window_height=window_height,
                window_width=window_width,
                stride_height=stride_height,
                stride_width=stride_width,
                circular_padding=circular_padding,
            )

            views_batch = [
                views[i : i + view_batch_size]
                for i in range(0, len(views), view_batch_size)
            ]
            views_scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(
                views_batch
            )
            count = torch.zeros_like(latents)
            value = torch.zeros_like(latents)

            # 6.5 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                    batch_size * num_images_per_prompt
                )
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=self.unet.config.time_cond_proj_dim,
                ).to(device=device, dtype=latents.dtype)

            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7.1 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(
                    keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
                )

            # 7.2 Prepare added time ids & embeddings
            if isinstance(image, list):
                if len(image) > 0:
                    original_size = original_size or image[0].shape[-2:]
                else:
                    original_size = (height, width)
            else:
                original_size = original_size or image.shape[-2:]
            target_size = target_size or (height, width)

            add_text_embeds = pooled_prompt_embeds
            if self.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

            add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )

            if negative_original_size is not None and negative_target_size is not None:
                negative_add_time_ids = self._get_add_time_ids(
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
            else:
                negative_add_time_ids = add_time_ids

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
                add_text_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, add_text_embeds], dim=0
                )
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(
                batch_size * num_images_per_prompt, 1
            )

            # 8. Denoising loop
            num_warmup_steps = (
                len(timesteps) - num_inference_steps * self.scheduler.order
            )

            # 8.1 Apply denoising_end
            if (
                self.denoising_end is not None
                and isinstance(self.denoising_end, float)
                and self.denoising_end > 0
                and self.denoising_end < 1
            ):
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (
                            self.denoising_end
                            * self.scheduler.config.num_train_timesteps
                        )
                    )
                )
                num_inference_steps = len(
                    list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
                )
                timesteps = timesteps[:num_inference_steps]

            # if there is only one view move the latents to the gpu
            if len(views_batch) == 1:
                latents = latents.to(device=device)
                value = value.to(device=device)

            debug_latents_list = []
            debug_per_pass_latents_list = []

            is_unet_compiled = is_compiled_module(self.unet)
            is_controlnet_compiled = is_compiled_module(self.controlnet)
            is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    count.zero_()
                    value.zero_()

                    per_pass_latents = []

                    current_prompt_embeds = prompt_embeds
                    # Relevant thread:
                    # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                    if (
                        is_unet_compiled and is_controlnet_compiled
                    ) and is_torch_higher_equal_2_1:
                        torch._inductor.cudagraph_mark_step_begin()
                    # expand the latents if we are doing classifier free guidance
                    for j, batch_view in enumerate(views_batch):
                        vb_size = len(batch_view)

                        latents_for_view = torch.cat(
                            [
                                latents[:, :, h_start:h_end, w_start:w_end]
                                for h_start, h_end, w_start, w_end in batch_view
                            ]
                        )

                        control_image_for_view = None
                        if image:
                            # take a view of the image
                            if isinstance(image, list):
                                control_image_for_view = []
                                for img in image:
                                    control_image_for_view.append(
                                        torch.cat(
                                            [
                                                img[
                                                    :,
                                                    :,
                                                    h_start * 8 : h_end * 8,
                                                    w_start * 8 : w_end * 8,
                                                ]
                                                for h_start, h_end, w_start, w_end in batch_view
                                            ]
                                        )
                                    )
                            else:
                                control_image_for_view = torch.cat(
                                    [
                                        image[
                                            :,
                                            :,
                                            h_start * 8 : h_end * 8,
                                            w_start * 8 : w_end * 8,
                                        ]
                                        for h_start, h_end, w_start, w_end in batch_view
                                    ]
                                )

                        latents_for_view = latents_for_view.to(
                            device=device, dtype=latents.dtype
                        )
                        if control_image_for_view is not None:
                            if isinstance(control_image_for_view, list):
                                control_image_for_view = [
                                    c.to(device=device, dtype=self.controlnet.dtype)
                                    for c in control_image_for_view
                                ]
                            else:
                                control_image_for_view = control_image_for_view.to(
                                    device=device, dtype=self.controlnet.dtype
                                )

                        # rematch block's scheduler status
                        self.scheduler.__dict__.update(views_scheduler_status[j])

                        # expand the latents if we are doing classifier free guidance

                        latent_model_input = (
                            torch.cat([latents_for_view] * 2)
                            if self.do_classifier_free_guidance
                            else latents_for_view
                        )
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        added_cond_kwargs = {
                            "text_embeds": add_text_embeds,
                            "time_ids": add_time_ids,
                        }

                        # controlnet(s) inference
                        if guess_mode and self.do_classifier_free_guidance:
                            # Infer ControlNet only for the conditional batch.
                            control_model_input = latents
                            control_model_input = self.scheduler.scale_model_input(
                                control_model_input, t
                            )
                            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                            controlnet_added_cond_kwargs = {
                                "text_embeds": add_text_embeds.chunk(2)[1],
                                "time_ids": add_time_ids.chunk(2)[1],
                            }
                        else:
                            control_model_input = latent_model_input
                            controlnet_prompt_embeds = prompt_embeds
                            controlnet_added_cond_kwargs = added_cond_kwargs

                        if isinstance(controlnet_keep[i], list):
                            cond_scale = [
                                c * s
                                for c, s in zip(
                                    controlnet_conditioning_scale, controlnet_keep[i]
                                )
                            ]
                        else:
                            controlnet_cond_scale = controlnet_conditioning_scale
                            if isinstance(controlnet_cond_scale, list):
                                controlnet_cond_scale = controlnet_cond_scale[0]
                            cond_scale = controlnet_cond_scale * controlnet_keep[i]

                        if (
                            isinstance(controlnet, MultiControlNetModel)
                            and len(controlnet.nets) == 0
                        ):
                            down_block_res_samples, mid_block_res_sample = None, None
                        else:
                            if control_image_for_view is not None:
                                # controlnet(s) inference
                                if guess_mode and self.do_classifier_free_guidance:
                                    # Infer ControlNet only for the conditional batch.
                                    control_model_input = latents
                                    control_model_input = (
                                        self.scheduler.scale_model_input(
                                            control_model_input, t
                                        )
                                    )
                                    controlnet_prompt_embeds = (
                                        current_prompt_embeds.chunk(2)[1]
                                    )
                                    controlnet_added_cond_kwargs = {
                                        "text_embeds": add_text_embeds.chunk(2)[1],
                                        "time_ids": add_time_ids.chunk(2)[1],
                                    }
                                else:
                                    control_model_input = latent_model_input
                                    controlnet_prompt_embeds = current_prompt_embeds
                                    controlnet_added_cond_kwargs = added_cond_kwargs

                                if isinstance(controlnet_keep[i], list):
                                    cond_scale = [
                                        c * s
                                        for c, s in zip(
                                            controlnet_conditioning_scale,
                                            controlnet_keep[i],
                                        )
                                    ]
                                else:
                                    controlnet_cond_scale = (
                                        controlnet_conditioning_scale
                                    )
                                    if isinstance(controlnet_cond_scale, list):
                                        controlnet_cond_scale = controlnet_cond_scale[0]
                                    cond_scale = (
                                        controlnet_cond_scale * controlnet_keep[i]
                                    )

                                if (
                                    ip_adapter_image_embeds is None
                                    and ip_adapter_image is not None
                                ):
                                    encoder_hidden_states = (
                                        self.unet.process_encoder_hidden_states(
                                            prompt_embeds,
                                            {"image_embeds": image_embeds},
                                        )
                                    )
                                    ip_adapter_image_embeds = encoder_hidden_states[1]

                                down_block_res_samples = None
                                mid_block_res_sample = None

                                for controlnet_index, _ in enumerate(controlnet.nets):
                                    ipa_index = ip_adapter_index[controlnet_index]
                                    if ipa_index is not None:
                                        control_prompt_embeds = ip_adapter_image_embeds[
                                            ipa_index
                                        ].squeeze(1)
                                    else:
                                        control_prompt_embeds = controlnet_prompt_embeds

                                    down_samples, mid_sample = self.controlnet.nets[
                                        controlnet_index
                                    ](
                                        control_model_input,
                                        t,
                                        encoder_hidden_states=control_prompt_embeds,
                                        controlnet_cond=control_image_for_view[
                                            controlnet_index
                                        ],
                                        conditioning_scale=cond_scale[controlnet_index],
                                        guess_mode=guess_mode,
                                        added_cond_kwargs=controlnet_added_cond_kwargs,
                                        return_dict=False,
                                    )

                                    if (
                                        len(controlnet_masks) > controlnet_index
                                        and controlnet_masks[controlnet_index]
                                        is not None
                                    ):
                                        down_samples = [
                                            down_sample * mask_weight
                                            for down_sample, mask_weight in zip(
                                                down_samples,
                                                controlnet_masks[controlnet_index],
                                            )
                                        ]
                                        mid_sample *= controlnet_masks[
                                            controlnet_index
                                        ][-1]

                                    if (
                                        down_block_res_samples is None
                                        and mid_block_res_sample is None
                                    ):
                                        down_block_res_samples = down_samples
                                        mid_block_res_sample = mid_sample
                                    else:
                                        down_block_res_samples = [
                                            samples_prev + samples_curr
                                            for samples_prev, samples_curr in zip(
                                                down_block_res_samples, down_samples
                                            )
                                        ]
                                        mid_block_res_sample += mid_sample

                            else:
                                down_block_res_samples = None
                                mid_block_res_sample = None

                        if (
                            guess_mode
                            and self.do_classifier_free_guidance
                            and mid_block_res_sample is not None
                        ):
                            # Inferred ControlNet only for the conditional batch.
                            # To apply the output of ControlNet to both the unconditional and conditional batches,
                            # add 0 to the unconditional batch to keep it unchanged.
                            down_block_res_samples = [
                                torch.cat([torch.zeros_like(d), d])
                                for d in down_block_res_samples
                            ]
                            mid_block_res_sample = torch.cat(
                                [
                                    torch.zeros_like(mid_block_res_sample),
                                    mid_block_res_sample,
                                ]
                            )

                        if (
                            ip_adapter_image is not None
                            or ip_adapter_image_embeds is not None
                        ):
                            added_cond_kwargs["image_embeds"] = image_embeds
                            if ip_adapter_masks is not None:
                                self._cross_attention_kwargs[
                                    "ip_adapter_masks"
                                ] = ip_adapter_masks

                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        # compute the previous noisy sample x_t -> x_t-1
                        latents_denoised_batch = self.scheduler.step(
                            noise_pred, t, latents_for_view, **extra_step_kwargs
                        ).prev_sample

                        # save views scheduler status after sample
                        views_scheduler_status[j] = copy.deepcopy(
                            self.scheduler.__dict__
                        )

                        # extract value from batch
                        for latents_view_denoised, (
                            h_start,
                            h_end,
                            w_start,
                            w_end,
                        ) in zip(latents_denoised_batch.chunk(vb_size), batch_view):
                            latents_view_denoised = latents_view_denoised.to(
                                device=value.device
                            )
                            weights = torch.ones_like(latents_view_denoised)
                            if circular_padding and w_end > latents.shape[3]:
                                # Case for circular padding
                                value[
                                    :, :, h_start:h_end, w_start:
                                ] += latents_view_denoised[
                                    :, :, h_start:h_end, : latents.shape[3] - w_start
                                ].cpu()
                                value[
                                    :, :, h_start:h_end, : w_end - latents.shape[3]
                                ] += latents_view_denoised[
                                    :, :, h_start:h_end, latents.shape[3] - w_start :
                                ].cpu()
                                count[:, :, h_start:h_end, w_start:] += weights
                                count[
                                    :, :, h_start:h_end, : w_end - latents.shape[3]
                                ] += weights
                            else:

                                value[
                                    :, :, h_start:h_end, w_start:w_end
                                ] += latents_view_denoised
                                count[:, :, h_start:h_end, w_start:w_end] += weights

                            if debug_per_pass_latents:
                                per_pass_latents.append(
                                    (
                                        (h_start, h_end, w_start, w_end),
                                        latents_view_denoised.cpu(),
                                        weights.cpu(),
                                    )
                                )
                    if debug_per_pass_latents:
                        debug_per_pass_latents_list.append(per_pass_latents)

                    latents = torch.where(count > 0, value / count, value)

                    if debug_latents:
                        debug_latents_list.append(latents.cpu())

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop(
                            "prompt_embeds", prompt_embeds
                        )
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            if not output_type == "latent":
                # make sure the VAE is in float32 mode, as it overflows in float16
                needs_upcasting = (
                    self.vae.dtype == torch.float16 and self.vae.config.force_upcast
                )

                try:
                    if needs_upcasting:
                        self.upcast_vae()
                        latents = latents.to(
                            next(iter(self.vae.post_quant_conv.parameters())).dtype
                        )

                    # unscale/denormalize the latents
                    # denormalize with the mean and std if available and not None
                    has_latents_mean = (
                        hasattr(self.vae.config, "latents_mean")
                        and self.vae.config.latents_mean is not None
                    )
                    has_latents_std = (
                        hasattr(self.vae.config, "latents_std")
                        and self.vae.config.latents_std is not None
                    )
                    if has_latents_mean and has_latents_std:
                        latents_mean = (
                            torch.tensor(self.vae.config.latents_mean)
                            .view(1, 4, 1, 1)
                            .to(latents.device, latents.dtype)
                        )
                        latents_std = (
                            torch.tensor(self.vae.config.latents_std)
                            .view(1, 4, 1, 1)
                            .to(latents.device, latents.dtype)
                        )
                        latents = (
                            latents * latents_std / self.vae.config.scaling_factor
                            + latents_mean
                        )

                        if debug_per_pass_latents:
                            temp_debug_per_pass_latents_list = []
                            for per_pass_latents in debug_per_pass_latents_list:
                                temp_per_pass_latents = []
                                for per_pass_latent in per_pass_latents:
                                    the_latents = per_pass_latent[1]
                                    per_pass_latent[1] = (
                                        the_latents
                                        * latents_std
                                        / self.vae.config.scaling_factor
                                    )
                                    +latents_mean
                                    temp_per_pass_latents.append(per_pass_latent)

                                temp_debug_per_pass_latents_list.append(
                                    temp_per_pass_latents
                                )
                            debug_per_pass_latents_list = (
                                temp_debug_per_pass_latents_list
                            )

                        if debug_latents:
                            temp_debug_latents_list = []
                            for the_latents in debug_latents_list:
                                the_latents = (
                                    the_latents
                                    * latents_std
                                    / self.vae.config.scaling_factor
                                    + latents_mean
                                )
                                temp_debug_latents_list.append(the_latents)
                        debug_latents_list = temp_debug_latents_list

                    else:
                        latents = latents / self.vae.config.scaling_factor

                        if debug_per_pass_latents:
                            temp_debug_per_pass_latents_list = []
                            for per_pass_latents in debug_per_pass_latents_list:
                                temp_per_pass_latents = []
                                for per_pass_latent in per_pass_latents:
                                    the_per_pass_latent = (
                                        per_pass_latent[1]
                                        / self.vae.config.scaling_factor
                                    )
                                    temp_per_pass_latents.append(
                                        (
                                            per_pass_latent[0],
                                            the_per_pass_latent,
                                            per_pass_latent[2],
                                        )
                                    )

                                temp_debug_per_pass_latents_list.append(
                                    temp_per_pass_latents
                                )
                            debug_per_pass_latents_list = (
                                temp_debug_per_pass_latents_list
                            )

                        if debug_latents:
                            temp_debug_latents_list = []
                            for the_latents in debug_latents_list:
                                the_latents = (
                                    the_latents / self.vae.config.scaling_factor
                                )
                                temp_debug_latents_list.append(the_latents)

                            debug_latents_list = temp_debug_latents_list

                    image = self.vae.decode(latents, return_dict=False)[0]

                    if debug_per_pass_latents:
                        temp_debug_per_pass_latents_list = []
                        for per_pass_latents in debug_per_pass_latents_list:
                            temp_per_pass_latents = []
                            for per_pass_latent in per_pass_latents:
                                dims = per_pass_latent[0]
                                the_per_pass_latent = self.vae.decode(
                                    per_pass_latent[1].to("cuda"), return_dict=False
                                )[0]
                                the_per_pass_latent = self.image_processor.postprocess(
                                    the_per_pass_latent,
                                    output_type="pil",
                                )

                                # need to convert the greyscale weight images to images
                                weight_images = self.image_processor.postprocess(
                                    per_pass_latent[2],
                                    output_type="pil",
                                )

                                temp_per_pass_latents.append(
                                    (dims, the_per_pass_latent, weight_images)
                                )

                            temp_debug_per_pass_latents_list.append(
                                temp_per_pass_latents
                            )
                        debug_per_pass_latents_list = temp_debug_per_pass_latents_list

                    if debug_latents:
                        temp_debug_latents_list = []
                        for the_latents in debug_latents_list:
                            # this needs to handle batches
                            the_latents = self.vae.decode(
                                the_latents.to("cuda"), return_dict=False
                            )[0]
                            the_latents = self.image_processor.postprocess(
                                the_latents, output_type="pil"
                            )
                            temp_debug_latents_list.append(the_latents)
                        debug_latents_list = temp_debug_latents_list

                finally:
                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=torch.float16)
            else:
                image = latents

            if not output_type == "latent":
                # apply watermark if available
                if self.watermark is not None:
                    image = self.watermark.apply_watermark(image)

                image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, debug_latents_list, debug_per_pass_latents_list)

            result = StableDiffusionXLPipelineOutput(images=image)
            result.debug_latents = debug_latents_list
            result.debug_per_pass_latents = debug_per_pass_latents_list
            return result

    class StableDiffusionControlNetPipeline(
        DiffusionPipeline,
        StableDiffusionMixin,
        TextualInversionLoaderMixin,
        LoraLoaderMixin,
        IPAdapterMixin,
        FromSingleFileMixin,
    ):
        r"""
        Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

        This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
        implemented for all pipelines (downloading, saving, running on a particular device, etc.).

        The pipeline also inherits the following loading methods:
            - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
            - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
            - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
            - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
            - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

        Args:
            vae ([`AutoencoderKL`]):
                Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
            text_encoder ([`~transformers.CLIPTextModel`]):
                Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
            tokenizer ([`~transformers.CLIPTokenizer`]):
                A `CLIPTokenizer` to tokenize text.
            unet ([`UNet2DConditionModel`]):
                A `UNet2DConditionModel` to denoise the encoded image latents.
            controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
                Provides additional conditioning to the `unet` during the denoising process. If you set multiple
                ControlNets as a list, the outputs from each ControlNet are added together to create one combined
                additional conditioning.
            scheduler ([`SchedulerMixin`]):
                A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
                [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
            safety_checker ([`StableDiffusionSafetyChecker`]):
                Classification module that estimates whether generated images could be considered offensive or harmful.
                Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
                about a model's potential harms.
            feature_extractor ([`~transformers.CLIPImageProcessor`]):
                A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
        """

        model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
        _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
        _exclude_from_cpu_offload = ["safety_checker"]
        _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

        def get_timesteps(self, num_inference_steps, strength, device):
            # get the original timestep using init_timestep
            init_timestep = min(
                int(num_inference_steps * strength), num_inference_steps
            )

            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)

            return timesteps, num_inference_steps - t_start

        def prepare_image_latents(
            self,
            image,
            timestep,
            batch_size,
            num_images_per_prompt,
            dtype,
            device,
            generator=None,
        ):
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )

            image = image.to(device=device, dtype=dtype)

            batch_size = batch_size * num_images_per_prompt

            if image.shape[1] == 4:
                init_latents = image

            else:
                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                elif isinstance(generator, list):
                    init_latents = [
                        retrieve_latents(
                            self.vae.encode(image[i : i + 1]), generator=generator[i]
                        )
                        for i in range(batch_size)
                    ]
                    init_latents = torch.cat(init_latents, dim=0)
                else:
                    init_latents = retrieve_latents(
                        self.vae.encode(image), generator=generator
                    )

                init_latents = self.vae.config.scaling_factor * init_latents

            if (
                batch_size > init_latents.shape[0]
                and batch_size % init_latents.shape[0] == 0
            ):
                # expand init_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate(
                    "len(prompt) != len(image)",
                    "1.0.0",
                    deprecation_message,
                    standard_warn=False,
                )
                additional_image_per_prompt = batch_size // init_latents.shape[0]
                init_latents = torch.cat(
                    [init_latents] * additional_image_per_prompt, dim=0
                )
            elif (
                batch_size > init_latents.shape[0]
                and batch_size % init_latents.shape[0] != 0
            ):
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                init_latents = torch.cat([init_latents], dim=0)

            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents

            return latents

        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            controlnet: ControlNetModel
            | list[ControlNetModel]
            | tuple[ControlNetModel]
            | MultiControlNetModel
            | None = None,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
        ):
            super().__init__()

            if controlnet is None:
                controlnet = MultiControlNetModel([])

            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )

            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )

            if isinstance(controlnet, (list, tuple)):
                controlnet = MultiControlNetModel(controlnet)

            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
            )
            self.ip_adapter_mask_processor = IPAdapterMaskProcessor()
            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor,
                do_convert_rgb=True,
                do_normalize=False,
            )
            self.register_to_config(requires_safety_checker=requires_safety_checker)

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
        def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: torch.FloatTensor | None = None,
            negative_prompt_embeds: torch.FloatTensor | None = None,
            lora_scale: float | None = None,
            **kwargs,
        ):
            deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
            deprecate(
                "_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False
            )

            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                **kwargs,
            )

            # concatenate for backwards comp
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

            return prompt_embeds

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
        def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: torch.FloatTensor | None = None,
            negative_prompt_embeds: torch.FloatTensor | None = None,
            lora_scale: float | None = None,
            clip_skip: int | None = None,
        ):
            r"""
            Encodes the prompt into text encoder hidden states.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    prompt to be encoded
                device: (`torch.device`):
                    torch device
                num_images_per_prompt (`int`):
                    number of images that should be generated per prompt
                do_classifier_free_guidance (`bool`):
                    whether to use classifier free guidance or not
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
                negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                    argument.
                lora_scale (`float`, *optional*):
                    A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
            """
            # set lora scale so that monkey patched LoRA
            # function of text encoder can correctly access it
            if lora_scale is not None and isinstance(self, LoraLoaderMixin):
                self._lora_scale = lora_scale

                # dynamically adjust the LoRA scale
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            if prompt_embeds is None:
                # textual inversion: process multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
                ):
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                if clip_skip is None:
                    prompt_embeds = self.text_encoder(
                        text_input_ids.to(device), attention_mask=attention_mask
                    )
                    prompt_embeds = prompt_embeds[0]
                else:
                    prompt_embeds = self.text_encoder(
                        text_input_ids.to(device),
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    # Access the `hidden_states` first, that contains a tuple of
                    # all the hidden states from the encoder layers. Then index into
                    # the tuple to access the hidden states from the desired layer.
                    prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                    # We also need to apply the final LayerNorm here to not mess with the
                    # representations. The `last_hidden_states` that we typically use for
                    # obtaining the final prompt representations passes through the LayerNorm
                    # layer.
                    prompt_embeds = self.text_encoder.text_model.final_layer_norm(
                        prompt_embeds
                    )

            if self.text_encoder is not None:
                prompt_embeds_dtype = self.text_encoder.dtype
            elif self.unet is not None:
                prompt_embeds_dtype = self.unet.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )

            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                uncond_tokens: list[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                # textual inversion: process multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    uncond_tokens = self.maybe_convert_prompt(
                        uncond_tokens, self.tokenizer
                    )

                max_length = prompt_embeds.shape[1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
                ):
                    attention_mask = uncond_input.attention_mask.to(device)
                else:
                    attention_mask = None

                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=prompt_embeds_dtype, device=device
                )

                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_images_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )

            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

            return prompt_embeds, negative_prompt_embeds

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
        def encode_image(
            self,
            image,
            device,
            num_images_per_prompt,
            output_hidden_states=None,
            unconditional_noising_factor=None,
        ):
            dtype = next(self.image_encoder.parameters()).dtype

            needs_encoding = not isinstance(image, torch.Tensor)
            if needs_encoding:
                image = self.feature_extractor(image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)

            additional_noise_for_uncond = (
                torch.rand_like(image) * unconditional_noising_factor
            )

            if output_hidden_states:
                if needs_encoding:
                    image_encoded = self.image_encoder(image, output_hidden_states=True)
                    image_enc_hidden_states = image_encoded.hidden_states[-2]
                else:
                    image_enc_hidden_states = image.unsqueeze(0).unsqueeze(0)
                image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(
                    num_images_per_prompt, dim=0
                )

                if needs_encoding:
                    uncond_image_encoded = self.image_encoder(
                        additional_noise_for_uncond, output_hidden_states=True
                    )
                    uncond_image_enc_hidden_states = uncond_image_encoded.hidden_states[
                        -2
                    ]
                else:
                    uncond_image_enc_hidden_states = (
                        additional_noise_for_uncond.unsqueeze(0).unsqueeze(0)
                    )
                uncond_image_enc_hidden_states = (
                    uncond_image_enc_hidden_states.repeat_interleave(
                        num_images_per_prompt, dim=0
                    )
                )
                return image_enc_hidden_states, uncond_image_enc_hidden_states
            else:
                if needs_encoding:
                    image_encoded = self.image_encoder(image)
                    image_embeds = image_encoded.image_embeds
                else:
                    image_embeds = image.unsqueeze(0).unsqueeze(0)
                if needs_encoding:
                    uncond_image_encoded = self.image_encoder(
                        additional_noise_for_uncond
                    )
                    uncond_image_embeds = uncond_image_encoded.image_embeds
                else:
                    uncond_image_embeds = additional_noise_for_uncond.unsqueeze(
                        0
                    ).unsqueeze(0)

                image_embeds = image_embeds.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                uncond_image_embeds = uncond_image_embeds.repeat_interleave(
                    num_images_per_prompt, dim=0
                )

                return image_embeds, uncond_image_embeds

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
        def prepare_ip_adapter_image_embeds(
            self,
            ip_adapter_image,
            ip_adapter_image_embeds,
            ip_adapter_mask,
            width,
            height,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            unconditional_noising_factors=None,
        ):
            if ip_adapter_image_embeds is None:
                if not isinstance(ip_adapter_image, list):
                    ip_adapter_image = [ip_adapter_image]

                if len(ip_adapter_image) != len(
                    self.unet.encoder_hid_proj.image_projection_layers
                ):
                    raise ValueError(
                        f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                    )

                image_embeds = []

                if unconditional_noising_factors is None:
                    unconditional_noising_factors = [0.0] * len(ip_adapter_image)
                else:
                    if not isinstance(unconditional_noising_factors, list):
                        unconditional_noising_factors = [unconditional_noising_factors]

                    if len(unconditional_noising_factors) != len(ip_adapter_image):
                        raise ValueError(
                            f"`unconditional_noising_factors` must have same length as the number of IP Adapters. Got {len(unconditional_noising_factors)} values and {len(ip_adapter_image)} IP Adapters."
                        )

                for (
                    single_ip_adapter_image,
                    unconditional_noising_factor,
                    image_proj_layer,
                ) in zip(
                    ip_adapter_image,
                    unconditional_noising_factors,
                    self.unet.encoder_hid_proj.image_projection_layers,
                ):
                    output_hidden_state = not isinstance(
                        image_proj_layer, ImageProjection
                    )
                    (
                        single_image_embeds,
                        single_negative_image_embeds,
                    ) = self.encode_image(
                        single_ip_adapter_image,
                        device,
                        1,
                        output_hidden_state,
                        unconditional_noising_factor,
                    )
                    single_image_embeds = torch.stack(
                        [single_image_embeds] * num_images_per_prompt, dim=0
                    )
                    single_negative_image_embeds = torch.stack(
                        [single_negative_image_embeds] * num_images_per_prompt, dim=0
                    )

                    if do_classifier_free_guidance:
                        single_image_embeds = torch.cat(
                            [single_negative_image_embeds, single_image_embeds]
                        )
                        single_image_embeds = single_image_embeds.to(device)

                    image_embeds.append(single_image_embeds)
            else:
                repeat_dims = [1]
                image_embeds = []
                for single_image_embeds in ip_adapter_image_embeds:
                    if do_classifier_free_guidance:
                        (
                            single_negative_image_embeds,
                            single_image_embeds,
                        ) = single_image_embeds.chunk(2)
                        single_image_embeds = single_image_embeds.repeat(
                            num_images_per_prompt,
                            *(repeat_dims * len(single_image_embeds.shape[1:])),
                        )
                        single_negative_image_embeds = (
                            single_negative_image_embeds.repeat(
                                num_images_per_prompt,
                                *(
                                    repeat_dims
                                    * len(single_negative_image_embeds.shape[1:])
                                ),
                            )
                        )
                        single_image_embeds = torch.cat(
                            [single_negative_image_embeds, single_image_embeds]
                        )
                    else:
                        single_image_embeds = single_image_embeds.repeat(
                            num_images_per_prompt,
                            *(repeat_dims * len(single_image_embeds.shape[1:])),
                        )
                    image_embeds.append(single_image_embeds)

            ip_adapter_masks = None
            if ip_adapter_mask is not None:
                ip_adapter_masks = self.ip_adapter_mask_processor.preprocess(
                    ip_adapter_mask, height=height, width=width
                )
                ip_adapter_masks = [mask.unsqueeze(0) for mask in ip_adapter_masks]

                reshaped_ip_adapter_masks = []
                for ip_img, mask in zip(image_embeds, ip_adapter_masks):
                    if isinstance(ip_img, list):
                        num_images = len(ip_img)
                        mask = mask.repeat(1, num_images, 1, 1)

                    reshaped_ip_adapter_masks.append(mask)

                ip_adapter_masks = reshaped_ip_adapter_masks

            return image_embeds, ip_adapter_masks

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
        def run_safety_checker(self, image, device, dtype):
            if self.safety_checker is None:
                has_nsfw_concept = None
            else:
                if torch.is_tensor(image):
                    feature_extractor_input = self.image_processor.postprocess(
                        image, output_type="pil"
                    )
                else:
                    feature_extractor_input = self.image_processor.numpy_to_pil(image)
                safety_checker_input = self.feature_extractor(
                    feature_extractor_input, return_tensors="pt"
                ).to(device)
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            return image, has_nsfw_concept

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
        def decode_latents(self, latents):
            deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
            deprecate(
                "decode_latents", "1.0.0", deprecation_message, standard_warn=False
            )

            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            return image

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
        def prepare_extra_step_kwargs(self, generator, eta):
            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]

            accepts_eta = "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()
            )
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            # check if the scheduler accepts generator
            accepts_generator = "generator" in set(
                inspect.signature(self.scheduler.step).parameters.keys()
            )
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            return extra_step_kwargs

        def check_inputs(
            self,
            prompt,
            image,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            callback_on_step_end_tensor_inputs=None,
        ):
            if callback_steps is not None and (
                not isinstance(callback_steps, int) or callback_steps <= 0
            ):
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )

            if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs
                for k in callback_on_step_end_tensor_inputs
            ):
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
                )

            if prompt is not None and prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                    " only forward one of the two."
                )
            elif prompt is None and prompt_embeds is None:
                raise ValueError(
                    "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
                )
            elif prompt is not None and (
                not isinstance(prompt, str) and not isinstance(prompt, list)
            ):
                raise ValueError(
                    f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
                )

            if negative_prompt is not None and negative_prompt_embeds is not None:
                raise ValueError(
                    f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                    f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
                )

            if prompt_embeds is not None and negative_prompt_embeds is not None:
                if prompt_embeds.shape != negative_prompt_embeds.shape:
                    raise ValueError(
                        "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                        f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                        f" {negative_prompt_embeds.shape}."
                    )

            # Check `image`
            is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
                self.controlnet, torch._dynamo.eval_frame.OptimizedModule
            )
            if (
                isinstance(self.controlnet, ControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)
            ):
                self.check_image(image, prompt, prompt_embeds)
            elif (
                isinstance(self.controlnet, MultiControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
            ):
                if not isinstance(image, list):
                    raise TypeError(
                        "For multiple controlnets: `image` must be type `list`"
                    )

                # When `image` is a nested list:
                # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
                elif any(isinstance(i, list) for i in image):
                    transposed_image = [list(t) for t in zip(*image)]
                    if len(transposed_image) != len(self.controlnet.nets):
                        raise ValueError(
                            f"For multiple controlnets: if you pass`image` as a list of list, each sublist must have the same length as the number of controlnets, but the sublists in `image` got {len(transposed_image)} images and {len(self.controlnet.nets)} ControlNets."
                        )
                    for image_ in transposed_image:
                        self.check_image(image_, prompt, prompt_embeds)
                elif len(image) != len(self.controlnet.nets):
                    raise ValueError(
                        f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
                    )

                for image_ in image:
                    self.check_image(image_, prompt, prompt_embeds)
            else:
                assert False

            # Check `controlnet_conditioning_scale`
            if (
                isinstance(self.controlnet, ControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)
            ):
                if not isinstance(controlnet_conditioning_scale, float):
                    raise TypeError(
                        "For single controlnet: `controlnet_conditioning_scale` must be type `float`."
                    )
            elif (
                isinstance(self.controlnet, MultiControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
            ):
                if isinstance(controlnet_conditioning_scale, list):
                    if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                        raise ValueError(
                            "A single batch of varying conditioning scale settings (e.g. [[1.0, 0.5], [0.2, 0.8]]) is not supported at the moment. "
                            "The conditioning scale must be fixed across the batch."
                        )
                elif isinstance(controlnet_conditioning_scale, list) and len(
                    controlnet_conditioning_scale
                ) != len(self.controlnet.nets):
                    raise ValueError(
                        "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                        " the same length as the number of controlnets"
                    )
            else:
                assert False

            if not isinstance(control_guidance_start, (tuple, list)):
                control_guidance_start = [control_guidance_start]

            if not isinstance(control_guidance_end, (tuple, list)):
                control_guidance_end = [control_guidance_end]

            if len(control_guidance_start) != len(control_guidance_end):
                raise ValueError(
                    f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
                )

            if isinstance(self.controlnet, MultiControlNetModel):
                if len(control_guidance_start) != len(self.controlnet.nets):
                    raise ValueError(
                        f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                    )

            for start, end in zip(control_guidance_start, control_guidance_end):
                if start >= end:
                    raise ValueError(
                        f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                    )
                if start < 0.0:
                    raise ValueError(
                        f"control guidance start: {start} can't be smaller than 0."
                    )
                if end > 1.0:
                    raise ValueError(
                        f"control guidance end: {end} can't be larger than 1.0."
                    )

            if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
                raise ValueError(
                    "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
                )

            if ip_adapter_image_embeds is not None:
                if not isinstance(ip_adapter_image_embeds, list):
                    raise ValueError(
                        f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                    )
                elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                    raise ValueError(
                        f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                    )

        def check_image(self, image, prompt, prompt_embeds):
            image_is_pil = isinstance(image, PIL.Image.Image)
            image_is_tensor = isinstance(image, torch.Tensor)
            image_is_np = isinstance(image, np.ndarray)
            image_is_pil_list = isinstance(image, list) and isinstance(
                image[0], PIL.Image.Image
            )
            image_is_tensor_list = isinstance(image, list) and isinstance(
                image[0], torch.Tensor
            )
            image_is_np_list = isinstance(image, list) and isinstance(
                image[0], np.ndarray
            )

            if (
                not image_is_pil
                and not image_is_tensor
                and not image_is_np
                and not image_is_pil_list
                and not image_is_tensor_list
                and not image_is_np_list
            ):
                raise TypeError(
                    f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
                )

            if image_is_pil:
                image_batch_size = 1
            else:
                image_batch_size = len(image)

            if prompt is not None and isinstance(prompt, str):
                prompt_batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                prompt_batch_size = len(prompt)
            elif prompt_embeds is not None:
                prompt_batch_size = prompt_embeds.shape[0]

            if image_batch_size != 1 and image_batch_size != prompt_batch_size:
                raise ValueError(
                    f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
                )

        def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        ):
            image = self.control_image_processor.preprocess(
                image, height=height, width=width
            ).to(dtype=torch.float32)
            image_batch_size = image.shape[0]

            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # image batch size is the same as prompt batch size
                repeat_by = num_images_per_prompt

            image = image.repeat_interleave(repeat_by, dim=0)

            image = image.to(device=device, dtype=dtype)

            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)

            return image

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
        def prepare_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
        ):
            shape = (
                batch_size,
                num_channels_latents,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if latents is None:
                latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
            else:
                latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents

        # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
        def get_guidance_scale_embedding(
            self,
            w: torch.Tensor,
            embedding_dim: int = 512,
            dtype: torch.dtype = torch.float32,
        ) -> torch.FloatTensor:
            """
            See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

            Args:
                w (`torch.Tensor`):
                    Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
                embedding_dim (`int`, *optional*, defaults to 512):
                    Dimension of the embeddings to generate.
                dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                    Data type of the generated embeddings.

            Returns:
                `torch.FloatTensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
            """
            assert len(w.shape) == 1
            w = w * 1000.0

            half_dim = embedding_dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
            emb = w.to(dtype)[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            if embedding_dim % 2 == 1:  # zero pad
                emb = torch.nn.functional.pad(emb, (0, 1))
            assert emb.shape == (w.shape[0], embedding_dim)
            return emb

        @property
        def guidance_scale(self):
            return self._guidance_scale

        @property
        def clip_skip(self):
            return self._clip_skip

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        @property
        def do_classifier_free_guidance(self):
            return (
                self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
            )

        @property
        def cross_attention_kwargs(self):
            return self._cross_attention_kwargs

        @property
        def num_timesteps(self):
            return self._num_timesteps

        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            self,
            prompt: str | list[str] = None,
            image: PipelineImageInput = None,
            control_mask: PipelineImageInput | None = None,
            ip_adapter_index: int | list[int] | None = None,
            height: int | None = None,
            width: int | None = None,
            num_inference_steps: int = 50,
            timesteps: list[int] = None,
            guidance_scale: float = 7.5,
            negative_prompt: str | list[str] | None = None,
            num_images_per_prompt: int | None = 1,
            image_for_noise: PipelineImageInput | None = None,
            strength: float = 0.0,
            eta: float = 0.0,
            generator: torch.Generator | list[torch.Generator] | None = None,
            latents: torch.FloatTensor | None = None,
            prompt_embeds: torch.FloatTensor | None = None,
            negative_prompt_embeds: torch.FloatTensor | None = None,
            ip_adapter_image: PipelineImageInput | None = None,
            ip_adapter_image_embeds: list[torch.FloatTensor] | None = None,
            ip_adapter_mask: PipelineImageInput | None = None,
            ip_adapter_unconditional_noising_factor: list[float] = [0.0],
            output_type: str | None = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: dict[str, Any] | None = None,
            controlnet_conditioning_scale: float | list[float] = 1.0,
            guess_mode: bool = False,
            control_guidance_start: float | list[float] = 0.0,
            control_guidance_end: float | list[float] = 1.0,
            clip_skip: int | None = None,
            callback_on_step_end: Callable[[int, int, dict], None] | None = None,
            callback_on_step_end_tensor_inputs: list[str] = ["latents"],
            tile_window_height: int = 512,
            tile_window_width: int = 512,
            tile_stride_height: int = 256,
            tile_stride_width: int = 256,
            debug_latents: bool = False,
            debug_per_pass_latents: bool = False,
            **kwargs,
        ):
            r"""
            The call function to the pipeline for generation.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                        `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                    The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                    specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                    accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                    and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                    `init`, images must be passed as a list such that each element of the list can be correctly batched for
                    input to a single ControlNet. When `prompt` is a list, and if a list of images is passed for a single ControlNet,
                    each will be paired with each prompt in the `prompt` list. This also applies to multiple ControlNets,
                    where a list of image lists can be passed to batch for each prompt and each ControlNet.
                height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The height in pixels of the generated image.
                width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The width in pixels of the generated image.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                timesteps (`List[int]`, *optional*):
                    Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                    in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                    passed will be used. Must be in descending order.
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    A higher guidance scale value encourages the model to generate images closely linked to the text
                    `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                    pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                eta (`float`, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                    to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                    generation deterministic.
                latents (`torch.FloatTensor`, *optional*):
                    Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor is generated by sampling using the supplied random `generator`.
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                    provided, text embeddings are generated from the `prompt` input argument.
                negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                    not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
                ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                    Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters.
                    Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding
                    if `do_classifier_free_guidance` is set to `True`.
                    If not provided, embeddings are computed from the `ip_adapter_image` input argument.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
                callback (`Callable`, *optional*):
                    A function that calls every `callback_steps` steps during inference. The function is called with the
                    following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
                callback_steps (`int`, *optional*, defaults to 1):
                    The frequency at which the `callback` function is called. If not specified, the callback is called at
                    every step.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                    [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                    The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                    to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                    the corresponding scale as a list.
                guess_mode (`bool`, *optional*, defaults to `False`):
                    The ControlNet encoder tries to recognize the content of the input image even if you remove all
                    prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
                control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                    The percentage of total steps at which the ControlNet starts applying.
                control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                    The percentage of total steps at which the ControlNet stops applying.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
                callback_on_step_end (`Callable`, *optional*):
                    A function that calls at the end of each denoising steps during the inference. The function is called
                    with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                    callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                    `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeline class.

            Examples:

            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                    If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                    otherwise a `tuple` is returned where the first element is a list with the generated images and the
                    second element is a list of `bool`s indicating whether the corresponding generated image contains
                    "not-safe-for-work" (nsfw) content.
            """

            if image is None:
                image = []

            ip_adapter_masks = None

            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            # check that either latents or image_for_noise is passed
            if latents is not None and image_for_noise is not None:
                raise ValueError(
                    "Either `latents` or `image_for_noise` should be passed, but not both."
                )

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )

            controlnet = (
                self.controlnet._orig_mod
                if is_compiled_module(self.controlnet)
                else self.controlnet
            )

            # align format for control guidance
            if not isinstance(control_guidance_start, list) and isinstance(
                control_guidance_end, list
            ):
                control_guidance_start = len(control_guidance_end) * [
                    control_guidance_start
                ]
            elif not isinstance(control_guidance_end, list) and isinstance(
                control_guidance_start, list
            ):
                control_guidance_end = len(control_guidance_start) * [
                    control_guidance_end
                ]
            elif not isinstance(control_guidance_start, list) and not isinstance(
                control_guidance_end, list
            ):
                mult = (
                    len(controlnet.nets)
                    if isinstance(controlnet, MultiControlNetModel)
                    else 1
                )
                control_guidance_start, control_guidance_end = (
                    mult * [control_guidance_start],
                    mult * [control_guidance_end],
                )

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                image,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                controlnet_conditioning_scale,
                control_guidance_start,
                control_guidance_end,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            if isinstance(controlnet, MultiControlNetModel) and isinstance(
                controlnet_conditioning_scale, float
            ):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                    controlnet.nets
                )

            if isinstance(controlnet, ControlNetModel):
                global_pool_conditions = controlnet.config.global_pool_conditions
            elif isinstance(controlnet, MultiControlNetModel):
                if len(controlnet.nets) != 0:
                    global_pool_conditions = controlnet.nets[
                        0
                    ].config.global_pool_conditions
                else:
                    global_pool_conditions = guess_mode

            guess_mode = guess_mode or global_pool_conditions

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                self.cross_attention_kwargs.get("scale", None)
                if self.cross_attention_kwargs is not None
                else None
            )
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds, ip_adapter_masks = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    ip_adapter_mask,
                    width,
                    height,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                    ip_adapter_unconditional_noising_factor,
                )

            if isinstance(ip_adapter_index, int):
                ip_adapter_index = [ip_adapter_index]

            # 4. Prepare image
            if isinstance(controlnet, ControlNetModel):
                image = self.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
            elif isinstance(controlnet, MultiControlNetModel):
                images = []

                # Nested lists as ControlNet condition
                if len(image) > 0 and isinstance(image[0], list):
                    # Transpose the nested image list
                    image = [list(t) for t in zip(*image)]

                for image_ in image:
                    image_ = self.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                if len(image) > 0:
                    height, width = image[0].shape[-2:]
            else:
                assert False

            if image_for_noise is not None:
                image_for_noise = prepare_noise_image(
                    image_for_noise,
                )

                # if the height and width is set
                # resize the image_for_noise to the same size
                if height is not None and width is not None:
                    image_for_noise = F.interpolate(
                        image_for_noise, size=(height, width), mode="bilinear"
                    )
                else:
                    height, width = image_for_noise.shape[-2:]

            # 4.2 Prepare control mask
            controlnet_masks = []
            if control_mask is not None:
                for mask in control_mask:
                    mask = np.array(mask)
                    mask_tensor = torch.from_numpy(mask).to(
                        device=device, dtype=prompt_embeds.dtype
                    )
                    mask_tensor = mask_tensor[:, :, 0] / 255.0
                    mask_tensor = mask_tensor[None, None]
                    h, w = mask_tensor.shape[-2:]
                    control_mask_list = []
                    for scale in [8, 8, 8, 16, 16, 16, 32, 32, 32]:
                        # Python uses IEEE 754 rounding rules, we need to add a small value to round like the unet model
                        w_n = round((w + 0.01) / 8)
                        h_n = round((h + 0.01) / 8)
                        if scale in [16, 32]:
                            w_n = round((w_n + 0.01) / 2)
                            h_n = round((h_n + 0.01) / 2)
                        if scale == 32:
                            w_n = round((w_n + 0.01) / 2)
                            h_n = round((h_n + 0.01) / 2)
                        scale_mask_weight_image_tensor = F.interpolate(
                            mask_tensor, (h_n, w_n), mode="bilinear"
                        )
                        control_mask_list.append(scale_mask_weight_image_tensor)
                    controlnet_masks.append(control_mask_list)

            # 5. Prepare timesteps

            if image_for_noise is not None:
                # reworks for automatic1111/comfyui style timesteps, e.g. always the same
                # regardless of strength
                new_num_inference_steps = int(num_inference_steps * (1 / strength))
                self.scheduler.set_timesteps(new_num_inference_steps, device=device)
                timesteps, num_inference_steps = self.get_timesteps(
                    new_num_inference_steps, strength, device
                )
                latent_timestep = timesteps[:1].repeat(
                    batch_size * num_images_per_prompt
                )

            else:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps, num_inference_steps = retrieve_timesteps(
                    self.scheduler, num_inference_steps, device, timesteps
                )
                self._num_timesteps = len(timesteps)

            # 6. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            if image_for_noise is not None:
                latents = self.prepare_image_latents(
                    image_for_noise,
                    latent_timestep,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                )
            else:
                latents = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
                )

            view_batch_size = 1
            circular_padding = False
            latent_height = latents.shape[2]
            latent_width = latents.shape[3]

            window_height = tile_window_height // 8
            window_width = tile_window_width // 8
            stride_height = tile_stride_height // 8
            stride_width = tile_stride_width // 8

            window_height = min(window_height, latent_height)
            window_width = min(window_width, latent_width)

            views = get_views(
                latent_height,
                latent_width,
                window_height=window_height,
                window_width=window_width,
                stride_height=stride_height,
                stride_width=stride_width,
                circular_padding=circular_padding,
            )

            views_batch = [
                views[i : i + view_batch_size]
                for i in range(0, len(views), view_batch_size)
            ]
            views_scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(
                views_batch
            )
            count = torch.zeros_like(latents)
            value = torch.zeros_like(latents)

            # 6.5 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                    batch_size * num_images_per_prompt
                )
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=self.unet.config.time_cond_proj_dim,
                ).to(device=device, dtype=latents.dtype)

            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7.1 Add image embeds for IP-Adapter
            added_cond_kwargs = (
                {"image_embeds": image_embeds}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None
                else None
            )

            if ip_adapter_masks is not None:
                self._cross_attention_kwargs["ip_adapter_masks"] = ip_adapter_masks

            # 7.2 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(
                    keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
                )

            debug_latents_list = []
            debug_per_pass_latents_list = []

            # 8. Denoising loop
            num_warmup_steps = (
                len(timesteps) - num_inference_steps * self.scheduler.order
            )
            is_unet_compiled = is_compiled_module(self.unet)
            is_controlnet_compiled = is_compiled_module(self.controlnet)
            is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

            if len(views_batch) == 1:
                latents = latents.to(device=device)
                value = value.to(device=device)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # Relevant thread:
                    # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                    if (
                        is_unet_compiled and is_controlnet_compiled
                    ) and is_torch_higher_equal_2_1:
                        torch._inductor.cudagraph_mark_step_begin()

                    count.zero_()
                    value.zero_()

                    per_pass_latents = []

                    for j, batch_view in enumerate(views_batch):
                        vb_size = len(batch_view)
                        latents_for_view = torch.cat(
                            [
                                latents[:, :, h_start:h_end, w_start:w_end]
                                for h_start, h_end, w_start, w_end in batch_view
                            ]
                        )

                        if image:
                            # take a view of the image
                            if isinstance(image, list):
                                control_image_for_view = []
                                for image_ in image:
                                    control_image_for_view.append(
                                        torch.cat(
                                            [
                                                image_[
                                                    :,
                                                    :,
                                                    h_start * 8 : h_end * 8,
                                                    w_start * 8 : w_end * 8,
                                                ]
                                                for h_start, h_end, w_start, w_end in batch_view
                                            ]
                                        )
                                    )
                            else:
                                control_image_for_view = torch.cat(
                                    [
                                        image[
                                            :,
                                            :,
                                            h_start * 8 : h_end * 8,
                                            w_start * 8 : w_end * 8,
                                        ]
                                        for h_start, h_end, w_start, w_end in batch_view
                                    ]
                                )

                            latents_for_view = latents_for_view.to(
                                device=device, dtype=latents.dtype
                            )
                            if isinstance(control_image_for_view, list):
                                control_image_for_view = [
                                    control_image.to(
                                        device=device, dtype=self.controlnet.dtype
                                    )
                                    for control_image in control_image_for_view
                                ]
                            else:
                                control_image_for_view = control_image_for_view.to(
                                    device=device, dtype=controlnet.dtype
                                )

                        # rematch block's scheduler status
                        self.scheduler.__dict__.update(views_scheduler_status[j])

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = (
                            torch.cat([latents_for_view] * 2)
                            if self.do_classifier_free_guidance
                            else latents_for_view
                        )
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        # repeat prompt_embeds for batch
                        # not sure why this isn't used yet
                        prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)

                        # controlnet(s) inference
                        if guess_mode and self.do_classifier_free_guidance:
                            # Infer ControlNet only for the conditional batch.
                            control_model_input = latents
                            control_model_input = self.scheduler.scale_model_input(
                                control_model_input, t
                            )
                            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                        else:
                            control_model_input = latent_model_input
                            controlnet_prompt_embeds = prompt_embeds

                        if (
                            ip_adapter_image_embeds is None
                            and ip_adapter_image is not None
                        ):
                            encoder_hidden_states = (
                                self.unet.process_encoder_hidden_states(
                                    prompt_embeds, {"image_embeds": image_embeds}
                                )
                            )
                            ip_adapter_image_embeds = encoder_hidden_states[1]

                        if isinstance(controlnet_keep[i], list):
                            cond_scale = [
                                c * s
                                for c, s in zip(
                                    controlnet_conditioning_scale, controlnet_keep[i]
                                )
                            ]
                        else:
                            controlnet_cond_scale = controlnet_conditioning_scale
                            if isinstance(controlnet_cond_scale, list):
                                controlnet_cond_scale = controlnet_cond_scale[0]
                            cond_scale = controlnet_cond_scale * controlnet_keep[i]

                        if (
                            isinstance(controlnet, MultiControlNetModel)
                            and len(controlnet.nets) == 0
                        ):
                            down_block_res_samples, mid_block_res_sample = None, None
                        else:
                            for controlnet_index in range(len(self.controlnet.nets)):
                                ipa_index = ip_adapter_index[controlnet_index]
                                if ipa_index is not None:
                                    control_prompt_embeds = ip_adapter_image_embeds[
                                        ipa_index
                                    ].squeeze(1)
                                else:
                                    control_prompt_embeds = controlnet_prompt_embeds

                                down_samples, mid_sample = self.controlnet.nets[
                                    controlnet_index
                                ](
                                    control_model_input,
                                    t,
                                    encoder_hidden_states=control_prompt_embeds,
                                    controlnet_cond=control_image_for_view[
                                        controlnet_index
                                    ],
                                    conditioning_scale=cond_scale[controlnet_index],
                                    guess_mode=guess_mode,
                                    added_cond_kwargs=controlnet_added_cond_kwargs,
                                    return_dict=False,
                                )

                                if (
                                    len(controlnet_masks) > controlnet_index
                                    and controlnet_masks[controlnet_index] is not None
                                ):
                                    down_samples = [
                                        down_sample * mask_weight
                                        for down_sample, mask_weight in zip(
                                            down_samples,
                                            controlnet_masks[controlnet_index],
                                        )
                                    ]
                                    mid_sample *= controlnet_masks[controlnet_index][-1]

                                if (
                                    down_block_res_samples is None
                                    and mid_block_res_sample is None
                                ):
                                    down_block_res_samples = down_samples
                                    mid_block_res_sample = mid_sample
                                else:
                                    down_block_res_samples = [
                                        samples_prev + samples_curr
                                        for samples_prev, samples_curr in zip(
                                            down_block_res_samples, down_samples
                                        )
                                    ]
                                    mid_block_res_sample += mid_sample

                        if guess_mode and self.do_classifier_free_guidance:
                            # Inferred ControlNet only for the conditional batch.
                            # To apply the output of ControlNet to both the unconditional and conditional batches,
                            # add 0 to the unconditional batch to keep it unchanged.
                            down_block_res_samples = [
                                torch.cat([torch.zeros_like(d), d])
                                for d in down_block_res_samples
                            ]
                            mid_block_res_sample = torch.cat(
                                [
                                    torch.zeros_like(mid_block_res_sample),
                                    mid_block_res_sample,
                                ]
                            )

                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        # compute the previous noisy sample x_t -> x_t-1
                        latents_denoised_batch = self.scheduler.step(
                            noise_pred, t, latents_for_view, **extra_step_kwargs
                        ).prev_sample

                        # save views scheduler status after sample
                        views_scheduler_status[j] = copy.deepcopy(
                            self.scheduler.__dict__
                        )

                        # extract value from batch
                        for latents_view_denoised, (
                            h_start,
                            h_end,
                            w_start,
                            w_end,
                        ) in zip(latents_denoised_batch.chunk(vb_size), batch_view):
                            latents_view_denoised = latents_view_denoised.to(
                                device=value.device
                            )
                            weights = torch.ones_like(latents_view_denoised)
                            if circular_padding and w_end > latents.shape[3]:
                                # Case for circular padding
                                value[
                                    :, :, h_start:h_end, w_start:
                                ] += latents_view_denoised[
                                    :, :, h_start:h_end, : latents.shape[3] - w_start
                                ].cpu()
                                value[
                                    :, :, h_start:h_end, : w_end - latents.shape[3]
                                ] += latents_view_denoised[
                                    :, :, h_start:h_end, latents.shape[3] - w_start :
                                ].cpu()
                                count[:, :, h_start:h_end, w_start:] += weights
                                count[
                                    :, :, h_start:h_end, : w_end - latents.shape[3]
                                ] += weights
                            else:

                                value[
                                    :, :, h_start:h_end, w_start:w_end
                                ] += latents_view_denoised
                                count[:, :, h_start:h_end, w_start:w_end] += weights

                            if debug_per_pass_latents:
                                per_pass_latents.append(
                                    (
                                        (h_start, h_end, w_start, w_end),
                                        latents_view_denoised.cpu(),
                                        weights.cpu(),
                                    )
                                )
                    if debug_per_pass_latents:
                        debug_per_pass_latents_list.append(per_pass_latents)

                    # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
                    latents = torch.where(count > 0, value / count, value)

                    if debug_latents:
                        debug_latents_list.append(latents.cpu())

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop(
                            "prompt_embeds", prompt_embeds
                        )
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            # If we do sequential model offloading, let's offload unet and controlnet
            # manually for max memory savings
            if (
                hasattr(self, "final_offload_hook")
                and self.final_offload_hook is not None
            ):
                self.unet.to("cpu")
                self.controlnet.to("cpu")
                torch.cuda.empty_cache()

            if debug_per_pass_latents:
                temp_debug_per_pass_latents_list = []
                for per_pass_latents in debug_per_pass_latents_list:
                    temp_per_pass_latents = []
                    for per_pass_latent in per_pass_latents:
                        dims = per_pass_latent[0]
                        the_per_pass_latent = (
                            per_pass_latent[1] / self.vae.config.scaling_factor
                        )
                        the_per_pass_latent = self.vae.decode(
                            the_per_pass_latent.to("cuda"), return_dict=False
                        )[0]
                        the_per_pass_latent = self.image_processor.postprocess(
                            the_per_pass_latent,
                            output_type="pil",
                        )

                        # need to convert the greyscale weight images to images
                        weight_images = self.image_processor.postprocess(
                            per_pass_latent[2],
                            output_type="pil",
                        )

                        temp_per_pass_latents.append(
                            (dims, the_per_pass_latent, weight_images)
                        )

                    temp_debug_per_pass_latents_list.append(temp_per_pass_latents)
                debug_per_pass_latents_list = temp_debug_per_pass_latents_list

            if debug_latents:
                temp_debug_latents_list = []
                for the_latents in debug_latents_list:
                    the_latents = self.vae.decode(
                        the_latents.to("cuda"), return_dict=False
                    )[0]
                    the_latents = self.image_processor.postprocess(
                        the_latents,
                        output_type="pil",
                    )
                    temp_debug_latents_list.append(the_latents)

                debug_latents_list = temp_debug_latents_list

            if not output_type == "latent":
                image = self.vae.decode(
                    latents / self.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                image, has_nsfw_concept = self.run_safety_checker(
                    image, device, prompt_embeds.dtype
                )
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=do_denormalize
            )

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (
                    image,
                    has_nsfw_concept,
                    debug_latents_list,
                    debug_per_pass_latents_list,
                )

            result = StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            )

            result.debug_latents = debug_latents_list
            result.debug_per_pass_latents = debug_per_pass_latents_list

            return result

    return StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline

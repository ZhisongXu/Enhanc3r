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

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import export_to_video
import os
import time

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import StableVideoDiffusionPipeline
        >>> from diffusers.utils import load_image, export_to_video

        >>> pipe = StableVideoDiffusionPipeline.from_pretrained(
        ...     "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd-docstring-example.jpeg"
        ... )
        >>> image = image.resize((1024, 576))

        >>> frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        >>> export_to_video(frames, "generated.mp4", fps=7)
        ```
"""


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor: VaeImageProcessor, output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.FloatTensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.FloatTensor]


class StableVideoDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder
            ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.FloatTensor:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)
            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            processor = GaussianVideoProcessor(device="cuda")
            image = processor.process_frame(image)
            image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    @ torch.no_grad()
    def encode_video_latents(self, video, encode_chunk_size: int = 14):
        video = video.to(device=self._execution_device)
        if video.dtype != self.vae.dtype:
            video = video.to(dtype=self.vae.dtype)

        for i in range(0, video.shape[0], encode_chunk_size):
            video_chunk = video[i : i + encode_chunk_size]
            video_chunk = self.vae.encode(video_chunk).latent_dist.mode()
            if i == 0:
                video_latents = video_chunk
            else:
                video_latents = torch.cat([video_latents, video_chunk], dim=0)
        video_latents = video_latents * self.vae.config.scaling_factor
        return video_latents

    def decode_latents(self, latents: torch.FloatTensor, num_frames: int, decode_chunk_size: int = 14, rescale: bool = False):

        if rescale:
            mean = torch.mean(latents[0,0], dim=(1,2), keepdim=True)
            std = torch.std(latents[0,0], dim=(1,2), keepdim=True)

            for f_id in range(num_frames):
                latents[0,f_id] = (latents[0,f_id] - torch.mean(latents[0,f_id], dim=(1,2), keepdim=True)) / \
                                                    torch.std(latents[0,f_id], dim=(1,2), keepdim=True)
                latents[0,f_id] = latents[0,f_id] * std + mean

        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
            and not isinstance(image, np.ndarray)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` or `np.ndarray` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.FloatTensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video: np.ndarray,
        mask: Optional[np.ndarray] = None,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        visualize_each_step_path: Optional[str] = None,
        resample_steps: Optional[int] = 1,
        guide_util: Optional[int] = 10,
    ):
        """
        处理输入视频并使用稳定视频扩散生成输出帧的函数。

        Args:
            video (`np.ndarray`): 输入的视频。
            mask (`np.ndarray`, *可选*, 默认为 None): 可选的掩码。
            height (`int`, *可选*, 默认为 576): 输出帧的高度。
            width (`int`, *可选*, 默认为 1024): 输出帧的宽度。
            num_frames (`int`, *可选*): 生成的帧数。
            num_inference_steps (`int`, *可选*, 默认为 25): 推理步骤数。
            min_guidance_scale (`float`, *可选*, 默认为 1.0): 最小引导尺度。
            max_guidance_scale (`float`, *可选*, 默认为 3.0): 最大引导尺度。
            fps (`int`, *可选*, 默认为 7): 每秒帧数。
            motion_bucket_id (`int`, *可选*, 默认为 127): 运动桶 ID。
            noise_aug_strength (`float`, *可选*, 默认为 0): 噪声增强强度。
            decode_chunk_size (`int`, *可选*): 解码的块大小。
            num_videos_per_prompt (`int`, *可选*, 默认为 1): 每个提示生成的视频数。
            generator (Optional[Union[torch.Generator, List[torch.Generator]]], *可选*): 随机数生成器。
            latents (Optional[torch.FloatTensor], *可选*): 预生成的潜在变量。
            output_type (`str`, *可选*, 默认为 "pil"): 输出格式类型。
            return_dict (`bool`, *可选*, 默认为 True): 是否返回字典。

        Returns:
            `StableVideoDiffusionPipelineOutput` 或 `tuple`: 如果 return_dict 为 True，则返回 `StableVideoDiffusionPipelineOutput`，
            否则返回包含帧的元组。

        Examples:
        """
        # Input validation
        assert resample_steps >= 1, "resample_steps must be at least 1"

        # 1. Set default height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        assert video.shape[1] == num_frames, (
            f"The number of frames in the video must be equal to num_frames, expected {num_frames} but got {video.shape[1]}"
        )

        # 2. Check inputs
        self.check_inputs(video, height, width)

        # 3. Define batch size and device
        batch_size = video.shape[0] if isinstance(video, np.ndarray) else 1
        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        # 4. Encode input image (first frame)
        image = video[0, 0][None, ...]
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # Adjust FPS to meet training conditions
        fps = fps - 1

        # 5. Use VAE to encode the video and image
        video_tensor = self.image_processor.preprocess(video[0], height=height, width=width).to(device)
        image_tensor = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image_tensor.shape, generator=generator, device=device, dtype=image_tensor.dtype)
        image_tensor = image_tensor + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image_tensor,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        ).to(image_embeddings.dtype)
        video_latents = self.encode_video_latents(video_tensor).to(image_embeddings.dtype)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 6. Get additional time IDs
        added_time_ids = self._get_add_time_ids(
            fps, motion_bucket_id, noise_aug_strength, image_embeddings.dtype, batch_size,
            num_videos_per_prompt, self.do_classifier_free_guidance
        ).to(device)

        # 7. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 8. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, num_frames, num_channels_latents, height, width,
            image_embeddings.dtype, device, generator, latents
        )

        # 9. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype).repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        # === Innovative Enhancement Features ===

        # Enhancement 1: Dynamic noise rescheduling
        def dynamic_noise_reschedule(timesteps, base_strength=0.7):
            weights = torch.linspace(base_strength, 1.0 - base_strength, len(timesteps))
            return weights.to(timesteps.device)
        noise_reschedule_weights = dynamic_noise_reschedule(timesteps)

        # Enhancement 2: Motion-aware latent initialization
        def compute_smoothed_motion(motion_features, window_size=10):
            B, F, C, H, W = motion_features.shape
            motion_reshaped = motion_features.permute(0, 2, 3, 4, 1).reshape(B, C * H * W, F)
            padding = (window_size - 1) // 2
            motion_padded = F.pad(motion_reshaped, (padding, padding), mode='replicate')
            smoothed = F.avg_pool1d(motion_padded, kernel_size=window_size, stride=1)
            smoothed = smoothed.reshape(B, C, H, W, F).permute(0, 4, 1, 2, 3)
            return smoothed

        def motion_aware_latent_init(video_latents, latents, num_frames, window_size=10):
            motion_features = video_latents[:, 1:] - video_latents[:, :-1]
            zero_pad = torch.zeros_like(video_latents[:, :1])
            motion_features = torch.cat([zero_pad, motion_features], dim=1)
            smoothed_motion_features = compute_smoothed_motion(motion_features, window_size=window_size)
            new_latents = latents + 0.1 * smoothed_motion_features
            return new_latents, smoothed_motion_features

        # 在主代码中
        if video_latents is not None:
            latents, smoothed_motion_features = motion_aware_latent_init(video_latents, latents, num_frames, window_size=10)
        else:
            smoothed_motion_features = None

        # Enhancement 3: Hierarchical guidance strategy
        class HierarchicalGuidance:
            def __init__(self, total_steps, guide_util):
                self.stage_thresholds = [
                    (0, 0.2, "coarse"),   # First 20%: Coarse guidance
                    (0.2, 0.8, "detail"), # Middle 60%: Detail optimization
                    (0.8, 1.0, "refine")  # Last 20%: Fine-tuning
                ]

            def adjust_guidance(self, progress):
                for min_p, max_p, stage in self.stage_thresholds:
                    if min_p <= progress < max_p:
                        if stage == "coarse":
                            return {"scale_factor": 1.2, "temporal_weight": 0.8}
                        elif stage == "detail":
                            return {"scale_factor": 0.9, "temporal_weight": 1.2}
                        else:
                            return {"scale_factor": 0.5, "temporal_weight": 1.5}
                return {"scale_factor": 1.0, "temporal_weight": 1.0}

        h_guidance = HierarchicalGuidance(num_inference_steps, guide_util)

        # 10. Denoising loop (with enhancement features)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                current_progress = i / len(timesteps)
                guidance_params = h_guidance.adjust_guidance(current_progress)
                adjusted_guidance_scale = guidance_scale * guidance_params["scale_factor"]

                for r in range(resample_steps):
                    # Classifier-free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Enhancement 4: Spatiotemporal consistency
                    if r > 0 and i < guide_util:
                        shifted_latents = torch.roll(latents, shifts=1, dims=1)
                        latents = 0.7 * latents + 0.3 * shifted_latents

                    # Enhancement 5: Concatenate motion features
                    if smoothed_motion_features is not None:
                        motion_to_add = smoothed_motion_features.repeat(
                            2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1
                        )
                        latent_model_input = latent_model_input + 0.1 * motion_to_add
                    else:
                        latent_model_input = latent_model_input + 0.1 * image_latents

                    # Use dynamic noise rescheduling to predict noise
                    noise_pred = self.unet(
                        latent_model_input,
                        t * noise_reschedule_weights[i],
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]

                    # Apply guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + adjusted_guidance_scale * (
                            guidance_params["temporal_weight"] * (noise_pred_cond - noise_pred_uncond)
                        )

                    # Adaptive resampling
                    if r < resample_steps - 1:
                        scheduler_output = self.scheduler.step(
                            noise_pred, t, latents,
                            mask=torch.from_numpy(mask).to(device) if mask is not None else None,
                            guided=i < guide_util and r < (resample_steps / 2 - 1),
                            video_latents=video_latents,
                        )
                        pred_original_sample = scheduler_output.pred_original_sample.half()
                        adaptive_noise = torch.randn_like(pred_original_sample) * (1 - current_progress)
                        self.scheduler._step_index -= 1  # Reset step index for resampling
                        latents = self.scheduler.add_noise(pred_original_sample, adaptive_noise, t)
                    else:
                        scheduler_output = self.scheduler.step(
                            noise_pred, t, latents,
                            mask=torch.from_numpy(mask).to(device) if mask is not None else None,
                            guided=i < guide_util,
                            video_latents=video_latents,
                        )
                        pred_original_sample = scheduler_output.pred_original_sample.half()
                        latents = scheduler_output.prev_sample

                # Enhancement 6: Latent space sharpening
                if current_progress > 0.8:
                    sharpening_kernel = torch.tensor(
                        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
                        dtype=latents.dtype, device=device
                    ).view(1, 1, 3, 3).expand(latents.shape[2], 1, 3, 3)  # [C, 1, 3, 3]
                    for b in range(latents.shape[0]):
                        for f in range(latents.shape[1]):
                            latents[b, f] = F.conv2d(latents[b, f].unsqueeze(0), sharpening_kernel, groups=latents.shape[2], padding=1).squeeze(0)

                # Visualization
                if visualize_each_step_path:
                    os.makedirs(visualize_each_step_path, exist_ok=True)
                    frames = self.decode_latents(pred_original_sample, num_frames, decode_chunk_size)
                    frames = tensor2vid(frames, self.image_processor, output_type=output_type)
                    export_to_video(frames[0], os.path.join(visualize_each_step_path, f"generated_step{i:02d}.mp4"), fps=7)

                # Callback function
                if callback_on_step_end:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i >= num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        # 11. Decode and return the result
        if output_type != "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        return StableVideoDiffusionPipelineOutput(frames=frames) if return_dict else frames


# resizing utils
# TODO: clean up later
class GaussianVideoProcessor:
    def __init__(self, device='cuda'):
        """Initialize the GaussianVideoProcessor with a specified device."""
        self.device = torch.device(device)
        self.prev_frame = None  # Used for temporal smoothing

    @staticmethod
    def detect_black_regions(frame_tensor, threshold=0.01):
        """Detect near-black regions and return a mask (1 for areas to repair, 0 to keep)."""
        mask = frame_tensor.mean(dim=0, keepdim=True) < threshold  # [1, H, W]
        return mask.expand_as(frame_tensor)  # Match the shape of frame_tensor

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        """Generate a 1D Gaussian kernel."""
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, kernel_size)  # Shape adapted for conv2d

    def _gaussian_blur2d(self, input_tensor, kernel_size, sigma):
        """Apply separable 2D Gaussian blur."""
        device = input_tensor.device
        kernel_size = tuple(k | 1 for k in kernel_size)  # Ensure kernel_size is odd

        # Horizontal convolution
        kernel_h = self.create_gaussian_kernel(kernel_size[0], sigma[0]).to(device)
        kernel_h = kernel_h.expand(input_tensor.size(1), -1, -1, -1)
        input_tensor = F.conv2d(input_tensor, kernel_h, padding=(kernel_size[0] // 2, 0), groups=input_tensor.size(1))

        # Vertical convolution
        kernel_v = self.create_gaussian_kernel(kernel_size[1], sigma[1]).to(device)
        kernel_v = kernel_v.expand(input_tensor.size(1), -1, -1, -1)
        return F.conv2d(input_tensor, kernel_v, padding=(0, kernel_size[1] // 2), groups=input_tensor.size(1))

    def temporal_smoothing(self, input_tensor, alpha=0.9):
        """Perform smoothing in the temporal dimension."""
        if self.prev_frame is None:
            self.prev_frame = input_tensor
        else:
            self.prev_frame = alpha * self.prev_frame + (1 - alpha) * input_tensor
        return self.prev_frame

    def process_frame(self, input_tensor, alpha=0.9):
        """Process a single frame: repair black regions + Gaussian blur + temporal smoothing."""
        # Calculate Gaussian blur parameters
        h, w = input_tensor.shape[-2:]
        sigma_h = max(h / 1000, 0.5)  # Empirical value, larger sigma means stronger blur
        sigma_w = max(w / 1000, 0.5)
        ks_h = max(int(6 * sigma_h + 0.5) | 1, 3)
        ks_w = max(int(6 * sigma_w + 0.5) | 1, 3)

        # Detect black regions
        mask = self.detect_black_regions(input_tensor)

        # Apply blur for inpainting
        blurred = self._gaussian_blur2d(input_tensor, (ks_h, ks_w), (sigma_h, sigma_w))
        inpainted = torch.where(mask, blurred, input_tensor)

        # Temporal smoothing
        smoothed = self.temporal_smoothing(inpainted, alpha)

        return smoothed

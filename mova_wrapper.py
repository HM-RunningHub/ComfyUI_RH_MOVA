"""
MOVA Inference Wrapper for ComfyUI

This module provides a non-distributed inference wrapper for MOVA,
allowing it to run in ComfyUI without requiring torchrun or distributed setup.
"""

import gc
import os
import sys
from typing import Optional, Tuple

import torch
from PIL import Image


_distributed_patched = False


def patch_torch_distributed():
    """
    Patch torch.distributed module to work without actual distributed setup.
    This must be called before importing any MOVA modules.
    """
    global _distributed_patched
    if _distributed_patched:
        return
    
    import torch.distributed as dist
    
    # Store original functions
    _original_is_initialized = dist.is_initialized
    _original_get_rank = getattr(dist, 'get_rank', lambda: 0)
    _original_get_world_size = getattr(dist, 'get_world_size', lambda: 1)
    
    def fake_is_initialized():
        try:
            return _original_is_initialized()
        except RuntimeError:
            return False
    
    def fake_get_rank():
        try:
            if fake_is_initialized():
                return _original_get_rank()
        except RuntimeError:
            pass
        return 0
    
    def fake_get_world_size():
        try:
            if fake_is_initialized():
                return _original_get_world_size()
        except RuntimeError:
            pass
        return 1
    
    def fake_barrier():
        try:
            if fake_is_initialized():
                dist.barrier()
        except RuntimeError:
            pass
    
    # Apply patches
    dist.is_initialized = fake_is_initialized
    dist.get_rank = fake_get_rank
    dist.get_world_size = fake_get_world_size
    dist.barrier = fake_barrier
    
    _distributed_patched = True
    print("[MOVA] torch.distributed patched for single-GPU inference")


def load_mova_pipeline(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    offload_mode: str = "group",
    device_id: int = 0,
):
    """
    Load MOVA pipeline for single-GPU inference (fully offline).
    
    Args:
        model_path: Path to the MOVA model directory (e.g., ComfyUI/models/MOVA/MOVA-360p)
        torch_dtype: Model precision (bfloat16, float16, float32)
        offload_mode: Memory offload strategy ("none", "cpu", "group")
        device_id: CUDA device ID
        
    Returns:
        MOVA pipeline ready for inference
        
    Note:
        This function uses local_files_only=True to ensure fully offline operation.
        Models must be pre-downloaded to the specified path.
    """
    from mova.diffusion.pipelines.pipeline_mova import MOVA
    
    print(f"[MOVA] Loading model from {model_path}...")
    # Use local_files_only=True to prevent any download attempts (fully offline)
    pipe = MOVA.from_pretrained(model_path, torch_dtype=torch_dtype, local_files_only=True)
    
    device = torch.device("cuda", device_id)
    
    if offload_mode == "none":
        pipe.to(device)
        print(f"[MOVA] Model loaded to GPU (no offload)")
    elif offload_mode == "cpu":
        pipe.enable_model_cpu_offload(device_id)
        print(f"[MOVA] Model loaded with CPU offload (~48GB VRAM)")
    elif offload_mode == "group":
        pipe.enable_group_offload(
            onload_device=device,
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
        )
        print(f"[MOVA] Model loaded with group offload (~12GB VRAM)")
    
    return pipe


def run_mova_inference(
    pipe,
    prompt: str,
    reference_image: Image.Image,
    negative_prompt: str = "",
    height: int = 352,
    width: int = 640,
    num_frames: int = 97,
    fps: float = 24.0,
    num_inference_steps: int = 25,
    cfg_scale: float = 5.0,
    sigma_shift: float = 5.0,
    seed: int = 42,
    remove_video_dit: bool = False,
    progress_callback=None,
) -> Tuple[list, torch.Tensor]:
    """
    Run MOVA inference to generate synchronized video and audio.
    
    Args:
        pipe: Loaded MOVA pipeline
        prompt: Text prompt describing the scene and speech
        reference_image: Reference image (PIL Image, already cropped/resized)
        negative_prompt: Negative prompt
        height: Video height (must be divisible by 16)
        width: Video width (must be divisible by 16)
        num_frames: Number of frames (num_frames - 1 must be divisible by 4)
        fps: Video frame rate
        num_inference_steps: Number of denoising steps
        cfg_scale: Classifier-free guidance scale
        sigma_shift: Sigma shift parameter
        seed: Random seed
        remove_video_dit: Whether to free video_dit after stage switch
        progress_callback: Optional callback function(step, total_steps) for progress updates
        
    Returns:
        Tuple of (video_frames, audio_tensor)
        - video_frames: List of PIL Images
        - audio_tensor: Audio waveform tensor [1, T]
    """
    # Set seed
    torch.manual_seed(seed)
    
    print(f"[MOVA] Starting inference: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")
    
    # Monkey-patch tqdm to capture progress
    from tqdm import tqdm as original_tqdm
    
    class ProgressTqdm:
        """Wrapper for tqdm that calls progress_callback."""
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', len(iterable) if iterable is not None else None)
            self.n = 0
            self.disable = kwargs.get('disable', False)
            
        def __iter__(self):
            if self.iterable is None:
                return self
            for item in self.iterable:
                yield item
                self.n += 1
                if progress_callback is not None and not self.disable:
                    progress_callback(self.n, self.total or self.n)
                    
        def update(self, n=1):
            self.n += n
            if progress_callback is not None and not self.disable:
                progress_callback(self.n, self.total or self.n)
                
        def close(self):
            pass
            
        def set_description(self, desc=None, refresh=True):
            pass
    
    # Temporarily replace tqdm in the pipeline module
    import mova.diffusion.pipelines.pipeline_mova as pipeline_module
    old_tqdm = getattr(pipeline_module, 'tqdm', original_tqdm)
    pipeline_module.tqdm = ProgressTqdm if progress_callback else original_tqdm
    
    try:
        # Run pipeline
        video, audio = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            image=reference_image,
            height=height,
            width=width,
            video_fps=fps,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            cfg_scale=cfg_scale,
            seed=seed,
            cp_mesh=None,  # No context parallel for single GPU
            remove_video_dit=remove_video_dit,
        )
    finally:
        # Restore original tqdm
        pipeline_module.tqdm = old_tqdm
    
    print(f"[MOVA] Inference complete!")
    
    # video[0] contains the video frames
    # audio[0] contains the audio tensor
    return video[0], audio[0]


def crop_and_resize_image(image: Image.Image, height: int, width: int) -> Image.Image:
    """
    Crop and resize image to target dimensions, preserving aspect ratio.
    
    Args:
        image: Input PIL Image
        height: Target height
        width: Target width
        
    Returns:
        Cropped and resized PIL Image
    """
    # Calculate aspect ratios
    target_aspect = width / height
    image_aspect = image.width / image.height
    
    if image_aspect > target_aspect:
        # Image is wider, crop width
        new_width = int(image.height * target_aspect)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    elif image_aspect < target_aspect:
        # Image is taller, crop height
        new_height = int(image.width / target_aspect)
        top = (image.height - new_height) // 2
        image = image.crop((0, top, image.width, top + new_height))
    
    # Resize to target dimensions
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    
    return image


def validate_dimensions(height: int, width: int, num_frames: int) -> Tuple[int, int, int]:
    """
    Validate and adjust dimensions to meet MOVA requirements.
    
    Returns:
        Adjusted (height, width, num_frames)
    """
    # Height and width must be divisible by 16
    if height % 16 != 0:
        height = (height // 16) * 16
        print(f"[MOVA] Adjusted height to {height}")
    if width % 16 != 0:
        width = (width // 16) * 16
        print(f"[MOVA] Adjusted width to {width}")
    
    # num_frames - 1 must be divisible by 4
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        print(f"[MOVA] Adjusted num_frames to {num_frames}")
    
    return height, width, num_frames

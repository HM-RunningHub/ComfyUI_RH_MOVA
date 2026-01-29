"""
MOVA ComfyUI Nodes - Video and Audio Generation

Nodes:
- MOVAModelLoader: Load the MOVA model pipeline
- MOVASampler: Generate synchronized video and audio
"""

import os
import gc
import sys
import uuid
import subprocess
import importlib.util
import torch
import numpy as np
import cv2
from PIL import Image

import folder_paths

# Import ComfyUI VIDEO type
try:
    from comfy_api.input_impl.video_types import VideoFromFile
except ImportError:
    VideoFromFile = None

# Add current directory to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import mova_wrapper - try different methods
try:
    # Already loaded by __init__.py
    import mova_wrapper
except ImportError:
    # Load directly
    spec = importlib.util.spec_from_file_location(
        "mova_wrapper", 
        os.path.join(CURRENT_DIR, "mova_wrapper.py")
    )
    mova_wrapper = importlib.util.module_from_spec(spec)
    sys.modules["mova_wrapper"] = mova_wrapper
    spec.loader.exec_module(mova_wrapper)

# Get the functions we need
load_mova_pipeline = mova_wrapper.load_mova_pipeline
run_mova_inference = mova_wrapper.run_mova_inference
crop_and_resize_image = mova_wrapper.crop_and_resize_image
validate_dimensions = mova_wrapper.validate_dimensions

# Register model folder
MOVA_MODELS_DIR = os.path.join(folder_paths.models_dir, "MOVA")
if not os.path.exists(MOVA_MODELS_DIR):
    os.makedirs(MOVA_MODELS_DIR, exist_ok=True)


def get_available_models():
    """Get list of available MOVA models in the models directory."""
    models = []
    if os.path.exists(MOVA_MODELS_DIR):
        for name in os.listdir(MOVA_MODELS_DIR):
            model_path = os.path.join(MOVA_MODELS_DIR, name)
            # Check if it's a valid model directory (has model_index.json)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model_index.json")):
                models.append(name)
    return models if models else ["(No models found - place MOVA-360p or MOVA-720p in ComfyUI/models/MOVA/)"]


class RunningHub_MOVA_Loader:
    """
    Load the MOVA model for video-audio generation (fully offline).
    
    Models should be placed in: ComfyUI/models/MOVA/
    Supported models: MOVA-360p, MOVA-720p
    Note: No download attempts will be made - models must be pre-downloaded.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_available_models(), {
                    "default": get_available_models()[0] if get_available_models() else "",
                    "tooltip": "Select MOVA model. Place models in ComfyUI/models/MOVA/"
                }),
                "offload_mode": (["group", "cpu", "none"], {
                    "default": "group",
                    "tooltip": "Memory offload strategy. 'group' uses ~12GB VRAM (RTX 4090 compatible), 'cpu' uses ~48GB VRAM, 'none' loads all to GPU"
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision. bfloat16 recommended for best quality/performance balance"
                }),
            },
        }
    
    RETURN_TYPES = ("RH_MOVA_Pipeline",)
    RETURN_NAMES = ("MOVA Pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "RunningHub/MOVA"
    DESCRIPTION = "Load the MOVA model for synchronized video-audio generation"
    
    def load_model(self, model_name, offload_mode, dtype):
        # Check model exists (offline mode - no downloads)
        model_path = os.path.join(MOVA_MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            raise ValueError(
                f"Model not found: {model_path}\n"
                f"Please manually download MOVA-360p or MOVA-720p to: {MOVA_MODELS_DIR}\n"
                f"(This node runs in offline mode and will not download models)"
            )
        
        # Parse dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]
        
        # Load pipeline
        pipe = load_mova_pipeline(
            model_path=model_path,
            torch_dtype=torch_dtype,
            offload_mode=offload_mode,
            device_id=0,
        )
        
        return ({
            "pipe": pipe,
            "offload_mode": offload_mode,
            "dtype": torch_dtype,
            "model_path": model_path,
        },)


class RunningHub_MOVA_Sampler:
    """
    Generate synchronized video and audio using MOVA.
    
    Outputs video frames and audio that are perfectly synchronized.
    Great for talking head videos with natural lip sync.
    """
    
    NEGATIVE_PROMPT_DEFAULT = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
    )
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("RH_MOVA_Pipeline", {
                    "tooltip": "MOVA pipeline from RunningHub_MOVA_Loader"
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Reference image for the video. Will be cropped/resized to match output dimensions."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A person speaks naturally in a casual setting. They say, \"Hello, this is a test of the MOVA video generation system.\"",
                    "tooltip": "Describe the scene and what the person says. Use quotes for speech, e.g., 'He says, \"Hello world\"'"
                }),
                "width": ("INT", {
                    "default": 640,
                    "min": 256,
                    "max": 1920,
                    "step": 16,
                    "tooltip": "Video width (auto-adjusted to be divisible by 16)"
                }),
                "height": ("INT", {
                    "default": 352,
                    "min": 256,
                    "max": 1080,
                    "step": 16,
                    "tooltip": "Video height (auto-adjusted to be divisible by 16)"
                }),
                "num_frames": ("INT", {
                    "default": 97,
                    "min": 17,
                    "max": 385,
                    "step": 4,
                    "tooltip": "Number of frames. 97 frames ≈ 4 seconds at 24fps. Auto-adjusted so (num_frames-1) is divisible by 4."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 0.1,
                    "tooltip": "Video frame rate"
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Denoising steps. More steps = better quality but slower. 10-25 recommended."
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. Higher = more prompt adherence."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducibility"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt. Leave empty to use optimized default."
                }),
                "sigma_shift": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Sigma shift parameter for scheduler"
                }),
                "remove_video_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Free stage-1 video_dit after switching to reduce Host RAM (~28GB savings). Enable if running low on RAM."
                }),
            },
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "RunningHub/MOVA"
    DESCRIPTION = "Generate synchronized video and audio from a reference image and text prompt"
    
    def create_video_with_audio(self, frames_tensor, fps, audio_tensor, sample_rate, output_path):
        """
        Create video from frames tensor and add audio.
        
        Args:
            frames_tensor: Tensor of shape (N, H, W, C) with values in [0, 1]
            fps: Frames per second
            audio_tensor: Audio tensor [T] or [1, T]
            sample_rate: Audio sample rate
            output_path: Output video file path
        """
        temp_video_path = output_path.replace('.mp4', '_temp_video.mp4')
        temp_audio_path = output_path.replace('.mp4', '_temp_audio.wav')
        
        # Convert tensor to numpy frames
        frames_np = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
        num_frames, height, width, channels = frames_np.shape
        
        # Write frames to temp video using cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {temp_video_path}")
        
        for i in range(num_frames):
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"[MOVA] Wrote {num_frames} frames to temp video")
        
        # Save audio to temp wav file
        audio_np = audio_tensor.cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        
        # Normalize audio to int16 range
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        import wave
        with wave.open(temp_audio_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        print(f"[MOVA] Saved audio to temp file: {temp_audio_path}")
        
        # Combine video with audio using ffmpeg
        print(f"[MOVA] Combining video and audio...")
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-i', temp_audio_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
        
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if process.returncode != 0:
                print(f"[MOVA] FFmpeg error: {process.stderr}")
                raise RuntimeError(f"FFmpeg failed: {process.stderr}")
            print(f"[MOVA] Video created successfully: {output_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video encoding timed out")
        finally:
            # Clean up temp files
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        
        return output_path
    
    def create_video_object(self, video_path):
        """Create ComfyUI VIDEO object"""
        if VideoFromFile is not None:
            return VideoFromFile(video_path)
        else:
            # Fallback: return file path as string
            print("[MOVA] Warning: VideoFromFile not available, returning path string")
            return video_path
    
    def generate(
        self,
        pipeline,
        reference_image,
        prompt,
        width,
        height,
        num_frames,
        fps,
        steps,
        cfg_scale,
        seed,
        negative_prompt="",
        sigma_shift=5.0,
        remove_video_dit=False,
    ):
        import comfy.utils
        
        pipe = pipeline["pipe"]
        
        # Validate and adjust dimensions
        height, width, num_frames = validate_dimensions(height, width, num_frames)
        
        # Use default negative prompt if empty
        if not negative_prompt or negative_prompt.strip() == "":
            negative_prompt = self.NEGATIVE_PROMPT_DEFAULT
        
        # Convert ComfyUI IMAGE tensor to PIL Image
        # ComfyUI IMAGE format: [B, H, W, C] with values 0-1
        if reference_image.dim() == 4:
            ref_img_np = reference_image[0].cpu().numpy()
        else:
            ref_img_np = reference_image.cpu().numpy()
        
        ref_img_np = (ref_img_np * 255).astype(np.uint8)
        ref_img_pil = Image.fromarray(ref_img_np, mode="RGB")
        
        # Crop and resize to target dimensions
        ref_img = crop_and_resize_image(ref_img_pil, height=height, width=width)
        
        # Create ComfyUI progress bar
        pbar = comfy.utils.ProgressBar(steps)
        
        def progress_callback(step, total):
            pbar.update(1)
        
        # Run inference with progress callback
        video_frames, audio_tensor = run_mova_inference(
            pipe=pipe,
            prompt=prompt,
            reference_image=ref_img,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=steps,
            cfg_scale=cfg_scale,
            sigma_shift=sigma_shift,
            seed=seed,
            remove_video_dit=remove_video_dit,
            progress_callback=progress_callback,
        )
        
        # Convert video (list of PIL Images) to ComfyUI IMAGE format [N, H, W, C]
        if isinstance(video_frames, list):
            # List of PIL Images
            frames_np = np.stack([np.array(frame) for frame in video_frames], axis=0)
        else:
            # Already numpy or tensor
            frames_np = np.array(video_frames)
        
        # Normalize to 0-1 range for ComfyUI
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0
        
        # Get audio sample rate
        sample_rate = pipe.audio_sample_rate
        
        # Create output video with audio
        output_dir = folder_paths.get_output_directory()
        output_filename = f"mova_{uuid.uuid4()}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        self.create_video_with_audio(frames_tensor, fps, audio_tensor, sample_rate, output_path)
        
        # Create VIDEO object
        video_obj = self.create_video_object(output_path)
        
        # Clear some memory
        gc.collect()
        torch.cuda.empty_cache()
        
        return (video_obj,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "RunningHub_MOVA_Loader": RunningHub_MOVA_Loader,
    "RunningHub_MOVA_Sampler": RunningHub_MOVA_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub_MOVA_Loader": "RunningHub MOVA Loader",
    "RunningHub_MOVA_Sampler": "RunningHub MOVA Sampler",
}

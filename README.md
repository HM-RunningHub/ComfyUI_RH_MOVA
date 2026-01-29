# ComfyUI-MOVA

ComfyUI nodes for **MOVA** (MOSS Video and Audio) - a foundation model for synchronized video-audio generation.

## Features

- **Native Bimodal Generation**: Generate video and synchronized audio in a single pass
- **Precise Lip-Sync**: State-of-the-art multilingual lip synchronization
- **ComfyUI Integration**: Easy-to-use nodes for the ComfyUI workflow

## Installation

### 1. Install MOVA Package

First, install the MOVA package from the parent directory:

```bash
cd /path/to/MOVA
pip install -e .
```

### 2. Install ComfyUI Node

Copy or symlink the `ComfyUI-MOVA` folder to your ComfyUI custom_nodes directory:

```bash
# Option 1: Symlink (recommended for development)
ln -s /path/to/MOVA/ComfyUI-MOVA /path/to/ComfyUI/custom_nodes/ComfyUI-MOVA

# Option 2: Copy
cp -r /path/to/MOVA/ComfyUI-MOVA /path/to/ComfyUI/custom_nodes/
```

### 3. Download Model

Download the MOVA model from HuggingFace and place it in `ComfyUI/models/MOVA/`:

```bash
# Using huggingface-cli
huggingface-cli download OpenMOSS-Team/MOVA-360p --local-dir /path/to/ComfyUI/models/MOVA/MOVA-360p

# Or for 720p model (larger, higher quality)
huggingface-cli download OpenMOSS-Team/MOVA-720p --local-dir /path/to/ComfyUI/models/MOVA/MOVA-720p
```

## Nodes

### MOVA Model Loader

Loads the MOVA model with configurable memory management.

**Inputs:**
- `model_name`: Select from available models in ComfyUI/models/MOVA/
- `offload_mode`: Memory management strategy
  - `group`: Lowest VRAM usage (~12GB) - recommended for RTX 4090
  - `cpu`: Medium VRAM usage (~48GB)
  - `none`: Loads everything to GPU (requires high VRAM)
- `dtype`: Model precision (bfloat16 recommended)

**Outputs:**
- `mova_model`: Model object for use with MOVA Sampler

### MOVA Sampler

Generates synchronized video and audio from a reference image and text prompt.

**Inputs:**
- `mova_model`: Model from MOVA Model Loader
- `reference_image`: Reference image (will be cropped/resized)
- `prompt`: Text description including speech in quotes
- `width/height`: Output video dimensions (auto-adjusted to be divisible by 16)
- `num_frames`: Number of frames (97 frames ≈ 4 seconds at 24fps)
- `fps`: Frame rate
- `steps`: Denoising steps (10-25 recommended)
- `cfg_scale`: Guidance scale
- `seed`: Random seed

**Outputs:**
- `video_frames`: Video frames as IMAGE tensor
- `audio`: Audio as AUDIO dict

## Usage Tips

### Prompt Format

For best results, describe the scene and use quotes for speech:

```
A man in a blue blazer speaks in a formal indoor setting. 
He says, "I would also say that this election wasn't surprising."
```

### Dimension Requirements

- Width and height must be divisible by 16
- num_frames - 1 must be divisible by 4
- The nodes auto-adjust values if needed

### Memory Requirements

| Offload Mode | VRAM | Host RAM | Recommended GPU |
|--------------|------|----------|-----------------|
| group | ~12GB | ~77GB | RTX 4090 |
| cpu | ~48GB | ~67GB | H100/A100 |
| none | ~80GB+ | Minimal | Multi-GPU |

## Example Workflow

1. **Load Image** → Load reference image of a person
2. **MOVA Model Loader** → Load model with `group` offload for RTX 4090
3. **MOVA Sampler** → Generate video with prompt
4. **VHS Video Combine** → Combine frames and audio into video file

## Troubleshooting

### "No models found"

Place MOVA model in `ComfyUI/models/MOVA/MODEL_NAME/`. The folder should contain `model_index.json`.

### Out of Memory

- Use `group` offload mode
- Reduce resolution (e.g., 352x640)
- Reduce num_frames (e.g., 49 frames)
- Enable `remove_video_dit` option

### Module Not Found

Make sure MOVA is installed: `pip install -e /path/to/MOVA`

## License

Apache 2.0 - Same as MOVA

## Credits

- [MOVA](https://github.com/OpenMOSS/MOVA) by OpenMOSS Team
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous

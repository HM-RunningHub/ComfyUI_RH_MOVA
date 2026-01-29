# ComfyUI_RH_MOVA

![License](https://img.shields.io/badge/License-Apache%202.0-green)

ComfyUI custom nodes for **MOVA** (MOSS Video and Audio) - a foundation model for synchronized video-audio generation with precise lip-sync.

> Based on [OpenMOSS/MOVA](https://github.com/OpenMOSS/MOVA)

## ✨ Features

- **Native Bimodal Generation**: Generate video and synchronized audio in a single pass
- **Precise Lip-Sync**: State-of-the-art multilingual lip synchronization
- **Sound Effects**: Environment-aware sound effects generation
- **Memory Efficient**: Support for RTX 4090 with group offload mode (~12GB VRAM)
- **ComfyUI Integration**: Easy-to-use nodes for the ComfyUI workflow

## 🛠️ Installation

### Step 1: Clone this repository

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HM-RunningHub/ComfyUI_RH_MOVA.git
```

### Step 2: Install MOVA package

The MOVA package must be installed in ComfyUI's Python environment:

```bash
# Option A: Install from GitHub (recommended)
pip install git+https://github.com/OpenMOSS/MOVA.git

# Option B: Install from local clone
git clone https://github.com/OpenMOSS/MOVA.git
cd MOVA
pip install -e .
```

### Step 3: Install dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI_RH_MOVA
pip install -r requirements.txt
```

### Step 4: Install FFmpeg

FFmpeg is required for video encoding with audio:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (using chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg
```

## 📦 Model Download

Download MOVA models from HuggingFace and place them in the correct directory.

### Model Directory Structure

```
ComfyUI/
└── models/
    └── MOVA/                          # ← Create this folder
        ├── MOVA-360p/                 # 360p model (smaller, faster)
        │   ├── model_index.json
        │   ├── scheduler/
        │   ├── text_encoder/
        │   ├── tokenizer/
        │   ├── video_dit/
        │   ├── video_dit_2/
        │   ├── audio_dit/
        │   ├── vae_2d/
        │   └── ...
        └── MOVA-720p/                 # 720p model (larger, higher quality)
            └── ...
```

### Download Commands

```bash
# Create the MOVA models directory
mkdir -p ComfyUI/models/MOVA

# Download 360p model (recommended for RTX 4090)
huggingface-cli download OpenMOSS-Team/MOVA-360p --local-dir ComfyUI/models/MOVA/MOVA-360p

# Or download 720p model (higher quality, requires more VRAM)
huggingface-cli download OpenMOSS-Team/MOVA-720p --local-dir ComfyUI/models/MOVA/MOVA-720p
```

### Alternative: Manual Download

1. Visit [https://huggingface.co/OpenMOSS-Team/MOVA-360p](https://huggingface.co/OpenMOSS-Team/MOVA-360p)
2. Click "Files and versions" tab
3. Download all files and folders
4. Place them in `ComfyUI/models/MOVA/MOVA-360p/`

## 🚀 Nodes

### RunningHub MOVA Loader

Loads the MOVA model with configurable memory management.

| Parameter | Description |
|-----------|-------------|
| `model_name` | Select from available models in `ComfyUI/models/MOVA/` |
| `offload_mode` | Memory strategy: `group` (~12GB VRAM), `cpu` (~48GB VRAM), `none` (full GPU) |
| `dtype` | Model precision: `bfloat16` (recommended), `float16`, `float32` |

### RunningHub MOVA Sampler

Generates synchronized video and audio from a reference image and text prompt.

| Parameter | Description |
|-----------|-------------|
| `pipeline` | MOVA pipeline from Loader |
| `reference_image` | Reference image (person/scene) |
| `prompt` | Text description with speech in quotes |
| `width/height` | Output dimensions (auto-adjusted to be divisible by 16) |
| `num_frames` | Frame count (97 frames ≈ 4 seconds at 24fps) |
| `fps` | Frame rate |
| `steps` | Denoising steps (10-25 recommended) |
| `cfg_scale` | Guidance scale |
| `seed` | Random seed for reproducibility |

**Output**: VIDEO object with synchronized audio

## 📂 Example Workflow

An example workflow is provided in the `workflows/` folder:

- **[mova_basic_example.json](workflows/mova_basic_example.json)** - Basic workflow for generating talking head video

### How to use:

1. In ComfyUI, drag and drop the JSON file into the canvas, or use "Load" to import
2. Replace `your_reference_image.jpg` with your own image
3. Modify the prompt text (use quotes for speech content)
4. Click "Queue Prompt" to generate

### Workflow Structure:

```
LoadImage → RunningHub MOVA Loader → RunningHub MOVA Sampler → Save Video
```

## 📝 Prompt Format

For best results, describe the scene and use quotes for speech:

```
A man in a blue blazer speaks in a formal indoor setting.
He says, "I would also say that this election wasn't surprising."
```

For multi-person scenes:

```
The scene shows a man and a child walking through a park.
The man asks, "What do you want to do when you grow up?"
The boy answers, "A bond trader."
```

## 💻 Hardware Requirements

| Offload Mode | VRAM | Host RAM | Recommended GPU |
|--------------|------|----------|-----------------|
| `group` | ~12GB | ~77GB | RTX 4090 |
| `cpu` | ~48GB | ~67GB | H100/A100 |
| `none` | ~80GB+ | Minimal | Multi-GPU |

### Performance Reference (8s 360p video)

| Hardware | Offload | Step Time |
|----------|---------|-----------|
| RTX 4090 | group | ~42s |
| RTX 4090 | cpu | ~38s |
| H100 | group | ~23s |
| H100 | cpu | ~9s |

## 🔧 Troubleshooting

### "No models found"

Ensure models are placed correctly:
```
ComfyUI/models/MOVA/MOVA-360p/model_index.json  ← This file must exist
```

### Out of Memory

- Use `group` offload mode (lowest VRAM)
- Reduce resolution (e.g., 352x640 for 360p)
- Reduce `num_frames` (e.g., 49 frames for ~2 seconds)
- Enable `remove_video_dit` option to save ~28GB Host RAM

### Module Not Found: mova

```bash
pip install git+https://github.com/OpenMOSS/MOVA.git
```

### FFmpeg not found

Install FFmpeg for your system (see Installation Step 4).

## 📄 License

Apache 2.0 - Same as [MOVA](https://github.com/OpenMOSS/MOVA)

## 🙏 Credits

- [MOVA](https://github.com/OpenMOSS/MOVA) by OpenMOSS Team
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous

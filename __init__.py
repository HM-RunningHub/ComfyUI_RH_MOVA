"""
ComfyUI-MOVA: Video and Audio Generation Node Package

This package provides ComfyUI nodes for MOVA (MOSS Video and Audio),
a foundation model for synchronized video-audio generation.

Nodes:
- MOVAModelLoader: Load the MOVA model pipeline
- MOVASampler: Generate synchronized video and audio

Models should be placed in: ComfyUI/models/MOVA/
Download from: https://huggingface.co/OpenMOSS-Team/MOVA-360p
"""

import os
import sys
import importlib.util

# Add parent directory (MOVA root) to path so we can import mova package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MOVA_ROOT = os.path.dirname(CURRENT_DIR)

if MOVA_ROOT not in sys.path:
    sys.path.insert(0, MOVA_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import mova_wrapper directly
def _import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load mova_wrapper
mova_wrapper = _import_module_from_path(
    "mova_wrapper", 
    os.path.join(CURRENT_DIR, "mova_wrapper.py")
)
mova_wrapper.patch_torch_distributed()

# Load nodes
nodes_module = _import_module_from_path(
    "mova_nodes",
    os.path.join(CURRENT_DIR, "nodes.py")
)

NODE_CLASS_MAPPINGS = nodes_module.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = nodes_module.NODE_DISPLAY_NAME_MAPPINGS

# Define web directory for any future frontend extensions
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

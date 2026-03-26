# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kimodo is a **ki**nematic **mo**tion **d**iffusi**o**n model for generating high-quality 3D human and robot motions from text prompts and kinematic constraints. It supports multiple skeletons (SOMA human body, Unitree G1 robot, SMPL-X) and outputs various formats (BVH, MuJoCo CSV, AMASS NPZ).

## Installation

```bash
pip install -e .           # Base install
pip install -e .[demo]     # With Viser 3D visualization (required for interactive demo)
pip install -e .[soma]     # With SOMA body model support
pip install -e .[all]      # Everything
```

The MotionCorrection C++ library (foot skate correction) builds automatically via CMake during install. Requires CMake 3.15+.

## CLI Commands

```bash
# Generate motion from text
kimodo_gen "a person walks forward" --model Kimodo-SOMA-RP-v1 --duration 5.0

# Launch interactive Gradio demo at http://127.0.0.1:7860
kimodo_demo

# Run text encoder as a server (for API-based text encoding)
kimodo_textencoder
```

Or equivalently: `python -m kimodo.scripts.generate`, `python -m kimodo.demo`, etc.

## Linting

```bash
ruff check .          # Lint + import sorting
ruff format .         # Format
flake8 .              # Style check (max-line-length: 120)
```

## Documentation

```bash
pip install -r docs/requirements.txt
cd docs && make html          # Build HTML docs
make apidoc && make html      # Regenerate API reference first
```

## Architecture

### Inference Pipeline

The core inference flow: text → embeddings → diffusion denoising → motion features → skeleton poses → post-processing → export.

1. **Text Encoding** (`kimodo/model/llm2vec/`) — LLM2Vec (Llama-3-8B-based) embeds text prompts. Can run locally or via a remote server (`kimodo_textencoder`).
2. **Diffusion Model** (`kimodo/model/backbone.py`, `diffusion.py`) — Transformer denoiser with DDIM sampling and classifier-free guidance (CFG). Supports `nocfg`, `regular`, and `separated` (text vs. constraint) CFG modes.
3. **Motion Representation** (`kimodo/motion_rep/`) — Encodes/decodes skeleton poses to/from the latent feature space the diffusion model operates on. Different reps for different skeletons.
4. **Constraints** (`kimodo/constraints.py`) — Full-body pose keyframes, end-effector positions/rotations, 2D root paths and waypoints. Constraints are injected into the diffusion process.
5. **Post-Processing** (`kimodo/postprocess.py`) — Foot skate correction (via the C++ MotionCorrection library) and constraint enforcement.
6. **Skeleton & Kinematics** (`kimodo/skeleton/`) — Base skeleton class with hierarchy and FK/IK. Subclasses for SOMA-30, SOMA-77, G1, SMPL-X, and BVH format.
7. **Exports** (`kimodo/exports/`) — Output formats: NPZ (default), MuJoCo CSV (G1), AMASS NPZ (SMPL-X), BVH (SOMA).

### Model Registry

`kimodo/model/registry.py` defines 5 canonical models. `kimodo/model/load_model.py` handles Hugging Face model resolution and text encoder selection (local vs. remote API). `kimodo/model/loading.py` instantiates configs.

### Interactive Demo

The demo (`kimodo/demo/`) is a Gradio web app with Viser 3D visualization. Key files:
- `app.py` — Main Gradio app, timeline editor UI
- `generation.py` — Wraps the Kimodo model for async generation
- `ui.py` — UI components and state management

### Visualization

`kimodo/viz/` handles 3D rendering: skeleton display, skinned mesh rendering (SOMA, G1, SMPL-X), and constraint visualization. Uses the custom Viser fork (`nv-tlabs/kimodo-viser`).

### Configuration

Hydra + OmegaConf manages model configs. Configs live under `kimodo/assets/` and model-specific asset directories.

## Contributing

All commits must be signed off (DCO requirement):

```bash
git commit -s -m "Your message"
```

All contributions require a GitHub PR review before merging.

## Hardware Requirements

- ~17GB VRAM for local inference (primarily due to the LLM2Vec text encoder)
- Tested on RTX 3090, RTX 4090, A100
- Linux recommended; Windows supported via Docker

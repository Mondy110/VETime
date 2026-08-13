# Multi-Scale STFT Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the auxiliary ViCO mixed-view image with a strict three-scale STFT time-frequency image while retaining the trend-decomposition time-domain branch.

**Architecture:** Add an `stft_multiscale` renderer that removes padding, computes log-magnitude Hann-window STFTs with fixed windows 32, 64, and 128 (hop = window / 4), and stacks the resized spectra as RGB channels. The existing renderer factory and trainer interface remain unchanged; configuration selects the new renderer.

**Tech Stack:** Python, NumPy, SciPy STFT, PyTorch, pytest.

## Global Constraints

- Do not resample or interpolate the one-dimensional source series before STFT.
- Use fixed windows `[32, 64, 128]` and hops `[8, 16, 32]` for every valid sequence.
- Resize only completed two-dimensional spectra to the vision encoder image size.
- Keep the existing trend-decomposition image branch and model fusion API unchanged.

---

### Task 1: Specify and test the strict time-frequency renderer

**Files:**
- Create: `tests/datasets/renderers/test_multiscale_stft_renderer.py`
- Create: `src/datasets/renderers/multiscale_stft.py`
- Modify: `src/datasets/renderers/__init__.py`

**Interfaces:**
- Produces `MultiScaleSTFTRenderer.render_batch(time_series, att_mask=None, img_size=224) -> torch.Tensor`.
- Registers factory name `stft_multiscale`.

- [ ] **Step 1: Write failing tests** for factory registration, `[B, 3, H, W]` float32 bounded output, and padding invariance.
- [ ] **Step 2: Run the renderer test** and confirm import/factory failure.
- [ ] **Step 3: Implement the minimal renderer** using per-channel STFT aggregation, log magnitudes, and post-STFT resizing.
- [ ] **Step 4: Run the renderer test** and confirm it passes.

### Task 2: Select the time-frequency renderer in the application configuration

**Files:**
- Modify: `configs/base.yaml`
- Modify: `tests/integration/test_renderer_integration.py`

**Interfaces:**
- `model.vision_branch.vico_renderer: stft_multiscale` constructs `MultiScaleSTFTRenderer` through the existing trainer factory.

- [ ] **Step 1: Write a failing integration assertion** for the configured renderer.
- [ ] **Step 2: Update config and imports** without changing default renderer behavior.
- [ ] **Step 3: Run renderer and integration tests** and confirm the configured path passes.

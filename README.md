# Anime Tagger + Image Processing — `convnextv2_huge.dbv4-full`

A local Gradio web app for multilabel tagging of anime-style images using the [`animetimm/convnextv2_huge.dbv4-full`](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full) model. Outputs Danbooru-style comma-separated captions suitable for LoRA and fine-tuning dataset preparation.

---

## Features

- **Single image mode** — tag one image at a time and view results broken out by category (rating, general, character), with a combined caption string
- **Batch tagging mode** — upload multiple images, tag them all, and download a `captions.zip` containing one `.txt` caption file per image
- **Image processing** — resize images to a target resolution (longest side); automatically upscales with Real-ESRGAN or downscales with Lanczos depending on the source size
- **Trigger word prefix** — optionally prepend a custom trigger word (e.g. a LoRA token) to every caption
- **Three threshold modes** — per-category recommended thresholds, per-tag optimal thresholds, or a custom global slider
- **VRAM management** — load and unload the tagger and upsampler independently from the UI to free VRAM between tasks
- **GPU acceleration** — automatically uses CUDA if available, falls back to CPU

---

## Model

| Property | Value |
|---|---|
| Architecture | ConvNeXtV2-Huge |
| Parameters | 692.6M |
| Input resolution | 512 × 512 |
| Tag vocabulary | 12,476 Danbooru tags |
| Tag categories | General (9,225) · Character (3,247) · Rating (4) |
| License | GPL-3.0 |

The model weights (~2.7 GB) are downloaded automatically from HuggingFace on first run. VRAM usage is approximately 4 GB during inference.

> **Note:** This model requires you to accept the HuggingFace repository terms before downloading. Log in first with:
> ```bash
> hf auth login
> ```

---

## Requirements

- Python 3.10+
- PyTorch (install separately — see below)
- 4 GB+ VRAM for GPU inference; CPU inference works but is significantly slower
- For image upscaling: `realesrgan` and `basicsr` (included in `requirements.txt`)

---

## Installation

**1. Install PyTorch** for your CUDA version from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). For example, for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**2. Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

**3. Authenticate with HuggingFace** (required to download the gated model):

```bash
hf auth login
```

---

## Usage

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

---

## Tabs

### Single Image

1. Upload an image
2. Select a threshold mode
3. Optionally enter a trigger word
4. Click **Tag Image**

Results are displayed as confidence-ranked label lists per category, plus a combined caption string in the order `trigger_word, rating, general, character`.

### Batch Tagging

1. Upload one or more images
2. Select a threshold mode and optional trigger word
3. Click **Tag All & Export ZIP**

Each image produces a `.txt` file named after the source image (e.g. `img001.txt`). All files are bundled into `captions.zip` for download. A processing log shows tag count and status per file.

### Image Processing

Resize a batch of images to a **target resolution** (longest side in px). The method is chosen automatically per image:

| Condition | Method |
|---|---|
| Source longest side < target | Real-ESRGAN x4 (`anime_6B`) → Lanczos to exact target |
| Source longest side > target | Lanczos downscale |
| Source longest side = target | Unchanged |

Aspect ratio is always preserved. Quick preset buttons are provided for common training resolutions: **512 · 768 · 1024 · 2048**.

Output can be saved as **PNG** (lossless) or **JPEG** (with adjustable quality). All processed images are bundled into `processed.zip`.

The Real-ESRGAN model (~17 MB) downloads automatically on first use. The upsampler can be unloaded from the UI to free VRAM after processing.

---

## Threshold Modes

| Mode | Description |
|---|---|
| Per-category (recommended) | Model card thresholds: General `0.38`, Character `0.51`, Rating `0.24` |
| Best per-tag | Per-tag optimal threshold from `selected_tags.csv` — maximises F1 per tag |
| Custom | Single global threshold slider applied to all tags |

---

## Caption Format

Captions are written as a single comma-separated line:

```
trigger_word, rating_tag, general_tag_1, general_tag_2, ..., character_tag_1, ...
```

The trigger word is omitted if left blank. This format is directly compatible with most LoRA training tools (kohya-ss, SimpleTuner, etc.).

---

## Compatibility Notes

**`basicsr` / `realesrgan` on newer torchvision:** `basicsr` depends on `torchvision.transforms.functional_tensor`, which was removed in torchvision 0.17+. The app patches this automatically at runtime using a compatibility shim — no downgrade required.

---

## Project Structure

```
.
├── app.py            # Main Gradio application
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Credits

- Model by [narugo1992](https://huggingface.co/narugo1992) and the [DeepGHS](https://github.com/deepghs) team
- Upscaling via [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (`RealESRGAN_x4plus_anime_6B`)
- Built on [timm](https://github.com/huggingface/pytorch-image-models) and [dghs-imgutils](https://github.com/deepghs/imgutils)
- UI powered by [Gradio](https://www.gradio.app/)

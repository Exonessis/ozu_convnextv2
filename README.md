# Anime Tagger — `convnextv2_huge.dbv4-full`

A local Gradio web app for multilabel tagging of anime-style images using the [`animetimm/convnextv2_huge.dbv4-full`](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full) model. Outputs Danbooru-style comma-separated captions suitable for LoRA and fine-tuning dataset preparation.

---

## Features

- **Single image mode** — tag one image at a time and view results broken out by category (rating, general, character), with a copyable caption string
- **Batch mode** — upload multiple images, tag them all, and download a `captions.zip` containing one `.txt` caption file per image
- **Trigger word prefix** — optionally prepend a custom trigger word (e.g. a LoRA token) to every caption
- **Three threshold modes** — choose between per-category recommended thresholds, per-tag optimal thresholds, or a custom global slider
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

The model weights (~2.7 GB) are downloaded automatically from HuggingFace on first run.

> **Note:** This model requires you to accept the HuggingFace repository terms before downloading. Log in first with:
> ```bash
> hf auth login
> ```

---

## Requirements

- Python 3.10+
- PyTorch (install separately — see below)
- 8 GB+ VRAM recommended for GPU inference (Though, it only consumes 4GB VRAM); CPU inference works but is significantly slower

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

### Single Image tab

1. Upload an image
2. Select a threshold mode
3. Optionally enter a trigger word
4. Click **Tag Image**

Results are displayed as confidence-ranked label lists per category, plus a combined caption string in the order `trigger_word, rating, general, character`.

### Batch Tagging tab

1. Upload one or more images
2. Select a threshold mode
3. Optionally enter a trigger word to be prepended to every caption
4. Click **Tag All & Export ZIP**

Each image produces a `.txt` file named after the image (e.g. `img001.txt`). All files are bundled into `captions.zip` for download. A processing log shows the status and tag count for each file.

---

## Threshold Modes

| Mode | Description |
|---|---|
| Per-category (recommended) | Uses the thresholds from the model card: General `0.38`, Character `0.51`, Rating `0.24` |
| Best per-tag | Uses the per-tag optimal threshold from `selected_tags.csv` — maximises F1 per tag |
| Custom | A single global threshold slider applied to all tags |

---

## Caption Format

Captions are written as a single comma-separated line:

```
trigger_word, rating_tag, general_tag_1, general_tag_2, ..., character_tag_1, ...
```

The trigger word is omitted if left blank. This format is directly compatible with most LoRA training tools (kohya-ss, SimpleTuner, etc.).

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
- Built on [timm](https://github.com/huggingface/pytorch-image-models) and [dghs-imgutils](https://github.com/deepghs/imgutils)
- UI powered by [Gradio](https://www.gradio.app/)

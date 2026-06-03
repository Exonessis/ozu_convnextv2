import json
import os
import tempfile
import zipfile
import numpy as np
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from imgutils.preprocess import create_torchvision_transforms
from timm import create_model
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ID = "animetimm/convnextv2_huge.dbv4-full"
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_THRESHOLDS = {0: 0.38, 4: 0.51, 9: 0.24}
CATEGORY_NAMES      = {0: "General", 4: "Character", 9: "Rating"}

# ── Tag metadata ──────────────────────────────────────────────────────────────
print("Loading tag metadata...")
with open(hf_hub_download(repo_id=REPO_ID, repo_type="model", filename="preprocess.json")) as f:
    preprocessor = create_torchvision_transforms(json.load(f)["test"])

df_tags = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, repo_type="model", filename="selected_tags.csv"),
    keep_default_na=False,
)
print("Tag metadata ready.")

# ── Tagger model (lazy) ───────────────────────────────────────────────────────
_tagger = None


def load_model():
    global _tagger
    if _tagger is not None:
        return "✅ Tagger already loaded."
    print("Loading tagger weights...")
    _tagger = create_model(f"hf-hub:{REPO_ID}", pretrained=True)
    _tagger.eval()
    _tagger.to(DEVICE)
    print(f"Tagger loaded on: {DEVICE}")
    return f"✅ Tagger loaded on {DEVICE}."


def unload_model():
    global _tagger
    if _tagger is None:
        return "⚠️ Tagger is not loaded."
    del _tagger
    _tagger = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "🗑️ Tagger unloaded — VRAM freed."


def model_status():
    return f"✅ Tagger on {DEVICE}" if _tagger is not None else "⚠️ Tagger not loaded"


load_model()

# ── Real-ESRGAN upsampler (lazy) ──────────────────────────────────────────────
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)
_upsampler = None


def get_upsampler():
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    # basicsr uses torchvision.transforms.functional_tensor which was removed
    # in torchvision 0.17+. Patch it as an alias before importing basicsr.
    import sys, types
    import torchvision.transforms.functional as _tvf
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        _ft = types.ModuleType("torchvision.transforms.functional_tensor")
        for _attr in dir(_tvf):
            setattr(_ft, _attr, getattr(_tvf, _attr))
        sys.modules["torchvision.transforms.functional_tensor"] = _ft

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    esrgan_model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=6, num_grow_ch=32, scale=4,
    )
    _upsampler = RealESRGANer(
        scale=4,
        model_path=REALESRGAN_URL,
        model=esrgan_model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=DEVICE.type == "cuda",
        device=DEVICE,
    )
    print("Real-ESRGAN upsampler ready.")
    return _upsampler


def unload_upsampler():
    global _upsampler
    if _upsampler is None:
        return "⚠️ Upsampler is not loaded."
    del _upsampler
    _upsampler = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "🗑️ Upsampler unloaded — VRAM freed."


def upsampler_status():
    return (
        f"✅ Upsampler on {DEVICE}"
        if _upsampler is not None
        else "⚪ Upsampler not loaded (loads automatically when needed)"
    )


# ── Tagger inference ──────────────────────────────────────────────────────────
def run_inference(pil_image: Image.Image) -> torch.Tensor:
    if _tagger is None:
        raise RuntimeError("Tagger is not loaded. Click 'Load Tagger' first.")
    image  = pil_image.convert("RGB")
    tensor = preprocessor(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(_tagger(tensor))[0].cpu()
    return pred


def prediction_to_dicts(prediction, threshold_mode, custom_threshold, trigger_word=""):
    results = {"General": {}, "Character": {}, "Rating": {}}
    for idx, row in df_tags.iterrows():
        if idx >= len(prediction):
            continue
        score = float(prediction[idx])
        cat   = row["category"]
        name  = row["name"]
        if threshold_mode == "Per-category (recommended)":
            thr = CATEGORY_THRESHOLDS.get(cat, 0.40)
        elif threshold_mode == "Best per-tag":
            thr = float(row["best_threshold"])
        else:
            thr = custom_threshold
        if score >= thr:
            results[CATEGORY_NAMES.get(cat, f"Category {cat}")][name] = round(score, 4)
    for k in results:
        results[k] = dict(sorted(results[k].items(), key=lambda x: -x[1]))
    parts = (
        list(results["Rating"].keys())
        + list(results["General"].keys())
        + list(results["Character"].keys())
    )
    trigger = trigger_word.strip()
    if trigger:
        parts = [trigger] + parts
    return results["Rating"], results["General"], results["Character"], ", ".join(parts)


def tag_single(pil_image, threshold_mode, custom_threshold, trigger_word):
    if pil_image is None:
        return {}, {}, {}, "", model_status()
    try:
        pred = run_inference(pil_image)
        rating, general, character, caption = prediction_to_dicts(
            pred, threshold_mode, custom_threshold, trigger_word
        )
        return rating, general, character, caption, model_status()
    except RuntimeError as e:
        return {}, {}, {}, str(e), model_status()


def tag_batch(files, threshold_mode, custom_threshold, trigger_word, progress=gr.Progress()):
    if not files:
        return None, "No files uploaded.", model_status()
    if _tagger is None:
        return None, "⚠️ Tagger is not loaded.", model_status()
    tmp_dir  = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "captions.zip")
    log_lines = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in progress.tqdm(files, desc="Tagging"):
            try:
                img_path = Path(file_path)
                pil_img  = Image.open(img_path).convert("RGB")
                pred     = run_inference(pil_img)
                _, _, _, caption = prediction_to_dicts(
                    pred, threshold_mode, custom_threshold, trigger_word
                )
                txt_name = img_path.stem + ".txt"
                txt_path = os.path.join(tmp_dir, txt_name)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                zf.write(txt_path, arcname=txt_name)
                log_lines.append(
                    f"✅ {img_path.name}  →  {txt_name}  ({len(caption.split(','))} tags)"
                )
            except Exception as e:
                log_lines.append(f"❌ {Path(file_path).name}  —  Error: {e}")
    return zip_path, "\n".join(log_lines), model_status()


# ── Image processing (target resolution) ─────────────────────────────────────
def _resize_to_target(img: Image.Image, target: int):
    """
    Resize so the longest side equals `target`, preserving aspect ratio.
      target > longest  →  Real-ESRGAN x4, then Lanczos to exact target
      target < longest  →  Lanczos straight down
      target == longest →  unchanged
    Returns (output_image, method_label).
    """
    w, h    = img.size
    longest = max(w, h)

    if longest == target:
        return img.copy(), "unchanged"

    scale = target / longest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    if target > longest:
        upsampler = get_upsampler()
        arr = np.array(img.convert("RGB"))
        out_arr, _ = upsampler.enhance(arr, outscale=4)
        out = Image.fromarray(out_arr).resize((new_w, new_h), Image.LANCZOS)
        method = "ESRGAN → Lanczos"
    else:
        out = img.resize((new_w, new_h), Image.LANCZOS)
        method = "Lanczos"

    return out, method


def process_images(files, target_res, out_format, jpeg_quality, progress=gr.Progress()):
    if not files:
        return None, "No files uploaded.", upsampler_status()
    tmp_dir  = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "processed.zip")
    log_lines = []
    ext    = ".png" if out_format == "PNG" else ".jpg"
    target = int(target_res)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in progress.tqdm(files, desc="Processing"):
            try:
                img_path       = Path(file_path)
                img            = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img.size

                out, method = _resize_to_target(img, target)

                out_name = img_path.stem + ext
                out_path = os.path.join(tmp_dir, out_name)
                save_kw  = {"quality": int(jpeg_quality), "optimize": True} if ext == ".jpg" else {}
                out.save(out_path, **save_kw)
                zf.write(out_path, arcname=out_name)

                new_w, new_h = out.size
                log_lines.append(
                    f"✅ {img_path.name}  {orig_w}×{orig_h} → {new_w}×{new_h}  [{method}]  →  {out_name}"
                )
            except Exception as e:
                log_lines.append(f"❌ {Path(file_path).name}  —  Error: {e}")

    return zip_path, "\n".join(log_lines), upsampler_status()


# ── Preset helper ─────────────────────────────────────────────────────────────
def make_preset(px):
    return lambda: px


# ── Shared UI helpers ─────────────────────────────────────────────────────────
def threshold_controls():
    mode = gr.Radio(
        choices=["Per-category (recommended)", "Best per-tag", "Custom"],
        value="Per-category (recommended)",
        label="Threshold mode",
    )
    custom = gr.Slider(0.0, 1.0, value=0.40, step=0.01,
                       label="Custom threshold", visible=False)
    mode.change(lambda m: gr.update(visible=(m == "Custom")),
                inputs=mode, outputs=custom)
    return mode, custom


# ── UI ────────────────────────────────────────────────────────────────────────
CSS = ".tag-box { font-family: monospace; font-size: 0.85em; }"

with gr.Blocks(title="Anime Tagger — convnextv2_huge.dbv4-full") as demo:
    gr.Markdown(
        "## 🏷️ Anime Tagger — `convnextv2_huge.dbv4-full`\n"
        "Multilabel classifier · 12,476 Danbooru tags · image resizer included."
    )

    # ── Shared tagger controls ───────────────────────────────────────────────
    with gr.Row():
        status_box = gr.Textbox(
            value=model_status(),   # static initial value — no polling
            label="Tagger status", interactive=False, scale=3,
        )
        load_btn   = gr.Button("Load Tagger",   variant="secondary", scale=1)
        unload_btn = gr.Button("Unload Tagger", variant="stop",      scale=1)
    load_btn.click(load_model,     outputs=status_box)
    unload_btn.click(unload_model, outputs=status_box)

    # ── Single image tab ─────────────────────────────────────────────────────
    with gr.Tab("Single Image"):
        with gr.Row():
            with gr.Column(scale=1):
                s_image   = gr.Image(type="pil", label="Input Image")
                s_mode, s_custom = threshold_controls()
                s_trigger = gr.Textbox(
                    label="Trigger word (prepended to caption)",
                    placeholder="e.g. my_lora_style",
                )
                s_btn = gr.Button("Tag Image", variant="primary")
            with gr.Column(scale=2):
                s_rating  = gr.Label(label="Rating",    num_top_classes=4)
                s_general = gr.Label(label="General",   num_top_classes=30)
                s_char    = gr.Label(label="Character", num_top_classes=20)
                s_caption = gr.Textbox(
                    label="Caption (comma-separated tags)",
                    lines=4, elem_classes=["tag-box"],
                )
        s_btn.click(
            tag_single,
            inputs=[s_image, s_mode, s_custom, s_trigger],
            outputs=[s_rating, s_general, s_char, s_caption, status_box],
        )

    # ── Batch tagging tab ────────────────────────────────────────────────────
    with gr.Tab("Batch Tagging"):
        gr.Markdown(
            "Upload multiple images — each gets a `.txt` caption file bundled into `captions.zip`."
        )
        with gr.Row():
            with gr.Column(scale=1):
                b_files   = gr.File(label="Upload Images",
                                    file_count="multiple", file_types=["image"])
                b_mode, b_custom = threshold_controls()
                b_trigger = gr.Textbox(
                    label="Trigger word (prepended to every caption)",
                    placeholder="e.g. my_lora_style",
                )
                b_btn = gr.Button("Tag All & Export ZIP", variant="primary")
            with gr.Column(scale=1):
                b_zip = gr.File(label="Download captions.zip", interactive=False)
                b_log = gr.Textbox(label="Processing Log", lines=20,
                                   elem_classes=["tag-box"], interactive=False)
        b_btn.click(
            tag_batch,
            inputs=[b_files, b_mode, b_custom, b_trigger],
            outputs=[b_zip, b_log, status_box],
        )

    # ── Image processing tab ─────────────────────────────────────────────────
    with gr.Tab("Image Processing"):
        gr.Markdown(
            "Set a **target resolution** (longest side in px). "
            "The method is chosen automatically per image:\n"
            "- **Larger than source** → Real-ESRGAN x4 (`anime_6B`) then Lanczos to exact target\n"
            "- **Smaller than source** → Lanczos downscale\n\n"
            "The Real-ESRGAN model (~17 MB) downloads automatically on first use."
        )
        with gr.Row():
            with gr.Column(scale=1):
                p_files = gr.File(label="Upload Images",
                                  file_count="multiple", file_types=["image"])

                p_target = gr.Slider(
                    minimum=64, maximum=4096, value=1024, step=64,
                    label="Target resolution — longest side (px)",
                )

                with gr.Row():
                    btn_512  = gr.Button("512")
                    btn_768  = gr.Button("768")
                    btn_1024 = gr.Button("1024")
                    btn_2048 = gr.Button("2048")

                btn_512.click(make_preset(512),  outputs=p_target)
                btn_768.click(make_preset(768),  outputs=p_target)
                btn_1024.click(make_preset(1024), outputs=p_target)
                btn_2048.click(make_preset(2048), outputs=p_target)

                gr.Markdown("---")

                p_format = gr.Radio(
                    choices=["PNG", "JPEG"], value="PNG", label="Output format"
                )
                p_jpeg_q = gr.Slider(
                    50, 100, value=95, step=1,
                    label="JPEG quality", visible=False,
                )
                p_format.change(
                    lambda f: gr.update(visible=(f == "JPEG")),
                    inputs=p_format, outputs=p_jpeg_q,
                )

                with gr.Row():
                    p_btn        = gr.Button("Process & Export ZIP", variant="primary")
                    p_unload_btn = gr.Button("Unload Upsampler", variant="stop")

                p_up_status = gr.Textbox(
                    value=upsampler_status(),   # static initial value — no polling
                    label="Upsampler status", interactive=False,
                )
                p_unload_btn.click(unload_upsampler, outputs=p_up_status)

            with gr.Column(scale=1):
                p_zip = gr.File(label="Download processed.zip", interactive=False)
                p_log = gr.Textbox(label="Processing Log", lines=20,
                                   elem_classes=["tag-box"], interactive=False)

        p_btn.click(
            process_images,
            inputs=[p_files, p_target, p_format, p_jpeg_q],
            outputs=[p_zip, p_log, p_up_status],
        )

demo.launch(css=CSS)
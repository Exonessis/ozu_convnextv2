import json
import os
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from imgutils.data import load_image
from imgutils.preprocess import create_torchvision_transforms
from timm import create_model
from PIL import Image

# ── Load model (once at startup) ──────────────────────────────────────────────
REPO_ID = "animetimm/convnextv2_huge.dbv4-full"

print("Loading model...")
model = create_model(f"hf-hub:{REPO_ID}", pretrained=True)
model.eval()

# Move to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)
print(f"Model on: {DEVICE}")

with open(hf_hub_download(repo_id=REPO_ID, repo_type="model", filename="preprocess.json")) as f:
    preprocessor = create_torchvision_transforms(json.load(f)["test"])

df_tags = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, repo_type="model", filename="selected_tags.csv"),
    keep_default_na=False,
)
print("Model ready.")

# Category thresholds from the model card
CATEGORY_THRESHOLDS = {0: 0.38, 4: 0.51, 9: 0.24}
CATEGORY_NAMES      = {0: "General", 4: "Character", 9: "Rating"}

# ── Core inference ─────────────────────────────────────────────────────────────
def run_inference(pil_image: Image.Image) -> torch.Tensor:
    image = pil_image.convert("RGB")
    input_tensor = preprocessor(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = torch.sigmoid(model(input_tensor))[0].cpu()
    return prediction


def prediction_to_dicts(prediction: torch.Tensor, threshold_mode: str, custom_threshold: float, trigger_word: str = ""):
    """Return (rating_dict, general_dict, character_dict, caption_str)."""
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
            cat_name = CATEGORY_NAMES.get(cat, f"Category {cat}")
            results[cat_name][name] = round(score, 4)

    for k in results:
        results[k] = dict(sorted(results[k].items(), key=lambda x: -x[1]))

    # Build caption string: rating first, then general, then character
    caption_parts = (
        list(results["Rating"].keys())
        + list(results["General"].keys())
        + list(results["Character"].keys())
    )
    trigger = trigger_word.strip()
    if trigger:
        caption_parts = [trigger] + caption_parts
    caption = ", ".join(caption_parts)

    return results["Rating"], results["General"], results["Character"], caption


# ── Single image tab ──────────────────────────────────────────────────────────
def tag_single(pil_image, threshold_mode, custom_threshold, trigger_word):
    if pil_image is None:
        return {}, {}, {}, ""
    pred = run_inference(pil_image)
    rating, general, character, caption = prediction_to_dicts(pred, threshold_mode, custom_threshold, trigger_word)
    return rating, general, character, caption


# ── Batch tab ─────────────────────────────────────────────────────────────────
def tag_batch(files, threshold_mode, custom_threshold, trigger_word, progress=gr.Progress()):
    if not files:
        return None, "No files uploaded."

    tmp_dir  = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "captions.zip")
    log_lines = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, file_path in enumerate(progress.tqdm(files, desc="Tagging images")):
            try:
                img_path = Path(file_path)
                pil_img  = Image.open(img_path).convert("RGB")
                pred     = run_inference(pil_img)
                _, _, _, caption = prediction_to_dicts(pred, threshold_mode, custom_threshold, trigger_word)

                txt_name = img_path.stem + ".txt"
                txt_path = os.path.join(tmp_dir, txt_name)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                zf.write(txt_path, arcname=txt_name)
                log_lines.append(f"✅ {img_path.name}  →  {txt_name}  ({len(caption.split(','))} tags)")

            except Exception as e:
                log_lines.append(f"❌ {Path(file_path).name}  —  Error: {e}")

    log = "\n".join(log_lines)
    return zip_path, log


# ── Shared threshold controls factory ─────────────────────────────────────────
def threshold_controls():
    mode = gr.Radio(
        choices=["Per-category (recommended)", "Best per-tag", "Custom"],
        value="Per-category (recommended)",
        label="Threshold mode",
    )
    custom = gr.Slider(0.0, 1.0, value=0.40, step=0.01,
                       label="Custom threshold",
                       visible=False)
    mode.change(lambda m: gr.update(visible=(m == "Custom")),
                inputs=mode, outputs=custom)
    return mode, custom


# ── UI ────────────────────────────────────────────────────────────────────────
CSS = ".tag-box { font-family: monospace; font-size: 0.85em; }"

with gr.Blocks(title="Anime Tagger — convnextv2_huge.dbv4-full") as demo:
    gr.Markdown(
        "## 🏷️ Anime Tagger — `convnextv2_huge.dbv4-full`\n"
        "Multilabel classifier with **12,476 Danbooru tags** (general · character · rating).  \n"
        "Switch between **Single** and **Batch** modes using the tabs below."
    )

    # ── Single tab ──────────────────────────────────────────────────────────
    with gr.Tab("Single Image"):
        with gr.Row():
            with gr.Column(scale=1):
                s_image  = gr.Image(type="pil", label="Input Image")
                s_mode, s_custom = threshold_controls()
                s_trigger = gr.Textbox(
                    label="Trigger word (prepended to caption)",
                    placeholder="e.g. my_lora_style",
                )
                s_btn    = gr.Button("Tag Image", variant="primary")

            with gr.Column(scale=2):
                s_rating  = gr.Label(label="Rating",    num_top_classes=4)
                s_general = gr.Label(label="General",   num_top_classes=30)
                s_char    = gr.Label(label="Character", num_top_classes=20)
                s_caption = gr.Textbox(
                    label="Caption (comma-separated tags)",
                    lines=4,
                    elem_classes=["tag-box"],
                )

        s_btn.click(
            tag_single,
            inputs=[s_image, s_mode, s_custom, s_trigger],
            outputs=[s_rating, s_general, s_char, s_caption],
        )

    # ── Batch tab ───────────────────────────────────────────────────────────
    with gr.Tab("Batch Tagging"):
        gr.Markdown(
            "Upload multiple images. Each will be tagged and saved as a `.txt` caption file.  \n"
            "All captions are bundled into a single **`captions.zip`** for download."
        )
        with gr.Row():
            with gr.Column(scale=1):
                b_files  = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"],
                )
                b_mode, b_custom = threshold_controls()
                b_trigger = gr.Textbox(
                    label="Trigger word (prepended to every caption)",
                    placeholder="e.g. my_lora_style",
                )
                b_btn    = gr.Button("Tag All & Export ZIP", variant="primary")

            with gr.Column(scale=1):
                b_zip = gr.File(label="Download captions.zip", interactive=False)
                b_log = gr.Textbox(
                    label="Processing Log",
                    lines=20,
                    elem_classes=["tag-box"],
                    interactive=False,
                )

        b_btn.click(
            tag_batch,
            inputs=[b_files, b_mode, b_custom, b_trigger],
            outputs=[b_zip, b_log],
        )

demo.launch(css=CSS)

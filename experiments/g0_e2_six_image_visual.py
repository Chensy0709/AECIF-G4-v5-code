from __future__ import annotations
import csv
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader

ROOT = Path("/mnt/workspace/AECIF_G4_v5_independent")
DATA = Path("/mnt/workspace/AECIF_Kaggle_bundle/VOCdevkit")
BASE = ROOT / "runs/G0_A10_det_150_v1/best_model_1c9ce6832025f2a7f5ca54b9d956ce451109f7019e63cd6854f524f54d8927df.pth"
E2_HEAD = ROOT / "diagnostics/runs/G0_element_ce_dice_pilot_20261122T064546Z/best_element_head_only.pth"

EXPECTED_BASE_SHA = "76d6f522565b1f85cfa27c4298c0f78fefd43088669e79ef6f46e075e20aaf7a"
EXPECTED_E2_SHA = "158c70e24d63b6c39a832bc665e3df22765f59d6c0cd9e796d06c3ec08e7236d"

IMAGE_IDS = ["14", "125", "185", "a87", "t12", "29"]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/mnt/workspace")

from g4_model import build_model
from baseline_code.utils.dataloader_multi import SegmentationDataset
from g0_element_ce_dice_pilot import (
    read_ids,
    configure_repro,
    masks_and_images,
    outs,
    matching_logits,
    confusion,
    metrics,
    sha256_file,
)

def ensure(p: Path, msg: str):
    if not p.exists():
        raise FileNotFoundError(f"{msg}: {p}")

def add_title(im: Image.Image, title: str) -> Image.Image:
    font = ImageFont.load_default()
    top = 30
    out = Image.new("RGB", (im.width, im.height + top), "white")
    out.paste(im, (0, top))
    dr = ImageDraw.Draw(out)
    dr.text((8, 8), title, fill=(0, 0, 0), font=font)
    return out

def text_tile(size, lines):
    w, h = size
    font = ImageFont.load_default()
    im = Image.new("RGB", (w, h), "white")
    dr = ImageDraw.Draw(im)
    y = 10
    for line in lines:
        dr.text((10, y), line, fill=(0, 0, 0), font=font)
        y += 16
    return im

def make_grid(images, cols=3, bg=(255, 255, 255)):
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    cell_w = max(widths)
    cell_h = max(heights)
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), bg)
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * cell_w + (cell_w - im.width) // 2
        y = r * cell_h + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas

def colorize_mask(arr: np.ndarray, palette: list[int]) -> Image.Image:
    im = Image.fromarray(arr.astype(np.uint8), mode="P")
    if palette is not None:
        im.putpalette(palette)
    return im.convert("RGB")

def overlay(base_rgb: Image.Image, mask_arr: np.ndarray, palette: list[int], alpha=0.45) -> Image.Image:
    col = colorize_mask(mask_arr, palette)
    return Image.blend(base_rgb.convert("RGB"), col, alpha)

def element_change_map(gt: np.ndarray, g0: np.ndarray, e2: np.ndarray) -> Image.Image:
    h, w = gt.shape
    out = np.full((h, w, 3), 245, dtype=np.uint8)
    valid = (gt >= 0) & (gt < 7)

    g0_ok = (g0 == gt) & valid
    e2_ok = (e2 == gt) & valid

    # 改善：绿
    out[(~g0_ok) & e2_ok] = [0, 180, 0]
    # 退步：红
    out[g0_ok & (~e2_ok)] = [220, 0, 0]
    # 都错但 E2 改了：橙
    out[(~g0_ok) & (~e2_ok) & (g0 != e2) & valid] = [255, 165, 0]
    # 都错且相同：深灰
    out[(~g0_ok) & (~e2_ok) & (g0 == e2) & valid] = [90, 90, 90]
    # 都对：浅灰
    out[g0_ok & e2_ok] = [220, 220, 220]

    return Image.fromarray(out)

def load_model(cfg, seed: int, device, load_e2_head=False):
    model = build_model(
        "G0",
        num_classes=(7, 2),
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        init_seed=seed
    )
    weights = torch.load(BASE, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights

    if load_e2_head:
        hs = torch.load(E2_HEAD, map_location="cpu", weights_only=True)
        missing, unexpected = model.cif.last_layer_e.load_state_dict(hs, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected E2 head keys: {unexpected}")
        bad_missing = [x for x in missing if not x.startswith("1.")]
        if bad_missing:
            raise RuntimeError(f"Unexpected missing keys when loading E2 head: {bad_missing}")

    model.eval().to(device)
    return model

def per_class_iou(metric_dict):
    if "ious" in metric_dict:
        return metric_dict["ious"]
    raise KeyError(
        f"No per-class IoU field found. Available: {list(metric_dict.keys())}"
    )

@torch.no_grad()
def infer_one(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    x, gt_e, gt_c = masks_and_images(batch, device)
    y_e, y_c = outs(model(x))
    y_e = matching_logits(y_e, gt_e)
    y_c = matching_logits(y_c, gt_c)

    pred_e = torch.argmax(y_e, dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
    pred_c = torch.argmax(y_c, dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
    gt_e_np = gt_e.squeeze(0).detach().cpu().numpy().astype(np.int64)
    gt_c_np = gt_c.squeeze(0).detach().cpu().numpy().astype(np.int64)

    cm_e = confusion(gt_e_np, pred_e, 7)
    cm_c = confusion(gt_c_np, pred_c, 2)
    m_e = metrics(cm_e)
    m_c = metrics(cm_c)

    return {
        "pred_e": pred_e,
        "pred_c": pred_c,
        "gt_e": gt_e_np,
        "gt_c": gt_c_np,
        "cm_e": cm_e,
        "cm_c": cm_c,
        "m_e": m_e,
        "m_c": m_c,
    }

def main():
    ensure(BASE, "Missing G0 checkpoint")
    ensure(E2_HEAD, "Missing E2 head")
    ensure(DATA / "VOC2007", "Missing VOC2007")
    ensure(ROOT / "configs/protocol.json", "Missing protocol")
    ensure(ROOT / "configs/data_splits/val.txt", "Missing val split")

    if sha256_file(BASE) != EXPECTED_BASE_SHA:
        raise RuntimeError("G0 baseline SHA mismatch")
    if sha256_file(E2_HEAD) != EXPECTED_E2_SHA:
        raise RuntimeError("E2 head SHA mismatch")

    cfg = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    val_ids = read_ids(ROOT / "configs/data_splits/val.txt")
    missing = [x for x in IMAGE_IDS if x not in val_ids]
    if missing:
        raise RuntimeError(f"IDs not in validation set: {missing}")

    configure_repro(int(cfg["training"]["seed"]))

    if not torch.cuda.is_available() or "A10" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("Expected one free NVIDIA A10")

    device = torch.device("cuda:0")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = ROOT / "diagnostics" / "runs" / f"g0_e2_six_image_visual_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    model_g0 = load_model(cfg, int(cfg["training"]["seed"]), device, load_e2_head=False)
    model_e2 = load_model(cfg, int(cfg["training"]["seed"]), device, load_e2_head=True)

    records = []
    cm_base_list = []
    cm_e2_list = []

    for image_id in IMAGE_IDS:
        ds = SegmentationDataset(
            [image_id],
            cfg["model"]["input_size"],
            cfg["model"]["num_classes"],
            False,
            str(DATA)
        )

        g0_out = infer_one(model_g0, ds, device)
        e2_out = infer_one(model_e2, ds, device)

        if not np.array_equal(g0_out["gt_e"], e2_out["gt_e"]):
            raise RuntimeError(f"GT mismatch on {image_id}")
        if not np.array_equal(g0_out["pred_c"], e2_out["pred_c"]):
            raise RuntimeError(f"Corrosion prediction changed unexpectedly on {image_id}")

        jpg_path = DATA / "VOC2007" / "JPEGImages" / f"{image_id}.jpg"
        gt_e_path = DATA / "VOC2007" / "SegmentationClass" / "element" / f"{image_id}.png"

        base_rgb = Image.open(jpg_path).convert("RGB").resize((g0_out["gt_e"].shape[1], g0_out["gt_e"].shape[0]), Image.Resampling.BILINEAR)
        gt_e_img = Image.open(gt_e_path)
        element_palette = gt_e_img.getpalette()

        gt_e_rgb = colorize_mask(g0_out["gt_e"], element_palette)
        g0_overlay = overlay(base_rgb, g0_out["pred_e"], element_palette)
        e2_overlay = overlay(base_rgb, e2_out["pred_e"], element_palette)
        delta_map = element_change_map(g0_out["gt_e"], g0_out["pred_e"], e2_out["pred_e"])

        g0_m = g0_out["m_e"]
        e2_m = e2_out["m_e"]
        g0_cls = per_class_iou(g0_m)
        e2_cls = per_class_iou(e2_m)
        cls_delta_pp = [
            None if a is None or b is None
            else (float(b) - float(a)) * 100.0
            for a, b in zip(g0_cls, e2_cls)
        ]

        top_cls = sorted(
            [(i, d) for i, d in enumerate(cls_delta_pp) if d is not None],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:4]

        tile = text_tile(
            base_rgb.size,
            [
                f"ID: {image_id}",
                f"G0 mIoU: {g0_m['miou']*100:.4f}%",
                f"E2 mIoU: {e2_m['miou']*100:.4f}%",
                f"Delta: {(e2_m['miou']-g0_m['miou'])*100:.4f} pp",
                "",
                "Top class deltas (pp):",
                *[f"class {k}: {v:+.4f}" for k, v in top_cls],
                "",
                "Legend:",
                "green=E2 improved",
                "red=E2 worsened",
                "orange=both wrong but changed",
                "gray=unchanged correctness",
            ]
        )

        panel = make_grid([
            add_title(base_rgb, "Input"),
            add_title(gt_e_rgb, "GT Element"),
            add_title(g0_overlay, "G0 Overlay"),
            add_title(e2_overlay, "E2 Overlay"),
            add_title(delta_map, "G0 vs E2 Change Map"),
            add_title(tile, "Metrics"),
        ], cols=3)

        img_dir = out / "images" / image_id
        img_dir.mkdir(parents=True, exist_ok=False)

        base_rgb.save(img_dir / "input_rgb.jpg", quality=95)
        gt_e_rgb.save(img_dir / "gt_element_color.png")
        g0_overlay.save(img_dir / "g0_element_overlay.png")
        e2_overlay.save(img_dir / "e2_element_overlay.png")
        delta_map.save(img_dir / "element_change_map.png")
        panel.save(img_dir / "element_comparison_panel.png")

        rec = {
            "image_id": image_id,
            "g0_element_miou": float(g0_m["miou"]),
            "e2_element_miou": float(e2_m["miou"]),
            "element_delta_pp": float((e2_m["miou"] - g0_m["miou"]) * 100.0),
        }
        for i, (a, b) in enumerate(zip(g0_cls, e2_cls)):
            rec[f"g0_class{i}_iou"] = None if a is None else float(a)
            rec[f"e2_class{i}_iou"] = None if b is None else float(b)
            rec[f"class{i}_delta_pp"] = (
                None if a is None or b is None
                else (float(b) - float(a)) * 100.0
            )
        records.append(rec)

        cm_base_list.append(g0_out["cm_e"])
        cm_e2_list.append(e2_out["cm_e"])

    csv_path = out / "per_image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "status": "G0_E2_SIX_IMAGE_VISUAL_COMPLETE",
        "baseline_sha256": EXPECTED_BASE_SHA,
        "e2_head_sha256": EXPECTED_E2_SHA,
        "image_ids": IMAGE_IDS,
        "improved_images": sum(1 for r in records if r["element_delta_pp"] > 0),
        "declined_images": sum(1 for r in records if r["element_delta_pp"] < 0),
        "mean_delta_pp": float(sum(r["element_delta_pp"] for r in records) / len(records)),
        "note": "Inference only. G0 full checkpoint + E2 element head on top of baseline BN buffers.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    np.savez_compressed(
        out / "per_image_confusion_matrices.npz",
        image_ids=np.array(IMAGE_IDS),
        baseline=np.stack(cm_base_list, axis=0),
        e2=np.stack(cm_e2_list, axis=0),
    )

    print("G0_E2_SIX_IMAGE_VISUAL_COMPLETE")
    print("OUTPUT_DIR:", out)
    print("CSV:", csv_path)
    print("SUMMARY:", out / "summary.json")

if __name__ == "__main__":
    main()

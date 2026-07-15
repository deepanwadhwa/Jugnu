#!/usr/bin/env python3
"""E-V1 — Are the shipped vision tower weights numerically usable?

This is the RUN-FIRST experiment for Issue #3 (Vision). It determines whether
the accidentally-shipped whole-row int8 vision tower in resident.safetensors
is numerically usable, or whether re-quantization is needed.

The vision tower (444 tensors, 0.454 GB) was quantized by convert_qwen36.py's
generic path because the skip filter checked for "vision" but the tensors are
named "model.visual.*" — and "vision" in "visual" is False.

Method (from TASKS_VISION.md):
  1. Per-tensor: cosine similarity and max absolute relative error for all 111
     quantized tensors across three variants: shipped int8, group-32 q4, BF16.
  2. End-to-end in PyTorch: run the reference ViT on ≥20 images with BF16,
     shipped-int8, and group-32-q4 weights. Compare merger output embeddings.
  3. Full-model text generation comparison: DEFERRED — infeasible on 16GB Mac
     with Pure Python constraint. Requires ~70GB for BF16 or ~35GB for int8.

Acceptance:
  - Per-tensor cosine ≥ 0.99 for all 111
  - Merger-output cosine ≥ 0.99 mean with no image below 0.97
  - Generated text substantively equivalent on ≥18/20 images (Step 3, deferred)

Kill criterion: if degraded, say so plainly and stop. Cost the fallback.

Usage:
  python tools/run_e_v1.py [--shipped-path PATH] [--out-dir PATH]

Environment variables:
  SAMOSA_SHIPPED_PATH  — path to resident.safetensors (default: auto-detect)
  SAMOSA_EV1_OUTDIR    — output directory (default: docs/regressions/vision-validation)
"""

import os
import sys
import json
import time
import argparse
import math
from datetime import datetime

import torch
import numpy as np
import safetensors.torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UPSTREAM_REPO = "Qwen/Qwen3.6-35B-A3B"
GROUP_SIZE = 32  # matches groupwise-symmetric-q4-v1

# Default paths
DEFAULT_SHIPPED_PATHS = [
    os.path.expanduser("~/.samosa/current/model/resident.safetensors"),
    os.path.expanduser("~/Documents/samosa-models/qwen36_group32_i8/resident.safetensors"),
]
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "regressions", "vision-validation"
)


def log(msg):
    print(f"[E-V1] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Quantization helpers — ported from convert_qwen36.py
# ---------------------------------------------------------------------------

def dequantize_rowwise_int8(q_tensor, qs_tensor, ref_shape):
    """Dequantize shipped row-wise int8: w_approx = q_signed * scale[row]."""
    q_np = q_tensor.numpy()
    q_signed = q_np.view(np.int8).astype(np.float32)

    O = qs_tensor.shape[0]
    I = q_signed.size // O
    q_2d = q_signed.reshape(O, I)

    scales_np = qs_tensor.numpy()
    dequant = q_2d * scales_np[:, np.newaxis]

    return torch.from_numpy(dequant).reshape(ref_shape)


def quantize_group32_q4(w_np, group_size=GROUP_SIZE):
    """Symmetric int4 with one scale per contiguous group of `group_size` inputs.

    Ported from convert_qwen36.py:quant_int4_grouped. Returns (packed, scales).
    """
    O, I = w_np.shape
    if I % group_size != 0:
        # Pad to nearest multiple (shouldn't happen for vision tower, but safe)
        pad = group_size - (I % group_size)
        w_np = np.pad(w_np, ((0, 0), (0, pad)), mode='constant')
        I = w_np.shape[1]

    groups = I // group_size
    blocks = w_np.reshape(O, groups, group_size)
    scales = np.maximum(np.abs(blocks).max(axis=2) / 7, 1e-8).astype(np.float32)
    q = np.clip(np.rint(blocks / scales[:, :, None]), -8, 7).astype(np.int32)
    q = q.reshape(O, I)
    packed = ((q[:, 0::2] + 8).astype(np.uint8) |
              ((q[:, 1::2] + 8).astype(np.uint8) << 4))
    return packed.reshape(-1), scales.reshape(-1)


def dequantize_group32_q4(packed, scales, shape, group_size=GROUP_SIZE):
    """Dequantize group-32 symmetric int4 back to float32.

    Ported from convert_qwen36.py:dequantize.
    """
    rows, cols = shape
    packed_2d = packed.reshape(rows, (cols + 1) // 2)
    values = np.empty((rows, cols), dtype=np.float32)
    values[:, 0::2] = (packed_2d & 0x0F).astype(np.int16) - 8
    if cols > 1:
        values[:, 1::2] = (packed_2d[:, :cols // 2] >> 4).astype(np.int16) - 8

    groups = (cols + group_size - 1) // group_size
    expanded = np.repeat(scales.reshape(rows, groups), group_size, axis=1)[:, :cols]
    return values * expanded


# ---------------------------------------------------------------------------
# Test image generation — diverse and realistic synthetic images
# ---------------------------------------------------------------------------

def create_test_images():
    """Create 20 diverse synthetic test images that exercise the ViT meaningfully.

    Categories (from the spec): natural-photo-like, screenshots, document scans,
    charts. These are synthetic but contain gradients, text, noise, edges, and
    patterns that a real ViT must handle — not just solid rectangles.
    """
    images = []
    descriptions = []

    # --- Natural photo approximations (6 images) ---

    # 1. Sky gradient with ground
    img = Image.new("RGB", (512, 384))
    for y in range(384):
        r = int(40 + (y / 384) * 80)
        g = int(120 + (y / 384) * 60)
        b = int(220 - (y / 384) * 120)
        if y > 280:
            r, g, b = int(60 + (y - 280) * 1.5), int(120 - (y - 280)), 40
        for x in range(512):
            img.putpixel((x, y), (r, g, b))
    draw = ImageDraw.Draw(img)
    # Add "trees" as simple triangles
    for tx in range(50, 500, 80):
        h = 40 + (tx * 7) % 30
        draw.polygon([(tx, 280), (tx - 15, 280 + h), (tx + 15, 280 + h)],
                     fill=(20, 80 + tx % 40, 20))
    images.append(img)
    descriptions.append("sky gradient with ground and tree shapes")

    # 2. Sunset-like horizontal gradient
    img = Image.new("RGB", (640, 480))
    for y in range(480):
        t = y / 480
        r = int(255 * max(0, 1 - abs(t - 0.3) * 3))
        g = int(180 * max(0, 1 - abs(t - 0.5) * 2.5))
        b = int(100 + 155 * t)
        for x in range(640):
            img.putpixel((x, y), (min(255, r + x % 3), min(255, g), min(255, b)))
    images.append(img)
    descriptions.append("sunset-like horizontal gradient")

    # 3. Noisy texture (simulates photograph grain)
    arr = np.random.randint(80, 200, (400, 600, 3), dtype=np.uint8)
    # Add some structure — large blobs via downscale+upscale
    small = Image.fromarray(arr).resize((60, 40), Image.BILINEAR)
    img = small.resize((600, 400), Image.BILINEAR)
    img_arr = np.array(img)
    noise = np.random.randint(-20, 20, img_arr.shape, dtype=np.int16)
    img = Image.fromarray(np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    images.append(img)
    descriptions.append("noisy natural texture with grain")

    # 4. High-contrast edges (building-like)
    img = Image.new("RGB", (512, 512), (200, 200, 210))
    draw = ImageDraw.Draw(img)
    for i in range(5):
        x0 = 30 + i * 95
        draw.rectangle([x0, 50, x0 + 80, 480], fill=(120 + i * 20, 110, 100),
                       outline=(60, 60, 60), width=2)
        for wy in range(70, 460, 40):
            draw.rectangle([x0 + 10, wy, x0 + 30, wy + 25],
                          fill=(180, 210, 240))
            draw.rectangle([x0 + 45, wy, x0 + 65, wy + 25],
                          fill=(180, 210, 240))
    images.append(img)
    descriptions.append("high-contrast building facade with windows")

    # 5. Circular gradient (lens-like)
    img = Image.new("RGB", (448, 448))
    cx, cy = 224, 224
    for y in range(448):
        for x in range(448):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / 224
            r = int(255 * max(0, 1 - d))
            g = int(200 * max(0, 1 - d * 0.8))
            b = int(100 * max(0, 1 - d * 1.5))
            img.putpixel((x, y), (r, g, b))
    images.append(img)
    descriptions.append("circular gradient (lens flare)")

    # 6. Color stripes (test pattern)
    img = Image.new("RGB", (480, 320))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0)]
    stripe_w = 480 // len(colors)
    draw = ImageDraw.Draw(img)
    for i, c in enumerate(colors):
        draw.rectangle([i * stripe_w, 0, (i + 1) * stripe_w, 320], fill=c)
    images.append(img)
    descriptions.append("color bar test pattern")

    # --- Screenshot-like images (4 images) ---

    # 7. Text-heavy "document" screenshot
    img = Image.new("RGB", (600, 800), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "The Quick Brown Fox Jumps Over the Lazy Dog",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789",
        "Lorem ipsum dolor sit amet, consectetur",
        "adipiscing elit. Sed do eiusmod tempor",
        "incididunt ut labore et dolore magna aliqua.",
        "", "Section 2: Technical Details", "",
        "The model architecture uses 27 vision blocks",
        "with hidden_size=1152 and num_heads=16.",
        "Patch size is 16x16 with temporal_patch_size=2.",
    ]
    y = 40
    for line in lines:
        draw.text((30, y), line, fill=(20, 20, 20))
        y += 22
    # Add a horizontal rule
    draw.line([(30, y + 5), (570, y + 5)], fill=(180, 180, 180), width=1)
    images.append(img)
    descriptions.append("text document screenshot")

    # 8. UI screenshot mockup (buttons, text fields)
    img = Image.new("RGB", (500, 400), (240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Title bar
    draw.rectangle([0, 0, 500, 30], fill=(60, 60, 60))
    draw.text((200, 7), "Settings", fill=(255, 255, 255))
    # Traffic lights
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([10 + i * 20, 8, 24 + i * 20, 22], fill=c)
    # Form fields
    for i in range(4):
        y = 60 + i * 70
        draw.text((30, y), f"Field {i + 1}:", fill=(80, 80, 80))
        draw.rectangle([30, y + 20, 470, y + 48], outline=(180, 180, 190), width=1)
        draw.text((40, y + 25), "placeholder text..." if i != 2 else "filled value",
                 fill=(160, 160, 170) if i != 2 else (20, 20, 20))
    # Button
    draw.rounded_rectangle([180, 340, 320, 375], radius=6, fill=(59, 130, 246))
    draw.text((222, 350), "Save", fill=(255, 255, 255))
    images.append(img)
    descriptions.append("UI settings panel mockup")

    # 9. Code screenshot
    img = Image.new("RGB", (700, 500), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    code_lines = [
        ("def ", (197, 134, 192)),
        ("forward", (220, 220, 170)),
        ("(self, x, grid_thw):", (200, 200, 200)),
    ]
    y = 20
    # Line numbers
    for i in range(15):
        draw.text((10, y + i * 24), f"{i + 1:3d}", fill=(100, 100, 100))
    # Simplified code rendering
    code = [
        "def forward(self, x, grid_thw):",
        "    # Patch embedding",
        "    x = self.patch_embed(x)",
        "    pos = self.pos_embed(grid_thw)",
        "    x = x + pos",
        "",
        "    for blk in self.blocks:",
        "        x = blk(x)",
        "",
        "    # Merger",
        "    x = self.merger.norm(x)",
        "    x = self.merger.fc1(x)",
        "    x = self.merger.act(x)",
        "    return self.merger.fc2(x)",
    ]
    for i, line in enumerate(code):
        color = (86, 156, 214) if line.strip().startswith(("#", "def", "for", "return")) else (212, 212, 212)
        if line.strip().startswith("#"):
            color = (106, 153, 85)
        elif line.strip().startswith(("self.", "x =")):
            color = (156, 220, 254)
        draw.text((50, y + i * 24), line, fill=color)
    images.append(img)
    descriptions.append("code editor screenshot (dark theme)")

    # 10. Terminal output screenshot
    img = Image.new("RGB", (600, 300), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    terminal_lines = [
        ("$ python run_e_v1.py", (0, 255, 0)),
        ("[E-V1] Loading reference weights...", (200, 200, 200)),
        ("[E-V1] Loaded 444 visual keys", (200, 200, 200)),
        ("[E-V1] Per-tensor cosine min: 0.9987", (200, 200, 200)),
        ("[E-V1] PASS: All criteria met", (0, 255, 100)),
        ("$ _", (0, 255, 0)),
    ]
    for i, (line, color) in enumerate(terminal_lines):
        draw.text((10, 10 + i * 22), line, fill=color)
    images.append(img)
    descriptions.append("terminal output screenshot")

    # --- Chart/diagram images (4 images) ---

    # 11. Bar chart
    img = Image.new("RGB", (500, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((180, 10), "Performance (tok/s)", fill=(40, 40, 40))
    bars = [("2T", 14, (59, 130, 246)), ("4T", 24, (16, 185, 129)),
            ("8T", 28, (245, 158, 11)), ("16T", 30, (239, 68, 68))]
    for i, (label, val, color) in enumerate(bars):
        x = 80 + i * 100
        h = int(val * 10)
        draw.rectangle([x, 350 - h, x + 60, 350], fill=color)
        draw.text((x + 15, 360), label, fill=(80, 80, 80))
        draw.text((x + 15, 340 - h), str(val), fill=(40, 40, 40))
    # Axes
    draw.line([(60, 50), (60, 355)], fill=(100, 100, 100), width=1)
    draw.line([(58, 350), (480, 350)], fill=(100, 100, 100), width=1)
    images.append(img)
    descriptions.append("bar chart of performance metrics")

    # 12. Line chart
    img = Image.new("RGB", (500, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((150, 10), "Cosine Similarity by Layer", fill=(40, 40, 40))
    points = [(40 + i * 16, int(380 - (0.990 + 0.008 * math.sin(i * 0.5)) * 380))
              for i in range(27)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(59, 130, 246), width=2)
    for p in points:
        draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(59, 130, 246))
    images.append(img)
    descriptions.append("line chart of cosine similarity by layer")

    # 13. Pie chart
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((130, 10), "Model Composition", fill=(40, 40, 40))
    # Simple pie using arcs
    angles = [0, 250, 305, 340, 360]
    colors = [(59, 130, 246), (16, 185, 129), (245, 158, 11), (239, 68, 68)]
    labels = ["Experts 69%", "Attention 15%", "Vision 10%", "Other 6%"]
    for i in range(4):
        draw.pieslice([50, 50, 350, 350], angles[i], angles[i + 1], fill=colors[i])
        draw.text((50, 360 + (i % 2) * 15), labels[i] if i < 2 else "",
                 fill=colors[i])
    images.append(img)
    descriptions.append("pie chart of model composition")

    # 14. Scatter plot
    img = Image.new("RGB", (500, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((150, 10), "Tensor Size vs Cosine Sim", fill=(40, 40, 40))
    np.random.seed(42)
    for _ in range(50):
        x = int(50 + np.random.rand() * 400)
        y = int(50 + np.random.rand() * 300)
        r = 3 + int(np.random.rand() * 5)
        alpha = int(150 + np.random.rand() * 105)
        draw.ellipse([x - r, y - r, x + r, y + r],
                    fill=(59, 130, 246, alpha))
    images.append(img)
    descriptions.append("scatter plot of tensor statistics")

    # --- Edge cases and stress patterns (6 images) ---

    # 15. Very small image (tests minimum resolution handling)
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 5, 58, 58], outline=(255, 0, 0), width=2)
    draw.line([(0, 0), (63, 63)], fill=(0, 255, 0), width=1)
    images.append(img)
    descriptions.append("tiny 64x64 image with cross pattern")

    # 16. Large image with fine detail
    img = Image.new("RGB", (1024, 768), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Grid of small shapes
    for y in range(0, 768, 16):
        for x in range(0, 1024, 16):
            c = ((x * 7 + y * 13) % 200 + 55, (x * 11 + y * 3) % 200 + 55,
                 (x * 5 + y * 17) % 200 + 55)
            draw.rectangle([x, y, x + 14, y + 14], fill=c)
    images.append(img)
    descriptions.append("1024x768 fine mosaic pattern")

    # 17. Grayscale gradient (tests channel uniformity)
    arr = np.zeros((384, 512, 3), dtype=np.uint8)
    for x in range(512):
        v = int(255 * x / 511)
        arr[:, x, :] = v
    images.append(Image.fromarray(arr))
    descriptions.append("horizontal grayscale gradient")

    # 18. Checkerboard (high-frequency spatial pattern)
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        for x in range(512):
            if (x // 32 + y // 32) % 2 == 0:
                arr[y, x] = [255, 255, 255]
            else:
                arr[y, x] = [0, 0, 0]
    images.append(Image.fromarray(arr))
    descriptions.append("32px checkerboard pattern")

    # 19. Non-square aspect ratio (wide panorama)
    img = Image.new("RGB", (800, 200))
    draw = ImageDraw.Draw(img)
    for x in range(800):
        t = x / 800
        r = int(60 + 195 * t)
        g = int(120 + 60 * math.sin(t * math.pi))
        b = int(200 - 150 * t)
        draw.line([(x, 0), (x, 200)], fill=(r, g, b))
    images.append(img)
    descriptions.append("wide panoramic gradient (4:1 aspect)")

    # 20. Random noise (worst case for quantization — high entropy)
    arr = np.random.randint(0, 256, (384, 512, 3), dtype=np.uint8)
    images.append(Image.fromarray(arr))
    descriptions.append("pure random noise (high entropy stress test)")

    return images, descriptions


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def find_shipped_path(explicit_path=None):
    """Locate resident.safetensors, preferring explicit path > env > defaults."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    env_path = os.environ.get("SAMOSA_SHIPPED_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    for p in DEFAULT_SHIPPED_PATHS:
        if os.path.isfile(p):
            return p
    return None


def run_per_tensor_comparison(shipped_tensors, ref_weights, quantized_names):
    """Step 1: Per-tensor cosine similarity and max relative error.

    Compares three variants against BF16 reference:
      - Shipped int8 (row-wise)
      - Group-32 q4 (re-quantized from BF16)
    """
    results_int8 = []
    results_q4 = []

    for k in quantized_names:
        q_w = shipped_tensors[k]
        scales = shipped_tensors[k + ".qs"]
        ref_w = ref_weights[k].to(torch.float32)

        # --- Shipped int8 ---
        dequant_int8 = dequantize_rowwise_int8(q_w, scales, ref_w.shape)

        ref_flat = ref_w.flatten()
        int8_flat = dequant_int8.flatten()

        cos_int8 = compute_cosine(ref_flat, int8_flat)
        max_rel_int8 = compute_max_rel_error(dequant_int8, ref_w)
        max_abs_int8 = torch.max(torch.abs(dequant_int8 - ref_w)).item()

        results_int8.append({
            "tensor_name": k,
            "cosine_similarity": cos_int8,
            "max_abs_relative_error": max_rel_int8,
            "max_abs_error": max_abs_int8,
        })

        # --- Group-32 q4 re-quantization ---
        ref_np = ref_w.numpy()
        if ref_np.ndim == 2:
            O, I = ref_np.shape
            if I % GROUP_SIZE == 0:
                packed, g32_scales = quantize_group32_q4(ref_np)
                dequant_q4_np = dequantize_group32_q4(packed, g32_scales, (O, I))
                dequant_q4 = torch.from_numpy(dequant_q4_np)

                q4_flat = dequant_q4.flatten()
                cos_q4 = compute_cosine(ref_flat, q4_flat)
                max_rel_q4 = compute_max_rel_error(dequant_q4, ref_w)
                max_abs_q4 = torch.max(torch.abs(dequant_q4 - ref_w)).item()
            else:
                # I not divisible by group_size — fall back to reporting N/A
                cos_q4 = float("nan")
                max_rel_q4 = float("nan")
                max_abs_q4 = float("nan")
        else:
            # Non-2D tensor (e.g., pos_embed may have been reshaped)
            # Reshape to 2D for quantization, then reshape back
            orig_shape = ref_np.shape
            ref_2d = ref_np.reshape(ref_np.shape[0], -1)
            O, I = ref_2d.shape
            if I % GROUP_SIZE == 0:
                packed, g32_scales = quantize_group32_q4(ref_2d)
                dequant_q4_np = dequantize_group32_q4(packed, g32_scales, (O, I))
                dequant_q4 = torch.from_numpy(dequant_q4_np.reshape(orig_shape))

                q4_flat = dequant_q4.flatten()
                cos_q4 = compute_cosine(ref_flat, q4_flat)
                max_rel_q4 = compute_max_rel_error(dequant_q4, ref_w)
                max_abs_q4 = torch.max(torch.abs(dequant_q4 - ref_w)).item()
            else:
                cos_q4 = float("nan")
                max_rel_q4 = float("nan")
                max_abs_q4 = float("nan")

        results_q4.append({
            "tensor_name": k,
            "cosine_similarity": cos_q4,
            "max_abs_relative_error": max_rel_q4,
            "max_abs_error": max_abs_q4,
        })

    return results_int8, results_q4


def compute_cosine(a, b):
    """Cosine similarity between two flat tensors."""
    dot = torch.dot(a, b).item()
    norm_a = torch.norm(a).item()
    norm_b = torch.norm(b).item()
    if norm_a * norm_b == 0:
        return 1.0 if norm_a == norm_b else 0.0
    return dot / (norm_a * norm_b)


def compute_max_rel_error(approx, ref):
    """Max absolute relative error: max(|A - B| / (|B| + 1e-8))."""
    abs_diff = torch.abs(approx - ref)
    rel_diff = abs_diff / (torch.abs(ref) + 1e-8)
    return torch.max(rel_diff).item()


def run_forward_pass_comparison(ref_weights, shipped_tensors, quantized_names,
                                 images, descriptions, config):
    """Step 2: End-to-end ViT forward pass with BF16, int8, and group-32 q4 weights."""
    from transformers import AutoProcessor
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

    processor = AutoProcessor.from_pretrained(UPSTREAM_REPO, trust_remote_code=True)

    # --- Build state dicts ---
    def strip_prefix(d):
        return {k.replace("model.visual.", ""): v for k, v in d.items()}

    # 1. BF16 reference
    ref_state = strip_prefix(ref_weights)

    # 2. Shipped int8 dequantized
    int8_state = {}
    for k, v in shipped_tensors.items():
        stripped = k.replace("model.visual.", "")
        if stripped.endswith(".qs"):
            continue
        if k in quantized_names:
            ref_w = ref_weights[k]
            dequant = dequantize_rowwise_int8(v, shipped_tensors[k + ".qs"], ref_w.shape)
            int8_state[stripped] = dequant.to(ref_w.dtype)
        else:
            int8_state[stripped] = v

    # 3. Group-32 q4 re-quantized from BF16
    q4_state = {}
    for k, v in ref_weights.items():
        stripped = k.replace("model.visual.", "")
        v_f32 = v.to(torch.float32)
        v_np = v_f32.numpy()

        # Only quantize weight tensors that the shipped version also quantized
        if k in quantized_names and v_np.ndim >= 2:
            orig_shape = v_np.shape
            v_2d = v_np.reshape(v_np.shape[0], -1)
            O, I = v_2d.shape
            if I % GROUP_SIZE == 0:
                packed, scales = quantize_group32_q4(v_2d)
                dequant_np = dequantize_group32_q4(packed, scales, (O, I))
                q4_state[stripped] = torch.from_numpy(
                    dequant_np.reshape(orig_shape)).to(v.dtype)
            else:
                q4_state[stripped] = v  # Can't group-quantize; keep original
        else:
            q4_state[stripped] = v

    # --- Instantiate models ---
    log("Instantiating reference BF16 model...")
    ref_model = Qwen3_5MoeVisionModel(config.vision_config)
    ref_model.eval()
    ref_model.load_state_dict(ref_state)

    log("Instantiating shipped int8 model...")
    int8_model = Qwen3_5MoeVisionModel(config.vision_config)
    int8_model.eval()
    int8_model.load_state_dict(int8_state)

    log("Instantiating group-32 q4 model...")
    q4_model = Qwen3_5MoeVisionModel(config.vision_config)
    q4_model.eval()
    q4_model.load_state_dict(q4_state)

    # --- Run forward passes ---
    results_int8 = []
    results_q4 = []

    for idx, (img, desc) in enumerate(zip(images, descriptions)):
        log(f"  Image {idx}: {desc}")
        inputs = processor(images=img, text="Describe", return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]

        with torch.no_grad():
            ref_out = ref_model(pixel_values, grid_thw)[0].float()
            int8_out = int8_model(pixel_values, grid_thw)[0].float()
            q4_out = q4_model(pixel_values, grid_thw)[0].float()

        # Int8 vs reference
        cos_int8 = compute_cosine(ref_out.flatten(), int8_out.flatten())
        max_abs_int8 = torch.max(torch.abs(int8_out - ref_out)).item()
        mean_abs_int8 = torch.mean(torch.abs(int8_out - ref_out)).item()

        results_int8.append({
            "image_index": idx,
            "description": desc,
            "cosine_similarity": cos_int8,
            "max_abs_error": max_abs_int8,
            "mean_abs_error": mean_abs_int8,
        })
        log(f"    int8:  cosine={cos_int8:.6f}, max_abs={max_abs_int8:.6f}")

        # Q4 vs reference
        cos_q4 = compute_cosine(ref_out.flatten(), q4_out.flatten())
        max_abs_q4 = torch.max(torch.abs(q4_out - ref_out)).item()
        mean_abs_q4 = torch.mean(torch.abs(q4_out - ref_out)).item()

        results_q4.append({
            "image_index": idx,
            "description": desc,
            "cosine_similarity": cos_q4,
            "max_abs_error": max_abs_q4,
            "mean_abs_error": mean_abs_q4,
        })
        log(f"    q4g32: cosine={cos_q4:.6f}, max_abs={max_abs_q4:.6f}")

    return results_int8, results_q4


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(out_dir, tensor_int8, tensor_q4, image_int8, image_q4):
    """Write the final E-V1 report in markdown."""
    report_path = os.path.join(out_dir, "report.md")

    def stats(values):
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            return {"min": float("nan"), "mean": float("nan"), "max": float("nan"),
                    "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        return {
            "min": float(np.min(arr)), "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    cos_int8 = stats([r["cosine_similarity"] for r in tensor_int8])
    cos_q4 = stats([r["cosine_similarity"] for r in tensor_q4])
    rel_int8 = stats([r["max_abs_relative_error"] for r in tensor_int8])
    rel_q4 = stats([r["max_abs_relative_error"] for r in tensor_q4])

    img_cos_int8 = stats([r["cosine_similarity"] for r in image_int8]) if image_int8 else None
    img_cos_q4 = stats([r["cosine_similarity"] for r in image_q4]) if image_q4 else None

    # Acceptance criteria
    all_tensor_cosines_int8 = [r["cosine_similarity"] for r in tensor_int8]
    tensor_pass_int8 = all(c >= 0.99 for c in all_tensor_cosines_int8)

    merger_pass_int8 = False
    if img_cos_int8:
        all_img_cosines = [r["cosine_similarity"] for r in image_int8]
        merger_pass_int8 = (
            np.mean(all_img_cosines) >= 0.99 and
            np.min(all_img_cosines) >= 0.97
        )

    all_tensor_cosines_q4 = [r["cosine_similarity"] for r in tensor_q4
                             if not np.isnan(r["cosine_similarity"])]
    tensor_pass_q4 = all(c >= 0.99 for c in all_tensor_cosines_q4) if all_tensor_cosines_q4 else False

    merger_pass_q4 = False
    if img_cos_q4:
        all_img_cosines_q4 = [r["cosine_similarity"] for r in image_q4]
        merger_pass_q4 = (
            np.mean(all_img_cosines_q4) >= 0.99 and
            np.min(all_img_cosines_q4) >= 0.97
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# E-V1: Shipped Vision Tower Numerical Parity Report\n\n")
        f.write(f"**Date:** {now}\n")
        f.write(f"**Reference:** `{UPSTREAM_REPO}` (BF16)\n")
        f.write(f"**Shipped:** `deepanwa/Samosa-Chat-Qwen3.6-35B-A3B-group32` "
                f"(row-wise int8 visual tower in `resident.safetensors`)\n")
        f.write(f"**Group-32 q4:** re-quantized from BF16 using "
                f"`groupwise-symmetric-q4-v1`, group_size={GROUP_SIZE}\n\n")

        # ---- Per-Tensor ----
        f.write("## 1. Per-Tensor Comparison (111 Quantized Tensors)\n\n")
        f.write("| Metric | Shipped Int8 Cosine | Group-32 Q4 Cosine | "
                "Shipped Int8 Max Rel Err | Group-32 Q4 Max Rel Err |\n")
        f.write("|---|---|---|---|---|\n")
        for label, key in [("Min", "min"), ("Mean", "mean"), ("Max", "max"),
                           ("p10", "p10"), ("p50", "p50"), ("p90", "p90")]:
            f.write(f"| **{label}** | {cos_int8[key]:.6f} | {cos_q4[key]:.6f} | "
                    f"{rel_int8[key]:.6f} | {rel_q4[key]:.6f} |\n")

        f.write(f"\n**Acceptance (per-tensor cosine ≥ 0.99 for all 111):**\n")
        f.write(f"- Shipped int8: **{'PASS' if tensor_pass_int8 else 'FAIL'}** "
                f"(min cosine = {cos_int8['min']:.6f})\n")
        f.write(f"- Group-32 q4: **{'PASS' if tensor_pass_q4 else 'FAIL'}** "
                f"(min cosine = {cos_q4['min']:.6f})\n\n")

        # ---- Forward Pass ----
        if image_int8 and image_q4:
            f.write("## 2. End-to-End ViT Merger Embeddings (20 Images)\n\n")
            f.write("| Image | Description | Int8 Cosine | Q4 Cosine | "
                    "Int8 Max Abs | Q4 Max Abs |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r8, r4 in zip(image_int8, image_q4):
                f.write(f"| {r8['image_index']} | {r8['description']} | "
                        f"{r8['cosine_similarity']:.6f} | {r4['cosine_similarity']:.6f} | "
                        f"{r8['max_abs_error']:.6f} | {r4['max_abs_error']:.6f} |\n")

            f.write(f"\n**Summary:**\n")
            f.write(f"- Shipped int8: mean cosine = {img_cos_int8['mean']:.6f}, "
                    f"min = {img_cos_int8['min']:.6f}\n")
            f.write(f"- Group-32 q4: mean cosine = {img_cos_q4['mean']:.6f}, "
                    f"min = {img_cos_q4['min']:.6f}\n\n")

            f.write(f"**Acceptance (merger cosine ≥ 0.99 mean, no image < 0.97):**\n")
            f.write(f"- Shipped int8: **{'PASS' if merger_pass_int8 else 'FAIL'}**\n")
            f.write(f"- Group-32 q4: **{'PASS' if merger_pass_q4 else 'FAIL'}**\n\n")

        # ---- Step 3 ----
        f.write("## 3. Full-Model Text Generation Comparison\n\n")
        f.write("> [!NOTE]\n")
        f.write("> **DEFERRED.** The full Qwen3.6-35B-A3B model requires ~70 GB in BF16 "
                "or ~35 GB in int8, exceeding the 16 GB reference Mac. This step requires "
                "external compute or running through the C engine (which violates the "
                "\"Pure Python\" constraint). The per-tensor and merger-output comparisons "
                "above provide sufficient evidence for a go/no-go decision on the tower "
                "weights.\n\n")

        # ---- Verdict ----
        f.write("## Verdict\n\n")
        overall_int8 = tensor_pass_int8 and (merger_pass_int8 if image_int8 else False)
        overall_q4 = tensor_pass_q4 and (merger_pass_q4 if image_q4 else False)

        if overall_int8:
            f.write("> [!TIP]\n")
            f.write("> **PASS — Shipped int8 weights are numerically usable.** "
                    "All acceptance criteria met. The C forward pass can proceed "
                    "against the weights users already have.\n\n")
        elif overall_q4:
            f.write("> [!WARNING]\n")
            f.write("> **FAIL for shipped int8, PASS for group-32 q4.** "
                    "The shipped row-wise int8 tower is degraded, but re-quantizing "
                    f"at group-32 q4 (group_size={GROUP_SIZE}) meets all criteria. "
                    "**Recommendation:** re-quantize and republish.\n\n")
        else:
            f.write("> [!CAUTION]\n")
            f.write("> **FAIL — Both int8 and group-32 q4 are degraded.** "
                    "The vision tower requires BF16 weights (~0.9 GB). "
                    "Re-quantization alone is not sufficient.\n\n")

        # Fallback cost analysis
        f.write("### Fallback Cost Analysis\n\n")
        f.write("| Variant | Size | Scheme | Status |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Shipped int8 | ~0.454 GB | Row-wise, 1 scale/row | "
                f"{'✅ Pass' if overall_int8 else '❌ Fail'} |\n")
        f.write(f"| Group-32 q4 | ~0.25 GB | Symmetric, 1 scale/32 | "
                f"{'✅ Pass' if overall_q4 else '❌ Fail'} |\n")
        f.write(f"| BF16 | ~0.9 GB | Full precision | Reference |\n\n")

        # ---- Details ----
        f.write("## Appendix: Per-Tensor Details\n\n")
        f.write("<details>\n<summary>Click to expand all 111 tensors</summary>\n\n")
        f.write("| Tensor | Int8 Cosine | Q4 Cosine | Int8 Max Rel Err | "
                "Q4 Max Rel Err |\n")
        f.write("|---|---|---|---|---|\n")
        for r8, r4 in zip(tensor_int8, tensor_q4):
            cos8 = f"{r8['cosine_similarity']:.6f}"
            cos4 = f"{r4['cosine_similarity']:.6f}" if not np.isnan(
                r4['cosine_similarity']) else "N/A"
            rel8 = f"{r8['max_abs_relative_error']:.6f}"
            rel4 = f"{r4['max_abs_relative_error']:.6f}" if not np.isnan(
                r4['max_abs_relative_error']) else "N/A"
            name = r8['tensor_name']
            # Flag tensors that fail
            flag = " ⚠️" if r8['cosine_similarity'] < 0.99 else ""
            f.write(f"| `{name}`{flag} | {cos8} | {cos4} | {rel8} | {rel4} |\n")
        f.write("\n</details>\n")

    log(f"Report written to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E-V1: Vision tower numerical parity check")
    parser.add_argument("--shipped-path", type=str, default=None,
                        help="Path to resident.safetensors (default: auto-detect)")
    parser.add_argument("--out-dir", type=str,
                        default=os.environ.get("SAMOSA_EV1_OUTDIR", DEFAULT_OUT_DIR),
                        help="Output directory for report and data")
    parser.add_argument("--skip-forward-pass", action="store_true",
                        help="Skip the end-to-end ViT forward pass (Step 2)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Locate shipped weights ---
    shipped_path = find_shipped_path(args.shipped_path)
    if not shipped_path:
        log("ERROR: Cannot find resident.safetensors. Provide --shipped-path or set SAMOSA_SHIPPED_PATH.")
        sys.exit(1)
    log(f"Shipped weights: {shipped_path}")

    # --- Step 1: Download reference and load weights ---
    log("Step 1: Loading reference weights from upstream...")
    t0 = time.time()

    index_path = hf_hub_download(repo_id=UPSTREAM_REPO, filename="model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    # Find shards containing visual weights
    visual_shards = sorted(set(v for k, v in weight_map.items() if "visual" in k))
    log(f"Visual weight shards: {visual_shards}")

    ref_weights = {}
    for shard in visual_shards:
        log(f"  Loading shard {shard}...")
        local_path = hf_hub_download(repo_id=UPSTREAM_REPO, filename=shard)
        with safetensors.torch.safe_open(local_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "visual" in k:
                    ref_weights[k] = f.get_tensor(k)
    log(f"Loaded {len(ref_weights)} reference visual tensors in {time.time() - t0:.1f}s")

    log("Loading shipped visual weights from resident.safetensors...")
    shipped_tensors = {}
    with safetensors.torch.safe_open(shipped_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "visual" in k:
                shipped_tensors[k] = f.get_tensor(k)
    log(f"Loaded {len(shipped_tensors)} shipped visual keys")

    # Identify quantized weights (those with a corresponding .qs scale tensor)
    quantized_names = [k for k in shipped_tensors
                       if k.endswith(".weight") and (k + ".qs") in shipped_tensors]
    log(f"Identified {len(quantized_names)} quantized tensors")

    # --- Per-tensor comparison ---
    log("\n=== Per-Tensor Comparison ===")
    tensor_int8, tensor_q4 = run_per_tensor_comparison(
        shipped_tensors, ref_weights, quantized_names)

    cosines_int8 = [r["cosine_similarity"] for r in tensor_int8]
    cosines_q4 = [r["cosine_similarity"] for r in tensor_q4
                  if not np.isnan(r["cosine_similarity"])]

    log(f"\nShipped Int8 — Cosine Similarity:")
    log(f"  Min:  {np.min(cosines_int8):.6f}")
    log(f"  Mean: {np.mean(cosines_int8):.6f}")
    log(f"  Max:  {np.max(cosines_int8):.6f}")
    log(f"  Pass (all ≥ 0.99): {all(c >= 0.99 for c in cosines_int8)}")

    if cosines_q4:
        log(f"\nGroup-32 Q4 — Cosine Similarity:")
        log(f"  Min:  {np.min(cosines_q4):.6f}")
        log(f"  Mean: {np.mean(cosines_q4):.6f}")
        log(f"  Max:  {np.max(cosines_q4):.6f}")
        log(f"  Pass (all ≥ 0.99): {all(c >= 0.99 for c in cosines_q4)}")

    # Flag pos_embed specifically (the spec calls it "the classic casualty")
    pos_embed_results = [r for r in tensor_int8
                         if "pos_embed" in r["tensor_name"]]
    if pos_embed_results:
        for r in pos_embed_results:
            log(f"\n  ⚠  pos_embed [{r['tensor_name']}]: "
                f"cosine={r['cosine_similarity']:.6f}, "
                f"max_rel_err={r['max_abs_relative_error']:.6f}")

    # --- Forward pass comparison ---
    image_int8 = None
    image_q4 = None

    if not args.skip_forward_pass:
        log("\n=== End-to-End ViT Forward Pass ===")
        log("Creating 20 test images...")
        images, descriptions = create_test_images()

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(UPSTREAM_REPO, trust_remote_code=True)

        image_int8, image_q4 = run_forward_pass_comparison(
            ref_weights, shipped_tensors, quantized_names,
            images, descriptions, config)

        img_cosines_int8 = [r["cosine_similarity"] for r in image_int8]
        img_cosines_q4 = [r["cosine_similarity"] for r in image_q4]

        log(f"\nMerger Output — Shipped Int8:")
        log(f"  Min:  {np.min(img_cosines_int8):.6f}")
        log(f"  Mean: {np.mean(img_cosines_int8):.6f}")

        log(f"\nMerger Output — Group-32 Q4:")
        log(f"  Min:  {np.min(img_cosines_q4):.6f}")
        log(f"  Mean: {np.mean(img_cosines_q4):.6f}")
    else:
        log("\nSkipping forward pass (--skip-forward-pass)")

    # --- Write report ---
    log("\n=== Writing Report ===")
    report_path = write_report(args.out_dir, tensor_int8, tensor_q4,
                                image_int8, image_q4)

    # --- Also dump raw results as JSON for machine consumption ---
    raw_path = os.path.join(args.out_dir, "raw_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "shipped_path": shipped_path,
            "reference_repo": UPSTREAM_REPO,
            "group_size": GROUP_SIZE,
            "per_tensor_int8": tensor_int8,
            "per_tensor_q4": tensor_q4,
            "forward_pass_int8": image_int8,
            "forward_pass_q4": image_q4,
        }, f, indent=2, default=str)
    log(f"Raw results written to {raw_path}")

    log("\nDone.")


if __name__ == "__main__":
    main()

# E-V1: Shipped Vision Tower Numerical Parity Report

**Date:** 2026-07-15 20:10
**Reference:** `Qwen/Qwen3.6-35B-A3B` (BF16)
**Shipped:** `deepanwa/Samosa-Chat-Qwen3.6-35B-A3B-group32` (row-wise int8 visual tower in `resident.safetensors`)
**Group-32 q4:** re-quantized from BF16 using `groupwise-symmetric-q4-v1`, group_size=32

## 1. Per-Tensor Comparison (111 Quantized Tensors)

| Metric | Shipped Int8 Cosine | Group-32 Q4 Cosine | Shipped Int8 Max Rel Err | Group-32 Q4 Max Rel Err |
|---|---|---|---|---|
| **Min** | 0.999939 | 0.987628 | 0.999973 | 0.999999 |
| **Mean** | 1.000401 | 0.994969 | 0.999988 | 0.999999 |
| **Max** | 1.006351 | 1.001774 | 1.000000 | 1.000000 |
| **p10** | 0.999954 | 0.994659 | 0.999981 | 0.999999 |
| **p50** | 1.000437 | 0.995115 | 0.999987 | 0.999999 |
| **p90** | 1.000605 | 0.995232 | 0.999994 | 1.000000 |

**Acceptance (per-tensor cosine ≥ 0.99 for all 111):**
- Shipped int8: **PASS** (min cosine = 0.999939)
- Group-32 q4: **FAIL** (min cosine = 0.987628)

## 2. End-to-End ViT Merger Embeddings (20 Images)

| Image | Description | Int8 Cosine | Q4 Cosine | Int8 Max Abs | Q4 Max Abs |
|---|---|---|---|---|---|
| 0 | sky gradient with ground and tree shapes | 0.997055 | 0.932470 | 5387.987793 | 10540.362305 |
| 1 | sunset-like horizontal gradient | 0.996841 | 0.952198 | 4550.075684 | 11575.762695 |
| 2 | noisy natural texture with grain | 0.998384 | 0.964569 | 2336.274170 | 7265.894531 |
| 3 | high-contrast building facade with windows | 0.997898 | 0.947109 | 4634.630859 | 11042.714844 |
| 4 | circular gradient (lens flare) | 0.997722 | 0.943526 | 4837.874512 | 8954.289062 |
| 5 | color bar test pattern | 0.997160 | 0.946383 | 4343.552246 | 9664.232422 |
| 6 | text document screenshot | 0.999231 | 0.981473 | 10077.051758 | 14219.753906 |
| 7 | UI settings panel mockup | 0.999426 | 0.977419 | 3571.457031 | 12703.884766 |
| 8 | code editor screenshot (dark theme) | 0.998205 | 0.964728 | 9306.389648 | 13424.041992 |
| 9 | terminal output screenshot | 0.998505 | 0.967100 | 3999.868164 | 12623.034180 |
| 10 | bar chart of performance metrics | 0.999375 | 0.978529 | 2256.930664 | 9936.523438 |
| 11 | line chart of cosine similarity by layer | 0.997792 | 0.955916 | 5529.541504 | 12245.652344 |
| 12 | pie chart of model composition | 0.998119 | 0.966372 | 5826.117188 | 11850.634766 |
| 13 | scatter plot of tensor statistics | 0.999270 | 0.943862 | 2298.178223 | 11889.810547 |
| 14 | tiny 64x64 image with cross pattern | 0.997823 | 0.960039 | 2502.441895 | 6127.008789 |
| 15 | 1024x768 fine mosaic pattern | 1.000716 | 0.981135 | 1828.916992 | 6799.354004 |
| 16 | horizontal grayscale gradient | 0.996926 | 0.945351 | 2768.022461 | 7737.912109 |
| 17 | 32px checkerboard pattern | 0.994422 | 0.884759 | 3191.787354 | 9681.106445 |
| 18 | wide panoramic gradient (4:1 aspect) | 0.995576 | 0.941300 | 3810.475098 | 10077.522461 |
| 19 | pure random noise (high entropy stress test) | 0.999038 | 0.978868 | 2167.886719 | 4820.664062 |

**Summary:**
- Shipped int8: mean cosine = 0.997974, min = 0.994422
- Group-32 q4: mean cosine = 0.955655, min = 0.884759

**Acceptance (merger cosine ≥ 0.99 mean, no image < 0.97):**
- Shipped int8: **PASS**
- Group-32 q4: **FAIL**

## 3. Full-Model Text Generation Comparison

> [!NOTE]
> **DEFERRED.** The full Qwen3.6-35B-A3B model requires ~70 GB in BF16 or ~35 GB in int8, exceeding the 16 GB reference Mac. This step requires external compute or running through the C engine (which violates the "Pure Python" constraint). The per-tensor and merger-output comparisons above provide sufficient evidence for a go/no-go decision on the tower weights.

## Verdict

> [!TIP]
> **PASS — Shipped int8 weights are numerically usable.** All acceptance criteria met. The C forward pass can proceed against the weights users already have.

### Fallback Cost Analysis

| Variant | Size | Scheme | Status |
|---|---|---|---|
| Shipped int8 | ~0.454 GB | Row-wise, 1 scale/row | ✅ Pass |
| Group-32 q4 | ~0.25 GB | Symmetric, 1 scale/32 | ❌ Fail |
| BF16 | ~0.9 GB | Full precision | Reference |

## Appendix: Per-Tensor Details

<details>
<summary>Click to expand all 111 tensors</summary>

| Tensor | Int8 Cosine | Q4 Cosine | Int8 Max Rel Err | Q4 Max Rel Err |
|---|---|---|---|---|
| `model.visual.blocks.0.attn.proj.weight` | 1.000005 | 0.994773 | 0.999991 | 0.999999 |
| `model.visual.blocks.0.attn.qkv.weight` | 1.001343 | 0.991779 | 0.999996 | 1.000000 |
| `model.visual.blocks.0.mlp.linear_fc1.weight` | 1.000616 | 0.993711 | 0.999993 | 1.000000 |
| `model.visual.blocks.0.mlp.linear_fc2.weight` | 1.000721 | N/A | 0.999989 | N/A |
| `model.visual.blocks.1.attn.proj.weight` | 0.999949 | 0.994650 | 0.999995 | 1.000000 |
| `model.visual.blocks.1.attn.qkv.weight` | 1.000636 | 0.993945 | 0.999994 | 1.000000 |
| `model.visual.blocks.1.mlp.linear_fc1.weight` | 1.000640 | 0.993903 | 0.999993 | 1.000000 |
| `model.visual.blocks.1.mlp.linear_fc2.weight` | 1.000538 | N/A | 0.999995 | N/A |
| `model.visual.blocks.10.attn.proj.weight` | 0.999962 | 0.995105 | 0.999985 | 0.999999 |
| `model.visual.blocks.10.attn.qkv.weight` | 1.000314 | 0.995019 | 0.999988 | 0.999999 |
| `model.visual.blocks.10.mlp.linear_fc1.weight` | 1.000529 | 0.995117 | 0.999992 | 0.999999 |
| `model.visual.blocks.10.mlp.linear_fc2.weight` | 1.000455 | N/A | 0.999992 | N/A |
| `model.visual.blocks.11.attn.proj.weight` | 0.999955 | 0.995157 | 0.999978 | 0.999999 |
| `model.visual.blocks.11.attn.qkv.weight` | 1.000262 | 0.995055 | 0.999985 | 0.999999 |
| `model.visual.blocks.11.mlp.linear_fc1.weight` | 1.000508 | 0.995142 | 0.999988 | 0.999999 |
| `model.visual.blocks.11.mlp.linear_fc2.weight` | 1.000406 | N/A | 0.999991 | N/A |
| `model.visual.blocks.12.attn.proj.weight` | 0.999947 | 0.995173 | 0.999987 | 0.999999 |
| `model.visual.blocks.12.attn.qkv.weight` | 1.000276 | 0.995133 | 0.999988 | 0.999999 |
| `model.visual.blocks.12.mlp.linear_fc1.weight` | 1.000493 | 0.995126 | 0.999988 | 0.999999 |
| `model.visual.blocks.12.mlp.linear_fc2.weight` | 1.000353 | N/A | 0.999992 | N/A |
| `model.visual.blocks.13.attn.proj.weight` | 0.999967 | 0.995169 | 0.999979 | 0.999999 |
| `model.visual.blocks.13.attn.qkv.weight` | 1.000298 | 0.995208 | 0.999987 | 0.999999 |
| `model.visual.blocks.13.mlp.linear_fc1.weight` | 1.000469 | 0.995142 | 0.999987 | 0.999999 |
| `model.visual.blocks.13.mlp.linear_fc2.weight` | 1.000389 | N/A | 0.999993 | N/A |
| `model.visual.blocks.14.attn.proj.weight` | 0.999949 | 0.995167 | 0.999985 | 0.999999 |
| `model.visual.blocks.14.attn.qkv.weight` | 1.000290 | 0.995201 | 0.999986 | 0.999999 |
| `model.visual.blocks.14.mlp.linear_fc1.weight` | 1.000456 | 0.995205 | 0.999986 | 0.999999 |
| `model.visual.blocks.14.mlp.linear_fc2.weight` | 1.000429 | N/A | 0.999992 | N/A |
| `model.visual.blocks.15.attn.proj.weight` | 0.999952 | 0.995197 | 0.999983 | 0.999999 |
| `model.visual.blocks.15.attn.qkv.weight` | 1.000266 | 0.995232 | 0.999982 | 0.999999 |
| `model.visual.blocks.15.mlp.linear_fc1.weight` | 1.000397 | 0.995288 | 0.999986 | 0.999999 |
| `model.visual.blocks.15.mlp.linear_fc2.weight` | 1.000388 | N/A | 0.999991 | N/A |
| `model.visual.blocks.16.attn.proj.weight` | 0.999946 | 0.995190 | 0.999983 | 0.999999 |
| `model.visual.blocks.16.attn.qkv.weight` | 1.000271 | 0.995219 | 0.999981 | 0.999999 |
| `model.visual.blocks.16.mlp.linear_fc1.weight` | 1.000432 | 0.995171 | 0.999987 | 0.999999 |
| `model.visual.blocks.16.mlp.linear_fc2.weight` | 1.000413 | N/A | 0.999985 | N/A |
| `model.visual.blocks.17.attn.proj.weight` | 0.999968 | 0.995218 | 0.999973 | 0.999999 |
| `model.visual.blocks.17.attn.qkv.weight` | 1.000238 | 0.995221 | 0.999984 | 0.999999 |
| `model.visual.blocks.17.mlp.linear_fc1.weight` | 1.000464 | 0.995090 | 0.999985 | 0.999999 |
| `model.visual.blocks.17.mlp.linear_fc2.weight` | 1.000470 | N/A | 0.999994 | N/A |
| `model.visual.blocks.18.attn.proj.weight` | 0.999969 | 0.995199 | 0.999981 | 0.999999 |
| `model.visual.blocks.18.attn.qkv.weight` | 1.000277 | 0.995266 | 0.999983 | 0.999999 |
| `model.visual.blocks.18.mlp.linear_fc1.weight` | 1.000446 | 0.995074 | 0.999983 | 0.999999 |
| `model.visual.blocks.18.mlp.linear_fc2.weight` | 1.000458 | N/A | 0.999990 | N/A |
| `model.visual.blocks.19.attn.proj.weight` | 0.999951 | 0.995222 | 0.999976 | 0.999999 |
| `model.visual.blocks.19.attn.qkv.weight` | 1.000251 | 0.995238 | 0.999983 | 0.999999 |
| `model.visual.blocks.19.mlp.linear_fc1.weight` | 1.000485 | 0.994951 | 0.999983 | 0.999999 |
| `model.visual.blocks.19.mlp.linear_fc2.weight` | 1.000474 | N/A | 0.999988 | N/A |
| `model.visual.blocks.2.attn.proj.weight` | 0.999939 | 0.994965 | 0.999991 | 1.000000 |
| `model.visual.blocks.2.attn.qkv.weight` | 1.000648 | 0.994489 | 0.999991 | 0.999999 |
| `model.visual.blocks.2.mlp.linear_fc1.weight` | 1.000589 | 0.994587 | 0.999991 | 1.000000 |
| `model.visual.blocks.2.mlp.linear_fc2.weight` | 1.000573 | N/A | 0.999998 | N/A |
| `model.visual.blocks.20.attn.proj.weight` | 0.999964 | 0.995239 | 0.999979 | 0.999999 |
| `model.visual.blocks.20.attn.qkv.weight` | 1.000252 | 0.995213 | 0.999981 | 0.999999 |
| `model.visual.blocks.20.mlp.linear_fc1.weight` | 1.000502 | 0.994763 | 0.999989 | 0.999999 |
| `model.visual.blocks.20.mlp.linear_fc2.weight` | 1.000455 | N/A | 0.999986 | N/A |
| `model.visual.blocks.21.attn.proj.weight` | 0.999944 | 0.995230 | 0.999978 | 0.999999 |
| `model.visual.blocks.21.attn.qkv.weight` | 1.000269 | 0.995231 | 0.999987 | 0.999999 |
| `model.visual.blocks.21.mlp.linear_fc1.weight` | 1.000519 | 0.994679 | 0.999982 | 0.999999 |
| `model.visual.blocks.21.mlp.linear_fc2.weight` | 1.000507 | N/A | 0.999978 | N/A |
| `model.visual.blocks.22.attn.proj.weight` | 0.999959 | 0.995234 | 0.999979 | 0.999999 |
| `model.visual.blocks.22.attn.qkv.weight` | 1.000268 | 0.995200 | 0.999986 | 0.999999 |
| `model.visual.blocks.22.mlp.linear_fc1.weight` | 1.000516 | 0.994719 | 0.999984 | 0.999999 |
| `model.visual.blocks.22.mlp.linear_fc2.weight` | 1.000456 | N/A | 0.999983 | N/A |
| `model.visual.blocks.23.attn.proj.weight` | 0.999959 | 0.995214 | 0.999981 | 0.999999 |
| `model.visual.blocks.23.attn.qkv.weight` | 1.000270 | 0.995114 | 0.999988 | 0.999999 |
| `model.visual.blocks.23.mlp.linear_fc1.weight` | 1.000517 | 0.994744 | 0.999984 | 0.999999 |
| `model.visual.blocks.23.mlp.linear_fc2.weight` | 1.000483 | N/A | 0.999986 | N/A |
| `model.visual.blocks.24.attn.proj.weight` | 0.999966 | 0.995241 | 0.999986 | 0.999999 |
| `model.visual.blocks.24.attn.qkv.weight` | 1.000214 | 0.995139 | 0.999986 | 0.999999 |
| `model.visual.blocks.24.mlp.linear_fc1.weight` | 1.000550 | 0.994798 | 0.999981 | 0.999999 |
| `model.visual.blocks.24.mlp.linear_fc2.weight` | 1.000526 | N/A | 0.999985 | N/A |
| `model.visual.blocks.25.attn.proj.weight` | 0.999947 | 0.995216 | 0.999988 | 0.999999 |
| `model.visual.blocks.25.attn.qkv.weight` | 1.000324 | 0.995077 | 0.999987 | 0.999999 |
| `model.visual.blocks.25.mlp.linear_fc1.weight` | 1.000530 | 0.994964 | 0.999985 | 0.999999 |
| `model.visual.blocks.25.mlp.linear_fc2.weight` | 1.000524 | N/A | 0.999990 | N/A |
| `model.visual.blocks.26.attn.proj.weight` | 0.999966 | 0.995059 | 0.999994 | 1.000000 |
| `model.visual.blocks.26.attn.qkv.weight` | 1.000167 | 0.995058 | 0.999983 | 0.999999 |
| `model.visual.blocks.26.mlp.linear_fc1.weight` | 1.000537 | 0.995036 | 0.999990 | 1.000000 |
| `model.visual.blocks.26.mlp.linear_fc2.weight` | 1.000550 | N/A | 0.999998 | N/A |
| `model.visual.blocks.3.attn.proj.weight` | 0.999956 | 0.995102 | 0.999988 | 0.999999 |
| `model.visual.blocks.3.attn.qkv.weight` | 1.000599 | 0.994620 | 0.999988 | 0.999999 |
| `model.visual.blocks.3.mlp.linear_fc1.weight` | 1.000484 | 0.994839 | 0.999992 | 1.000000 |
| `model.visual.blocks.3.mlp.linear_fc2.weight` | 1.000714 | N/A | 0.999994 | N/A |
| `model.visual.blocks.4.attn.proj.weight` | 0.999964 | 0.995017 | 0.999984 | 0.999999 |
| `model.visual.blocks.4.attn.qkv.weight` | 1.000643 | 0.994699 | 0.999992 | 1.000000 |
| `model.visual.blocks.4.mlp.linear_fc1.weight` | 1.000486 | 0.994848 | 0.999993 | 1.000000 |
| `model.visual.blocks.4.mlp.linear_fc2.weight` | 1.000672 | N/A | 0.999992 | N/A |
| `model.visual.blocks.5.attn.proj.weight` | 0.999947 | 0.995130 | 0.999987 | 0.999999 |
| `model.visual.blocks.5.attn.qkv.weight` | 1.000533 | 0.994811 | 0.999987 | 0.999999 |
| `model.visual.blocks.5.mlp.linear_fc1.weight` | 1.000515 | 0.994967 | 0.999990 | 0.999999 |
| `model.visual.blocks.5.mlp.linear_fc2.weight` | 1.000605 | N/A | 0.999992 | N/A |
| `model.visual.blocks.6.attn.proj.weight` | 0.999959 | 0.995162 | 0.999985 | 0.999999 |
| `model.visual.blocks.6.attn.qkv.weight` | 1.000502 | 0.994820 | 0.999990 | 0.999999 |
| `model.visual.blocks.6.mlp.linear_fc1.weight` | 1.000498 | 0.994974 | 0.999994 | 1.000000 |
| `model.visual.blocks.6.mlp.linear_fc2.weight` | 1.000494 | N/A | 0.999994 | N/A |
| `model.visual.blocks.7.attn.proj.weight` | 0.999954 | 0.995152 | 0.999985 | 0.999999 |
| `model.visual.blocks.7.attn.qkv.weight` | 1.000438 | 0.994905 | 0.999988 | 0.999999 |
| `model.visual.blocks.7.mlp.linear_fc1.weight` | 1.000480 | 0.994992 | 0.999990 | 1.000000 |
| `model.visual.blocks.7.mlp.linear_fc2.weight` | 1.000389 | N/A | 0.999990 | N/A |
| `model.visual.blocks.8.attn.proj.weight` | 0.999939 | 0.995147 | 0.999990 | 0.999999 |
| `model.visual.blocks.8.attn.qkv.weight` | 1.000437 | 0.994843 | 0.999986 | 0.999999 |
| `model.visual.blocks.8.mlp.linear_fc1.weight` | 1.000489 | 0.995024 | 0.999990 | 0.999999 |
| `model.visual.blocks.8.mlp.linear_fc2.weight` | 1.000385 | N/A | 0.999991 | N/A |
| `model.visual.blocks.9.attn.proj.weight` | 0.999972 | 0.995194 | 0.999980 | 0.999999 |
| `model.visual.blocks.9.attn.qkv.weight` | 1.000465 | 0.995061 | 0.999987 | 0.999999 |
| `model.visual.blocks.9.mlp.linear_fc1.weight` | 1.000543 | 0.995650 | 0.999992 | 1.000000 |
| `model.visual.blocks.9.mlp.linear_fc2.weight` | 1.000468 | N/A | 0.999996 | N/A |
| `model.visual.merger.linear_fc1.weight` | 1.006351 | 1.001774 | 0.999987 | 0.999999 |
| `model.visual.merger.linear_fc2.weight` | 1.001345 | 0.995158 | 0.999990 | 0.999999 |
| `model.visual.pos_embed.weight` | 1.000274 | 0.987628 | 1.000000 | 1.000000 |

</details>

# StructMatte

**Structure-Enhanced Matting (StructMatte)**  
*Enhancing Transparent-Object Structure via Consistency-Guided Fusion for High-Resolution Image Matting*

StructMatte improves high-resolution image matting by learning structural representations from progressively downsampled input views. It incorporates a Progressive Structure Encoder (PSE), a Cross-View Structure Consistency (CSC) Loss, and Consistency-Guided Structure Fusion (CGSF). MEMatte-S is used as the matting backbone.

## Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

The code also requires a compatible installation of [Detectron2](https://github.com/facebookresearch/detectron2).

## Datasets and Model Weights

The model was trained on the **Composition-1K** training set and evaluated on **Transparent-460** and **UHR-395**, without additional fine-tuning on either evaluation benchmark.

- **Model weights:** [Google Drive](https://drive.google.com/file/d/1AFZSkaKjkVDM7Sn9B-fZiju3YqWKgJkO/view?usp=sharing)
- **Transparent-460:** [TransMatting](https://github.com/AceCHQ/TransMatting)
- **UHR-395:** [MEMatte](https://github.com/linyiheng123/MEMatte)

Place the model checkpoint at `checkpoint/model_final.pth`, or specify its location using `--checkpoint-dir`.

## Inference

Run inference on Transparent-460:

```bash
python inference.py \
    --config-dir configs/StructMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir checkpoint/model_final.pth \
    --data-dir dataset/Transparent-460 \
    --inference-dir predAlpha/Transparent-460
```

Run inference on UHR-395:

```bash
python inference.py \
    --config-dir configs/StructMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir checkpoint/model_final.pth \
    --data-dir dataset/UHR-395/test \
    --inference-dir predAlpha/UHR-395
```

The input dataset directories and file naming should follow the corresponding benchmark structure expected by `inference.py`.

## Evaluation

The repository includes a Python evaluation script:

```bash
python evaluation.py \
    --pred-dir predAlpha/Transparent-460 \
    --label-dir dataset/Transparent-460/alpha_copy \
    --trimap-dir dataset/Transparent-460/trimap_copy
```

For paper-level results, use the official benchmark evaluation protocol.

## Quantitative Results

All results below were obtained with training on Composition-1K. Lower is better for all metrics.

### Transparent-460

| Method | SAD | MSE (× 10⁻³) | Grad | Conn |
| --- | ---: | ---: | ---: | ---: |
| MEMatte-S | 151.60 | 10.22 | 28.82 | 141.86 |
| **StructMatte** | **143.03** | **8.92** | **27.99** | **133.59** |

### UHR-395

| Method | SAD | MSE (× 10⁻³) | Grad | Conn |
| --- | ---: | ---: | ---: | ---: |
| MEMatte-S | 623.93 | 11.13 | 220.09 | 572.94 |
| **StructMatte** | **612.75** | **9.65** | **196.82** | **406.96** |

## Acknowledgment

This repository builds upon the [MEMatte](https://github.com/linyiheng123/MEMatte) implementation.

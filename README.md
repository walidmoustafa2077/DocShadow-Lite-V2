# LP-IOANet: Document Shadow Removal

High-resolution document shadow removal using Laplacian Pyramid IO-Attention Network.

## Architecture Overview

```
[Input Image @ 768x1024] 
          |
[Laplacian Decomposition] ----> [Residual Level 1 @ 768x1024] 
          |                           |
[Downsample x2] ---------------> [Residual Level 2 @ 384x512]
          |                           |
[Downsample x2]                       |
          |                           |
[Low-Freq Image @ 192x256]            |
          |                           |
   (STAGE 1: IOANet)                  |
   [Input Attention]                  |
          |                           |
   [MobileNet Encoder]                |
          |                           |
   [7x7x1024 Bottleneck]              |
          |                           |
   [Feature Blending Decoder]         |
          |                           |
   [Output Attention Blending]        |
          |                           |
   (STAGE 2: LPTN-Lite)               |
   [Refinement Module 2] <------------/ 
          |
   [Upsample x2]
          |
   [Refinement Module 1] <------------/
          |
[Final Output @ 768x1024]
```

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
DocShadow-Lite V2/
├── config/
│   └── config.yaml          # Training configuration
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── lp_ioanet.py     # Model architecture
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py       # Dataset classes
│   ├── losses/
│   │   ├── __init__.py
│   │   └── shadow_loss.py   # Loss functions
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py       # Evaluation metrics
│       └── visualization.py # Visualization tools
├── Dataset/
│   ├── train/
│   │   ├── input/           # Shadowed images
│   │   ├── target/          # Ground truth
│   │   └── mask/            # Shadow masks (optional)
│   └── test/
│       ├── input/
│       ├── target/
│       └── mask/
├── train.py                 # Training script
├── inference.py             # Inference script
├── evaluate.py              # Evaluation script
└── requirements.txt
```

## Dataset Preparation

1. Place your training data in `Dataset/train/`:
   - `input/`: Shadowed document images
   - `target/`: Shadow-free ground truth images
   - `mask/`: Binary shadow masks (optional)

2. Place test data in `Dataset/test/` with the same structure.

3. Ensure matching filenames between input, target, and mask folders.

## Training

### Stage 1: Train IOANet (Low Resolution)

```bash
python train.py --config config/config.yaml --stage 1
```

This trains the core shadow removal network at 192×256 resolution with:
- L1 + LPIPS loss
- 1000 epochs
- MobileNetV2 encoder (pretrained)

### Stage 2: Train LP-IOANet (High Resolution)

```bash
python train.py --config config/config.yaml --stage 2
```

This trains the refinement modules at 768×1024 resolution:
- IOANet weights are frozen
- L1 loss only
- 200 epochs

### Resume Training

```bash
python train.py --config config/config.yaml --stage 1 --resume checkpoints/stage1/checkpoint.pth
```

## Inference

### Single Image

```bash
python inference.py --checkpoint checkpoints/stage2/best_model.pth --input image.jpg --output result.jpg
```

### Directory Processing

```bash
python inference.py --checkpoint checkpoints/stage2/best_model.pth --input ./input_dir --output ./output_dir
```

### Benchmark

```bash
python inference.py --checkpoint checkpoints/stage2/best_model.pth --benchmark
```

### Export Models

```bash
# Export to ONNX
python inference.py --checkpoint checkpoints/stage2/best_model.pth --export-onnx model.onnx

# Export to TorchScript
python inference.py --checkpoint checkpoints/stage2/best_model.pth --export-torchscript model.pt
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/stage2/best_model.pth --test_dir Dataset/test
```

## Configuration

Edit `config/config.yaml` to customize:

- Model architecture (backbone, channels)
- Training parameters (epochs, batch size, learning rate)
- Loss weights
- Data augmentation
- Hardware settings

## Key Features

1. **Two-Stage Training**: 
   - Stage 1: Low-res shadow removal with IOANet
   - Stage 2: High-res refinement with LPTN-Lite

2. **Efficient Architecture**:
   - ~1.47 GFLOPs at 768×1024 (vs 550 GFLOPs for BEDSR)
   - MobileNetV2 backbone for efficient feature extraction
   - Depthwise separable convolutions in refiners

3. **Input/Output Attention**:
   - Input attention: Focuses encoder on shadow regions
   - Output attention: Alpha-blends refined output with original

4. **Laplacian Pyramid**:
   - Decouples content correction from detail preservation
   - Enables efficient high-resolution processing

## Performance

| Metric | Value |
|--------|-------|
| GFLOPs | ~1.47 |
| Parameters | ~3.5M |
| FPS (GPU) | ~20 |

## Citation

If you use this code, please cite the original LP-IOANet paper.

## License

MIT License

# TianLong GPT

A character-level GPT language model trained on Jin Yong's classic wuxia novel "Demi-Gods and Semi-Devils" (天龙八部), based on [nanoGPT](https://github.com/karpathy/nanoGPT).

## Overview

This project trains a GPT-style transformer model to generate Chinese wuxia-style text using "Demi-Gods and Semi-Devils" as the training corpus. The model uses character-level tokenization, making it straightforward to handle Chinese text without complex segmentation.

## Features

- **Character-level tokenization**: Directly maps Chinese characters to integers
- **UTF-8 support**: Full support for Chinese text processing
- **Lightweight**: Can be trained on a single GPU or CPU
- **Flash Attention**: Supports PyTorch 2.0+ Flash Attention for faster training

## Project Structure

```
.
├── nanoGPT-master/          # Core nanoGPT implementation
│   ├── model.py             # GPT model definition (~300 lines)
│   ├── train.py             # Training script with DDP support
│   ├── sample.py            # Text generation script
│   ├── configurator.py      # Configuration parser
│   ├── config/              # Training configurations
│   │   └── train_tianlong.py    # Config for TianLong training
│   ├── data/                # Dataset directory
│   │   └── tianlong/        # TianLong dataset
│   │       └── prepare.py   # Data preprocessing
│   └── ...
├── input.txt                # Source text: Demi-Gods and Semi-Devils
└── README.md
```

## Quick Start

### Prerequisites

```bash
cd nanoGPT-master
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### Data Preparation

Copy the novel text to the data directory and run preprocessing:

```bash
# Copy input.txt to the tianlong data directory
cp input.txt nanoGPT-master/data/tianlong/

# Run preprocessing
cd nanoGPT-master/data/tianlong
python prepare.py
```

This generates:
- `train.bin` - Training data (90% of the text)
- `val.bin` - Validation data (10% of the text)
- `meta.pkl` - Character to integer mappings

### Training

```bash
cd nanoGPT-master
python train.py config/train_tianlong.py
```

**Training Configuration** ([config/train_tianlong.py](nanoGPT-master/config/train_tianlong.py)):

| Parameter | Value |
|-----------|-------|
| Layers | 6 |
| Attention Heads | 6 |
| Embedding Dim | 384 |
| Context Length | 256 |
| Batch Size | 32 |
| Training Iterations | 5,000 |
| Learning Rate | 1e-3 |
| Dropout | 0.2 |

### Text Generation

```bash
cd nanoGPT-master
python sample.py --out_dir=out-tianlong --start="段誉" --num_samples=5 --max_new_tokens=200
```

Or use a custom prompt from a file:

```bash
python sample.py --out_dir=out-tianlong --start="FILE:prompt.txt" --num_samples=3
```

## Model Architecture

The model follows the standard GPT architecture:

- Transformer decoder with causal self-attention
- Layer normalization with optional bias
- GELU activation
- Residual connections
- Dropout for regularization

## Training Details

- **Optimizer**: AdamW with weight decay
- **Learning Rate Schedule**: Warmup (100 steps) + cosine decay
- **Gradient Clipping**: 1.0
- **Mixed Precision**: bfloat16/float16 support
- **Distributed Training**: DDP support via `torchrun`

### Multi-GPU Training

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_tianlong.py
```

## Customization

You can modify the training configuration by editing `config/train_tianlong.py` or passing command-line arguments:

```bash
python train.py config/train_tianlong.py --batch_size=64 --max_iters=10000 --learning_rate=5e-4
```

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | GPT model implementation with Flash Attention support |
| `train.py` | Training loop with checkpointing and logging |
| `sample.py` | Text generation with temperature and top-k sampling |
| `config/train_tianlong.py` | Training hyperparameters for TianLong |
| `data/tianlong/prepare.py` | Dataset preparation with UTF-8 encoding |

## Key Modifications for Chinese Text

The original nanoGPT was adapted for Chinese text processing:

1. **UTF-8 Encoding**: Modified `prepare.py` to read text with `encoding='utf-8'`
2. **Character-level Tokenization**: Each Chinese character is treated as a token
3. **Removed Auto-download**: Disabled automatic Shakespeare dataset download

## Example Output

After training, the model can generate text in the style of Jin Yong's wuxia novels:

```
段誉心中一动，说道：“这位姑娘，在下段誉，误入贵处，还请见谅。”
那女子微微一笑，道：“段公子客气了，小女子姓王，单名一个语字。”
```

## Acknowledgments

- Based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Training data from Jin Yong's "Demi-Gods and Semi-Devils" (天龙八部)

## License

This project follows the same license as the original nanoGPT project. See [LICENSE](nanoGPT-master/LICENSE) for details.

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{tianlonggpt,
  title = {TianLong GPT: Chinese Wuxia Text Generation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/tianlong-gpt}
}
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

**Note**: This project is for educational and research purposes. The training data is from a copyrighted work; please respect applicable copyright laws in your jurisdiction.

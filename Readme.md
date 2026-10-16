# MiniMoE

A small language model With MoE architecture trained on the TinyStories dataset.

## Features

- **MoEarchitecture**
- **Multi-head Latent Attention**
- **RoPE Positional Embeddings**
- **AMP Training**
- **Top-k sampling with temperature control**

## Installation
 
```bash
pip install -r requirements.txt
```

**Train the model:**
```bash
python main.py
```

**Generate text (after training):**
```bash
python test_generate.py
```


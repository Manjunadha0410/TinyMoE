MODEL_CONFIG = {
    "vocab_size": 32000,
    "emb_dim": 384,
    "hidden_dim": 1024,
    "num_layers": 8,
    "num_heads": 6,
    "comkv_dim": 128,
    "comq_dim": 128,
    "content_head_dimension": 32,
    "pos_rope_head_dim": 32,
    "max_length": 256,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
}

TRAIN_CONFIG = {
    "batch_size": 2,
    "stride": 128,
    "lr": 3e-4,
    "num_steps": 10000,
    "warmup_steps": 100,
    "use_amp": True,
    "tokenizer_dir": "./tokenizer",
    "grad_accum_steps": 4,
    "grad_clip": 1.0,
}
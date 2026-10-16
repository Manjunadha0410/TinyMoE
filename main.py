import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import os
from config import MODEL_CONFIG, TRAIN_CONFIG
from tokenizer import load_tokenizer
from dataset import dataloader_stream
from model import LangModel
from train import train_model
from generate import generate

def load_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LangModel(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['config']

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(save_dir=TRAIN_CONFIG["tokenizer_dir"])
    dataloader, vocab_sz = dataloader_stream(
        b_sz=TRAIN_CONFIG["batch_size"],
        max_l=MODEL_CONFIG["max_length"],
        stride=TRAIN_CONFIG["stride"],
        tokenizer_dir=TRAIN_CONFIG["tokenizer_dir"]
    )
    MODEL_CONFIG["vocab_size"] = vocab_sz
    model = LangModel(MODEL_CONFIG).to(device)
    #model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=0.01)
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        num_steps=TRAIN_CONFIG["num_steps"],
        device=device,
        use_amp=TRAIN_CONFIG["use_amp"],
        warmup_steps=TRAIN_CONFIG.get("warmup_steps", 100),
        grad_clip=TRAIN_CONFIG.get("grad_clip", 1.0),
        grad_accum_steps=TRAIN_CONFIG.get("grad_accum_steps", 1)
    )

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': MODEL_CONFIG,
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"\nModel saved to {model_path}")
    prompt = "Once upon a time"
    output_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=250,
        device=device
    )
    print("\nGenerated Output:")
    print(output_text)

if __name__ == "__main__":
    main()
    

import torch
from model import LangModel
from config import MODEL_CONFIG, TRAIN_CONFIG
from transformers import AutoTokenizer
from generate import generate

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")
tokenizer = AutoTokenizer.from_pretrained(TRAIN_CONFIG["tokenizer_dir"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
checkpoint_path = "./checkpoints/model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
if "config" in checkpoint:
    model = LangModel(checkpoint["config"])
else:
    model = LangModel(MODEL_CONFIG)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
print("Model loaded successfully")
prompt = input("Enter a prompt: ").strip()
if not prompt:
    raise SystemExit("No prompt entered.")
print(f"\nPrompt: {prompt}")

try:
    output = generate(model, tokenizer, prompt, max_new_tokens=200, device=device)
    print(f"Generated: {output}")
except Exception as e:
    print(f"Error during generation: {e}")
    import traceback
    traceback.print_exc()

import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7, top_k=20, device="cuda"):
    model.eval()
    model.to(device)
    model.reset_cache()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    generated = input_ids[0].tolist()
    logits, _ = model(input_ids,start_pos=0,use_cache = True)
    if temperature > 0:
        next_token_logits = logits[:, -1, :] / temperature
    else:
        next_token_logits = logits[:, -1, :]

    if top_k is not None:
        top_k = min(top_k, next_token_logits.size(-1))
        v, _ = torch.topk(next_token_logits,top_k)
        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

    probs = F.softmax(next_token_logits, dim =-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated.append(next_token.item())
    curr_token = next_token

    for i in range(1,max_new_tokens):
        logits , _ = model(curr_token, start_pos = input_ids.shape[1] + i - 1,use_cache = True )
        next_token_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token.item())
        curr_token = next_token
    return tokenizer.decode(generated)
    
    
    
import torch
import torch.nn as nn
import math

def get_lr(step, num_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def train_model(model,dataloader,optimizer, num_steps = 200 , device= "cuda", use_amp=False, warmup_steps=100, grad_clip=1.0, grad_accum_steps=1):
    model.train()
    model.to(device)
    loss_fn  = nn.CrossEntropyLoss()
    device_type = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(use_amp and device == "cuda"))
    base_lr = optimizer.param_groups[0]['lr']
    step = 0
    accum_step = 0
    current_lr = base_lr
    print("start_training")
    for input_ids, target_ids in dataloader:
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        with torch.amp.autocast(device_type, enabled=use_amp):
            logits, aux_loss = model(input_ids)
            loss = loss_fn(logits.view(-1,logits.size(-1)),target_ids.view(-1))
            total_loss = (loss+0.01*aux_loss) / grad_accum_steps
        scaler.scale(total_loss).backward()

        accum_step += 1
        if accum_step % grad_accum_steps == 0:
            current_lr = get_lr(step, num_steps, warmup_steps, base_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % 20 == 0:
                print(f"Step {step}/{num_steps} | LR: {current_lr:.6f} | CE Loss: {loss.item():.4f} | Aux Loss: {aux_loss.item():.4f}")

            step += 1
            if step >= num_steps:
                break

    if accum_step % grad_accum_steps != 0:
        current_lr = get_lr(step, num_steps, warmup_steps, base_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Ropeemb(nn.Module):
    def __init__(self, rot_dim,maxseq_len , theta_freq):
        super().__init__()
        self.rot_dim = rot_dim
        dim_pair_indices = torch.arange(0,rot_dim,2).float()
        inv_freq= 1.0/( theta_freq**(dim_pair_indices/rot_dim))
        seq_pos_ind = torch.arange(maxseq_len).float()
        pos_ang_matrix = torch.einsum('p,f -> pf', seq_pos_ind, inv_freq)
        full_dim_ang_emb = torch.cat((pos_ang_matrix,pos_ang_matrix),dim = -1)
        self.register_buffer("cosine_table",full_dim_ang_emb.cos(),persistent = False )
        self.register_buffer("sine_table",full_dim_ang_emb.sin(),persistent = False )
        
    def _rotate_half(self, x):
        half_dim = x.shape[-1]//2
        return torch.cat((-x[...,half_dim:],x[...,:half_dim]),dim=-1)
    
    def forward(self,x,start_pos: int = 0):
        seq_len = x.size(1)
        
        cos = self.cosine_table[start_pos:start_pos+seq_len,:].to(dtype=x.dtype, device=x.device)
        sin = self.sine_table[start_pos:start_pos+seq_len,:].to(dtype=x.dtype, device=x.device)
        
        if x.ndim==4:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
            
        return (x*cos) + (self._rotate_half(x)*sin)
    
class RMSNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(emb_dim))
        
    def forward(self,x):
        x_f32 = x.float()
        rrms = torch.rsqrt((x_f32*x_f32).mean(-1, keepdim=True) + self.eps)
        return x*rrms*self.weight
        
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x, approximate="tanh")

class FF(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"],cfg["hidden_dim"]),
                                    GELU(),nn.Linear(cfg["hidden_dim"],cfg["emb_dim"]))
        
    def forward(self,x):
        return self.layers(x)

class MLA(nn.Module):
    def __init__(self,d_in,num_heads,comkv_dim,comq_dim,
                 content_head_dimension,
                 pos_rope_head_dim,max_l):
        super().__init__()
        self.d_model = d_in
        self.num_heads= num_heads
        self.d_c_kv = comkv_dim
        self.d_c_q = comq_dim 
        self.d_hc = content_head_dimension
        self.d_hr = pos_rope_head_dim
        
        self.kv_down_proj = nn.Linear(self.d_model,self.d_c_kv,bias = False)
        self.key_up_proj = nn.Linear(self.d_c_kv,self.num_heads*self.d_hc , bias =False )
        self.value_up_proj = nn.Linear(self.d_c_kv,self.num_heads*self.d_hc , bias =False ) 
        self.key_pos_proj = nn.Linear(self.d_model,self.d_hr, bias =False )
        self.q_down_proj = nn.Linear(self.d_model,self.d_c_q,bias = False)
        self.q_up_proj = nn.Linear(self.d_c_q, self.num_heads*self.d_hc , bias =False )
        self.q_pos_proj = nn.Linear(self.d_c_q, self.num_heads * self.d_hr, bias=False)
        self.out_proj = nn.Linear(self.num_heads*self.d_hc ,self.d_model,bias = False )
        
        self.rope = Ropeemb(rot_dim=self.d_hr ,maxseq_len = max_l  , theta_freq=10000.0)
        
        self.cache_c_kv = None
        self.cache_k_rope=None
        
    def reset_cache(self):
        self.cache_c_kv = None
        self.cache_k_rope = None
        
    def forward(self,x,start_pos=0,use_cache=False,causal_mask=None):
        b, seq_len,_ = x.shape
        c_q = self.q_down_proj(x)
        q_content = self.q_up_proj(c_q).view(b,seq_len,self.num_heads,self.d_hc)
        q_pos = self.q_pos_proj(c_q).view(b,seq_len,self.num_heads,self.d_hr)
        q_pos = self.rope(q_pos,start_pos=start_pos)
        
        c_kv = self.kv_down_proj(x)
        k_pos = self.key_pos_proj(x).unsqueeze(2)
        k_pos = self.rope(k_pos, start_pos = start_pos)
        
        if use_cache:
            if self.cache_c_kv is None or start_pos == 0:
                self.cache_c_kv = c_kv
                self.cache_k_rope = k_pos
            else:
                self.cache_c_kv = torch.cat([self.cache_c_kv,c_kv],dim = 1)
                self.cache_k_rope = torch.cat([self.cache_k_rope,k_pos],dim = 1)
                
            c_kv_total = self.cache_c_kv
            k_pos_total = self.cache_k_rope
        else:
            c_kv_total = c_kv
            k_pos_total = k_pos
            
        kv_len = c_kv_total.size(1)
        
        scale = 1.0/ math.sqrt(self.d_hc+self.d_hr)
        
        if use_cache and seq_len ==1 :
           w_uk = self.key_up_proj.weight.view(self.num_heads,self.d_hc, self.d_c_kv)
           q_absorbed = torch.einsum("bshd,hdl->bshl", q_content, w_uk).transpose(1, 2)
           scores_content = torch.matmul(q_absorbed, c_kv_total.unsqueeze(1).transpose(-2, -1))

           k_pos_expanded = k_pos_total.expand(-1, -1, self.num_heads, -1)
           q_pos_t = q_pos.transpose(1, 2)
           k_pos_t = k_pos_expanded.transpose(1, 2)
           scores_pos = torch.matmul(q_pos_t, k_pos_t.transpose(-2, -1))
           attn_scores = (scores_content + scores_pos)*scale
           attn_weights = F.softmax(attn_scores,dim = -1)
           context_latent = torch.matmul(attn_weights, c_kv_total.unsqueeze(1))
           w_uv = self.value_up_proj.weight.view(self.num_heads, self.d_hc, self.d_c_kv)
           context_output = torch.einsum("bhql,hdl->bhqd", context_latent, w_uv)
           
        else:
            k_content = self.key_up_proj(c_kv_total).view(b,kv_len,self.num_heads,self.d_hc)
            v_content = self.value_up_proj(c_kv_total).view(b, kv_len, self.num_heads, self.d_hc)
            k_pos_expanded = k_pos_total.expand(-1, -1, self.num_heads, -1)
            q_full = torch.cat([q_content, q_pos], dim=-1).transpose(1, 2)
            k_full = torch.cat([k_content, k_pos_expanded], dim=-1).transpose(1, 2)
            v_full = v_content.transpose(1, 2)
            
            if causal_mask is not None:
                context_output = F.scaled_dot_product_attention(
                    q_full, k_full, v_full, attn_mask=(causal_mask != 0)
                )
            else:
                is_causal = (seq_len > 1 and seq_len == kv_len)
                context_output = F.scaled_dot_product_attention(
                    q_full, k_full, v_full, is_causal=is_causal
                )
             
        reshaped_context = context_output.transpose(1, 2).contiguous().view(
            b, seq_len, self.num_heads * self.d_hc
        )
        return self.out_proj(reshaped_context)
                    

class MOE(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.num_experts = cfg["num_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.num_shared_experts = cfg.get("num_shared_experts",0)
        
        self.gate = nn.Linear(self.emb_dim,self.num_experts, bias = False)
        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.emb_dim, self.hidden_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.emb_dim, self.hidden_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts,self.hidden_dim,self.emb_dim))
        if self.num_shared_experts>0:
            shared_hidden = self.hidden_dim*self.num_shared_experts
            self.shared_w1 = nn.Linear(self.emb_dim, shared_hidden, bias = False )
            self.shared_w2 = nn.Linear(self.emb_dim, shared_hidden, bias = False )
            self.shared_w3 = nn.Linear(shared_hidden ,self.emb_dim, bias = False )
            
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w1)
        nn.init.kaiming_uniform_(self.w2)
        nn.init.kaiming_uniform_(self.w3)
        
    def compute_aux_loss(self,router_logits,selected_experts):
        probs = F.softmax(router_logits,dim =-1)
        tokens_per_expert = torch.bincount( selected_experts.view(-1), minlength = self.num_experts ).float()
        density = tokens_per_expert/selected_experts.numel()
        mean_prob = probs.mean(dim=0)
        aux_loss = self.num_experts*torch.sum(density*mean_prob)
        return aux_loss
    
    def forward(self, x):
        batch, seq_len, emb_dim = x.shape
        x_flat = x.view(-1, emb_dim)
        num_tokens = x_flat.shape[0]
        
        router_logits = self.gate(x_flat)
        topk_weights, topk_indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        topk_probs = F.softmax(topk_weights, dim=-1)
        aux_loss = self.compute_aux_loss(router_logits, topk_indices) if self.training else torch.tensor(0.0, device=x.device)
        
        flat_experts = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1, 1)
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.num_experts_per_tok)
        
        sorted_expert_ids, sort_positions = torch.sort(flat_experts)
        sorted_tokens = x_flat[token_indices[sort_positions]]
        sorted_probs = flat_probs[sort_positions]
        
        out_sorted = torch.zeros_like(sorted_tokens)

        for expert_idx in range(self.num_experts):
            mask = (sorted_expert_ids == expert_idx)
            if not mask.any():
                continue
            expert_in = sorted_tokens[mask]
            h = F.silu(F.linear(expert_in, self.w1[expert_idx].T)) * F.linear(expert_in, self.w2[expert_idx].T)
            expert_out = F.linear(h, self.w3[expert_idx].T)
            out_sorted[mask] = expert_out * sorted_probs[mask]
        
        out_routed = torch.zeros_like(x_flat)
        orig_token_indices = token_indices[sort_positions]
        out_routed.index_add_(0, orig_token_indices, out_sorted)
        out = out_routed.view(batch, seq_len, emb_dim)
        
        if self.num_shared_experts > 0:
            shared_h = F.silu(self.shared_w1(x)) * self.shared_w2(x)
            out = out + self.shared_w3(shared_h)
        
        return out, aux_loss
        
     
            
class Transformerblock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn_norm1 = RMSNorm(cfg["emb_dim"])
        self.mla = MLA(d_in = cfg["emb_dim"],num_heads=cfg["num_heads"],comkv_dim=cfg["comkv_dim"],
                       comq_dim=cfg["comq_dim"],content_head_dimension=cfg["content_head_dimension"],
                        pos_rope_head_dim=cfg["pos_rope_head_dim"],max_l=cfg["max_length"] )
        self.attn_norm2 = RMSNorm(cfg["emb_dim"])
        self.moe = MOE(cfg)
        
    def forward(self,x,start_pos=0,use_cache=False, causal_mask= None):
        h = x+self.mla(self.attn_norm1(x),start_pos = start_pos, use_cache = use_cache, causal_mask = causal_mask )
        moe_out , aux_loss = self.moe(self.attn_norm2(h))
        return h+moe_out, aux_loss

class LangModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.trfblks= nn.ModuleList( [ Transformerblock(cfg) for _ in range(cfg["num_layers"]) ])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.lm_head = nn.Linear(cfg["emb_dim"],cfg["vocab_size"], bias = False)
        
    def reset_cache(self):
        for blk in self.trfblks:
            blk.mla.reset_cache()
            
    def forward(self, input_ids , start_pos =0 , use_cache =False):
        x = self.token_emb(input_ids)
        total_aux_loss = 0.0
        for blk in self.trfblks:
            x,aux_loss= blk(x,start_pos= start_pos, use_cache = use_cache)
            total_aux_loss += aux_loss
        logits = self.lm_head(self.final_norm(x))
        return logits , total_aux_loss



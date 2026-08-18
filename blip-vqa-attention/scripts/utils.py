from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import math
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODE = "post"
BETA = 7.0
GAMMA = 0.5
LAMBDA = 0.9

def load_demo_image(image_path, image_size, device):  
    raw_image = Image.open(image_path).convert('RGB') 

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

def soften_mask(grid, ksize=5, iters=1, eps=1e-6):
    x = grid.float().unsqueeze(0).unsqueeze(0)
    for _ in range(iters):
        x = torch.nn.functional.avg_pool2d(x, kernel_size=ksize, stride=1, padding=ksize//2)
    x = x.squeeze(0).squeeze(0)
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return x

def grid_to_row(grid, cls_focus=1e-3):

    g = grid.to(dtype=torch.float32, device=device)
    flat = g.flatten()
    flat = flat / flat.sum()  
    patches = (1.0 - cls_focus) * flat

    cls_tok = torch.tensor([cls_focus], device=device, dtype=patches.dtype)
    row = torch.cat([cls_tok, patches])

    return row

def resolve_layer_indices(spec, total_layers):
    if spec == -1:
        candidates = list(range(total_layers))
    elif isinstance(spec, int):
        candidates = [spec]
    elif isinstance(spec, (list, tuple, set)):
        candidates = list(spec)
    else:
        raise TypeError(f'Unsupported layer spec: {spec!r}')
    normalized = []
    for idx in candidates:
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise ValueError(f'Layer index {idx} out of range for {total_layers} layers')
        if idx not in normalized:
            normalized.append(idx)
    return normalized

def make_forward(heads, override_rows, LAMBDA=LAMBDA):
    def new_forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=True):

        is_cross_attention = encoder_hidden_states is not None
        src = encoder_hidden_states
        attention_mask = encoder_attention_mask
        
        # ---- unchanged ------------------------
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(src))
        v = self.transpose_for_scores(self.value(src))

        # ---- new attention ---------------------------------------
        b, _, q_len, _ = q.shape
        k_len = k.size(-2)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if override_rows:
            if isinstance(heads, int):
                if heads < 0:
                    heads_idx = torch.arange(12, device=attention_scores.device)
                else:
                    heads_idx = torch.tensor([heads], device=attention_scores.device)
            else:
                heads_idx = torch.tensor(heads, device=attention_scores.device)
            
            
            if MODE == "pre":
                # Pre-softmax bias (soft & stable)
                for idx, grid in override_rows.items():
                    row = grid_to_row(grid) 
                    if row.numel() != k_len:
                        raise RuntimeError(f"Row length {row.numel()} != key length {k_len}")
                    # positive bias on ROI (+BETA*row), optional negative elsewhere (-GAMMA*(1-row))
                    bias = BETA * row - GAMMA * (1.0 - row)
                    attention_scores[:, heads_idx, idx, :] = attention_scores[:, heads_idx, idx, :] + bias

                attention_probs = nn.Softmax(dim=-1)(attention_scores)

            elif MODE == "post":
                # Post-softmax convex blend
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
                attention_probs = attention_probs.clone()
                for idx, grid in override_rows.items():
                    target = grid_to_row(grid)
                    current = attention_probs[:, heads_idx, idx, :]
                    attention_probs[:, heads_idx, idx, :] = (1 - LAMBDA) * current + LAMBDA * target 

        attention_probs.requires_grad_(True)

        # ------ unchanged --------------------------------------      
        self.save_attention = True
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients) 

        attention_probs_dropped = self.dropout(attention_probs) 
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        ctx = torch.matmul(attention_probs_dropped, v) 
        ctx = ctx.permute(0, 2, 1, 3).contiguous()    
        ctx = ctx.view(b, q_len, self.all_head_size) 

        outputs = (ctx, attention_probs) if output_attentions else (ctx,)
        outputs += ((k, v),)  
        return outputs
    return new_forward


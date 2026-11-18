from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F

"""Minimal implementation of MHA"""
class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask: Optional[Tensor]) ->  Tensor:

        B, N, C = x.size()
        q, k, v = self.qkv(x).view(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4).unbind(0)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False,)

        attn = attn.transpose(1,2).contiguous().view(B, N, C)
        out = self.proj(attn)
        return out


class AttentionBlock(nn.Module):

    def __init__(self, hidden_dim) -> None:
        super().__init__()


        self.hidden_dim = hidden_dim
        self.attn = Attention(hidden_dim)
        self.ffn = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:

        y = self.ln1(x)
        y = self.attn(y, mask)
        y = self.ln2(x + y)
        x = x + self.ffn(y)
        return x




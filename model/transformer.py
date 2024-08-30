import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, causal: bool, cross_attention: bool
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.cross_attention = cross_attention

        self.in_proj_weight = nn.Parameter(torch.rand((embed_dim * 3, embed_dim)))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value):
        if not self.cross_attention:
            key = query
            value = query

        q = F.linear(query, self.in_proj_weight[: self.embed_dim])
        k = F.linear(key, self.in_proj_weight[self.embed_dim : 2 * self.embed_dim])
        v = F.linear(value, self.in_proj_weight[2 * self.embed_dim :])
        q, k, v = [
            x.reshape(
                x.shape[0], x.shape[1], self.num_heads, x.shape[2] // self.num_heads
            )
            for x in [q, k, v]
        ]

        B, T, h, d = q.shape
        x = flash_attn_func(q, k, v, causal=self.causal)
        x = x.view(B, T, self.embed_dim)
        x = self.out_proj(x)
        return x, None


def create_sin_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    half_dim = dim // 2
    adim = torch.arange(half_dim, device=positions.device, dtype=positions.dtype).view(
        1, 1, -1
    )
    max_period_tensor = torch.full([], 10000, device=positions.device)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, ff_dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiheadAttention(
            embed_dim=dim, causal=True, cross_attention=False, num_heads=heads
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attention = MultiheadAttention(
            embed_dim=dim, causal=False, cross_attention=True, num_heads=heads
        )
        self.dropout_cross = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, ff_dim, bias=False)
        self.linear2 = nn.Linear(ff_dim, dim, bias=False)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, cross_attention_src):
        x_ = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x_, x_, x_)[0])

        x_ = self.norm_cross(x)
        x = x + self.dropout_cross(
            self.cross_attention(x_, cross_attention_src, cross_attention_src)[0]
        )

        x_ = self.norm2(x)
        x = x + self.dropout2(self.linear2(F.gelu(self.linear1(x_))))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 24,
        ff_dim: int = 4096,
        heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(dim, ff_dim, heads, dropout) for _ in range(depth)]
        )

    def pos_embed(self, x):
        B, T, dim = x.shape
        positions = (
            torch.arange(T, device=x.device).view(1, -1, 1).to(x.dtype).to(x.device)
        )
        x = x + create_sin_embedding(positions, dim)
        return x

    def forward(self, x, cross_attention_src):
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, cross_attention_src=cross_attention_src)
        return x

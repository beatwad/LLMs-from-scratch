from typing import Any, Dict

import torch
import torch.nn as nn
from constants import GPT_CONFIG_124M


class GPTModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.drp = nn.Dropout(cfg["drop_rate"])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, num_tokens = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(num_tokens, device=in_idx.device))
        x = tok_emb + pos_emb
        x = self.drp(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.ln1 = LayerNorm(cfg["emb_dim"])
        self.mha = MultiHeadAttention(cfg)
        self.drp = nn.Dropout(cfg["drop_rate"])
        self.ln2 = LayerNorm(cfg["emb_dim"])
        self.ffn = FeedForwardNetwork(cfg)

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        x = self.mha(x)
        x = self.drp(x)
        x = x + shortcut

        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.drp(x)
        x = x + shortcut
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.attn_dim = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.drop_rate = cfg["drop_rate"]
        self.context_len = cfg["context_length"]
        self.qkv_bias = cfg["qkv_bias"]
        self.flash = cfg["flash"]

        self.attn = nn.Linear(self.attn_dim, self.attn_dim * 3, bias=self.qkv_bias)
        self.dropout = nn.Dropout(self.drop_rate)
        self.out_proj = nn.Linear(
            self.attn_dim, self.attn_dim
        )  # Linear layer to combine head outputs

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.context_len, self.context_len)).view(
                1, 1, self.context_len, self.context_len
            ),
        )

    def forward(self, x):  # (batch_size, num_tokens)
        batch_size, num_tokens, _ = x.shape

        attn = self.attn(x)  # (batch_size, num_tokens, attn_dim * 3)
        attn = attn.view(batch_size, num_tokens, self.n_heads, self.head_dim * 3).transpose(
            2, 1
        )  # (batch_size, n_heads, num_tokens, head_dim * 3)
        Q, K, V = torch.split(
            attn, self.head_dim, dim=-1
        )  # (batch_size, num_tokens, n_heads, head_dim)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=self.drop_rate, is_causal=True
            )
        else:
            attn = (
                Q @ K.transpose(-2, -1) * self.head_dim**-0.5
            )  # (batch_size, n_heads, num_tokens, num_tokens)
            attn = attn.masked_fill((self.mask == 0)[:, :, :num_tokens, :num_tokens], float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ V

        out = out.transpose(1, 2)  # (batch_size, num_tokens, n_heads, head_dim)
        # ensure that a tensor's memory is stored in a contiguous block
        out = out.contiguous().view(
            batch_size, num_tokens, self.attn_dim
        )  # (batch_size, num_tokens, attn_dim)
        out = self.out_proj(out)

        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = 1e-8

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )

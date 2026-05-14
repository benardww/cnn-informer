import math
import torch
import torch.nn as nn


class ProbAttention(nn.Module):
    """
    ProbSparse 多头自注意力（来自 Informer 论文，AAAI 2021）。
    仅对 top-u 个查询计算全注意力，其余使用 mean(V) 作为上下文，
    从而将复杂度从 O(L²) 降至 O(L log L)。
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 8,
        factor: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(
        self,
        Q: torch.Tensor,  # [B, H, L_Q, d_k]
        K: torch.Tensor,  # [B, H, L_K, d_k]
        sample_k: int,
        n_top: int,
    ):
        """计算 M 稀疏性得分，返回 top-u 查询的注意力分数和索引。"""
        B, H, L_K, _ = K.shape
        _, _, L_Q, _ = Q.shape

        # 随机采样 sample_k 个键估计 M 得分
        idx_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        # K_sample: [B, H, L_Q, sample_k, d_k]
        K_sample = K.unsqueeze(2).expand(B, H, L_Q, L_K, self.d_k)
        K_sample = K_sample.gather(
            3,
            idx_sample.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            .expand(B, H, L_Q, sample_k, self.d_k),
        )  # [B, H, L_Q, sample_k, d_k]

        # [B, H, L_Q, sample_k]
        QK_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)
        ).squeeze(-2)

        # M = max - mean
        M = QK_sample.max(-1).values - QK_sample.mean(-1)  # [B, H, L_Q]
        top_idx = M.topk(n_top, dim=-1, sorted=False).indices  # [B, H, n_top]

        # 仅对 top-u 查询计算完整注意力分数
        Q_top = Q.gather(
            2,
            top_idx.unsqueeze(-1).expand(B, H, n_top, self.d_k),
        )  # [B, H, n_top, d_k]
        scores_top = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [B, H, n_top, L_K]
        return scores_top, top_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model] → [B, L, d_model]"""
        B, L, d_model = x.shape
        H, d_k = self.n_heads, self.d_k

        Q = self.W_q(x).view(B, L, H, d_k).transpose(1, 2)  # [B,H,L,d_k]
        K = self.W_k(x).view(B, L, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, d_k).transpose(1, 2)

        sample_k = min(self.factor * math.ceil(math.log(L)), L)
        n_top    = min(self.factor * math.ceil(math.log(L)), L)

        # 初始化上下文为 mean(V)
        context = V.mean(dim=2, keepdim=True).expand_as(V).clone()  # [B,H,L,d_k]

        scores_top, top_idx = self._prob_QK(Q, K, sample_k, n_top)
        attn_top = self.dropout(torch.softmax(scores_top, dim=-1))   # [B,H,n_top,L]
        v_top = torch.matmul(attn_top, V)                            # [B,H,n_top,d_k]

        # 将 top-u 结果写回上下文
        context.scatter_(
            2,
            top_idx.unsqueeze(-1).expand(B, H, n_top, d_k),
            v_top,
        )

        out = context.transpose(1, 2).contiguous().view(B, L, d_model)
        return self.W_o(out)

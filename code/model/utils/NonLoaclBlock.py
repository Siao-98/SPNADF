# lcwa_ura_gate_head_in_sdpa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable
from timm.models.layers import DropPath, Mlp, trunc_normal_


# -----------------------------
# 分块 / 反分块
# -----------------------------
def window_partition(x: torch.Tensor, window_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
    B, C, H, W = x.shape
    Wh, Ww = window_size
    x = x.view(B, C, H // Wh, Wh, W // Ww, Ww)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, Wh, Ww, C)
    return windows

def window_reverse(windows: torch.Tensor, original_size: Tuple[int, int], window_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
    H, W = original_size
    Wh, Ww = window_size
    num_windows_per_img = (H // Wh) * (W // Ww)
    B = windows.shape[0] // num_windows_per_img
    C = windows.shape[-1]
    x = windows.view(B, H // Wh, W // Ww, Wh, Ww, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
    return x

def grid_partition(x: torch.Tensor, grid_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
    B, C, H, W = x.shape
    Gh, Gw = grid_size
    x = x.view(B, C, Gh, H // Gh, Gw, W // Gw)
    grids = x.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, Gh, Gw, C)
    return grids

def grid_reverse(grids: torch.Tensor, original_size: Tuple[int, int], grid_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
    H, W = original_size
    Gh, Gw = grid_size
    C = grids.shape[-1]
    num_grids_per_img = (H // Gh) * (W // Gw)
    B = grids.shape[0] // num_grids_per_img
    x = grids.view(B, H // Gh, W // Gw, Gh, Gw, C)
    x = x.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return x


# -----------------------------
# 相对位置 index
# -----------------------------
def get_relative_position_index(win_h: int, win_w: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index


# -----------------------------
# URA 自注意力：u_temp 控温；u_bias 列偏置；gate 在 SDPA 之后（按 head）
# -----------------------------
class RelativeSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, win_size=(7, 7),
                 attn_drop: float = 0., proj_drop: float = 0.,
                 lambda_k: float = 1.0,       # Key 抑制强度
                 temp_alpha: float = 1.0,     # 兼容保留（未直接用）
                 temp_max: float = 4.0):      # 温度上限
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.head_dim = dim // num_heads
        self.lambda_k = lambda_k
        self.temp_alpha = temp_alpha
        self.temp_max = temp_max

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_p = attn_drop

        Wh, Ww = win_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )
        self.register_buffer("relative_position_index", get_relative_position_index(Wh, Ww), persistent=False)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(
        self,
        x: torch.Tensor,              # [B', N, C]
        token_u_temp: torch.Tensor,   # [B', N] (raw)
        token_u_bias: torch.Tensor,   # [B', N] (raw)
        token_gate: torch.Tensor      # [B', H, N] in (g_min,1]
    ) -> torch.Tensor:
        B_, N, C = x.shape

        # qkv
        qkv = self.qkv(x).view(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B', H, N, Dh]

        # 行温度：q_i <- q_i / T(uT(i))，T(u)=temp_max*sigmoid(u)
        uT = token_u_temp
        T = (0.1+self.temp_max * torch.sigmoid(uT)).view(B_, 1, N, 1)   # [B',1,N,1]
        q = q / T

        # 相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)              # [1, H, N, N]
        attn_mask = relative_position_bias

        # 列偏置：-lambda_k * relu(uB_j)
        uB = torch.relu(token_u_bias).view(B_, 1, 1, N)              # [B',1,1,N]
        attn_mask = attn_mask + (-self.lambda_k) * uB                # 广播到 [B', H, N, N]

        # SDPA
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )  # [B', H, N, Dh]

        # ★ SDPA 之后按 head 的门控
        # token_gate: [B', H, N] -> [B', H, N, 1]
        g = token_gate.view(B_, self.num_heads, N, 1)
        x = x * g

        # 输出整形 + 投影
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------
# 学习到的按 head 残差门控（per-token, per-head）
# -----------------------------
class GateHeadMLP(nn.Module):
    """
    输入: [B', N, C] 的特征 + [B', N] 的 uT  ->  [B', N, H]  (再映射到 (g_min,1])
    说明: 先在通道维聚合，再为每个 head 预测一个 gate。
    """
    def __init__(self, in_dim: int, num_heads: int, hidden_ratio: float = 0.5, g_min: float = 0.0):
        super().__init__()
        h = max(4, int(in_dim * hidden_ratio))
        self.fc1 = nn.Linear(in_dim, h, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, num_heads, bias=True)  # 输出每 token 的 H 个门
        self.g_min = g_min

    def forward(self, feat: torch.Tensor, uT: torch.Tensor) -> torch.Tensor:
        # feat: [B', N, C], uT: [B', N]
        x = torch.cat([feat, uT.unsqueeze(-1)], dim=-1)   # [B', N, C+1]
        g = torch.sigmoid(self.fc2(self.act(self.fc1(x))))  # [B', N, H] in (0,1)
        # 映射到 (g_min, 1]
        return self.g_min + (1.0 - self.g_min) * g        # [B', N, H]


# -----------------------------
# MaxViT 块：u_temp 控温；按 head 学习门控（门在 SDPA 后乘）；u_bias 列偏置
# -----------------------------
class MaxViTTransformerBlock(nn.Module):
    def __init__(self, dim: int,
                 partition_function: Callable,
                 reverse_function: Callable,
                 win_size: Tuple[int, int],
                 num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 attn_drop: float = 0.,
                 drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 lambda_k: float = 1.0,
                 temp_alpha: float = 1.0,
                 temp_max: float = 4.0,
                 gate_hidden_ratio: float = 0.5,
                 gate_gmin: float = 0.0):
        super().__init__()
        self.partition_function = partition_function
        self.reverse_function = reverse_function
        self.win_size = win_size
        self.num_heads = num_heads

        self.norm1 = norm_layer(dim)
        self.attn = RelativeSelfAttention(
            dim=dim, num_heads=num_heads, win_size=win_size,
            attn_drop=attn_drop, proj_drop=drop,
            lambda_k=lambda_k, temp_alpha=temp_alpha, temp_max=temp_max
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # 按 head 学习门控：输入维 (C+1) -> H
        self.gate_mlp = GateHeadMLP(in_dim=dim + 1, num_heads=num_heads,
                                    hidden_ratio=gate_hidden_ratio, g_min=gate_gmin)

    def forward(self,
                x: torch.Tensor,        # [B, C, H, W]
                u_temp: torch.Tensor,   # [B, 1, H, W]  用于温度与门控
                u_bias: torch.Tensor    # [B, 1, H, W]  用于列偏置
                ) -> torch.Tensor:
        B, C, H, W = x.shape
        Wh, Ww = self.win_size

        # 分块（特征与两路不确定度）
        x_part  = self.partition_function(x,       self.win_size).view(-1, Wh * Ww, C)   # [B', N, C]
        uT_part = self.partition_function(u_temp,  self.win_size).view(-1, Wh * Ww)      # [B', N]
        uB_part = self.partition_function(u_bias,  self.win_size).view(-1, Wh * Ww)      # [B', N]

        # 注意：门使用“注意力输入特征” → LN(x_part)
        x_norm = self.norm1(x_part)                       # [B', N, C]

        # 学习到的按 head 门（用于 SDPA 之后）：[B', N, H] -> [B', H, N]
        gate = self.gate_mlp(x_norm, uT_part).permute(0, 2, 1).contiguous()  # [B', H, N]

        # 自注意力（行温度: uT_part；列偏置: uB_part；SDPA 后乘 head 门）
        x_attn = self.attn(
            x_norm,
            token_u_temp=uT_part,
            token_u_bias=uB_part,
            token_gate=gate                       # [B', H, N]
        )  # [B', N, C]

        # 残差（此处不再乘 gate，避免重复）
        x_part = x_part + self.drop_path(x_attn)

        # MLP 残差（保持标准形式）
        x_mlp = self.mlp(self.norm2(x_part))
        x_part = x_part + self.drop_path(x_mlp)

        # 反分块
        x_out = self.reverse_function(x_part, (H, W), self.win_size)
        return x_out


# -----------------------------
# 简单残差卷积块
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, act_layer=nn.GELU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            act_layer(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# -----------------------------
# UA：两路输出（u_temp, u_bias）— raw 输出
# -----------------------------
class UncertaintyAdapter(nn.Module):
    """
    输入外部不确定图 -> 输出两路 raw：
      u_temp (raw)  → T = temp_max * sigmoid(u_temp)
      u_bias (raw)  → 列偏置使用 relu(u_bias)
    """
    def __init__(self, in_ch=1, mid=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=True),
            nn.GELU(),
        )
        self.head_temp = nn.Conv2d(mid, 1, 3, padding=1, bias=True)
        self.head_bias = nn.Conv2d(mid, 1, 3, padding=1, bias=True)

    def forward(self, u: torch.Tensor):
        h = self.enc(u)
        u_temp = self.head_temp(h)  # [B,1,H,W] raw
        u_bias = self.head_bias(h)  # [B,1,H,W] raw
        return u_temp, u_bias


# -----------------------------
# LCWA with URA + UA（head-specific gate in SDPA）
# -----------------------------
class LocalCrossWindowAttention(nn.Module):
    """
    - UA 输出 (u_temp, u_bias)
    - Key 抑制用 u_bias；Query 温度与门控用 u_temp
    - 两路注意力：窗口 + （可选）Grid
    - 门控为按 head 的逐 token 乘性调制（在 SDPA 之后）
    """
    def __init__(self, n_feat: int, r: int,
                 num_heads: int = 8, mlp_ratio: float = 4.,
                 attn_drop: float = 0., drop: float = 0., drop_path: float = 0.,
                 bottleneck_ratio: int = 2, use_grid_attention: bool = True,
                 lambda_k: float = 1.0, temp_alpha: float = 1.0, temp_max: float = 4.0,
                 gate_hidden_ratio: float = 0.5, gate_gmin: float = 0.0,
                 ua_in_ch: int = 1, ua_mid: int = 8):
        super().__init__()
        self.r = r
        self.use_grid_attention = use_grid_attention

        bottleneck_dim = max(1, n_feat // bottleneck_ratio)
        self.reduce = nn.Conv2d(n_feat, bottleneck_dim, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(bottleneck_dim, n_feat, kernel_size=1, bias=True)

        # UA（固定启用）
        self.ua = UncertaintyAdapter(in_ch=ua_in_ch, mid=ua_mid)

        # 窗口注意力
        self.window_attention = MaxViTTransformerBlock(
            dim=bottleneck_dim, partition_function=window_partition, reverse_function=window_reverse,
            win_size=(r, r), num_heads=num_heads, mlp_ratio=mlp_ratio,
            attn_drop=attn_drop, drop=drop, drop_path=drop_path,
            lambda_k=lambda_k, temp_alpha=temp_alpha, temp_max=temp_max,
            gate_hidden_ratio=gate_hidden_ratio, gate_gmin=gate_gmin
        )

        # Grid 注意力（可选）
        if use_grid_attention:
            self.grid_attention = MaxViTTransformerBlock(
                dim=bottleneck_dim, partition_function=grid_partition, reverse_function=grid_reverse,
                win_size=(r, r), num_heads=num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop, drop=drop, drop_path=drop_path,
                lambda_k=lambda_k, temp_alpha=temp_alpha, temp_max=temp_max,
                gate_hidden_ratio=gate_hidden_ratio, gate_gmin=gate_gmin
            )
        else:
            self.grid_attention = None

        num_paths = 2 + int(use_grid_attention)
        self.res_blk = ResidualBlock(num_paths * bottleneck_dim, act_layer=nn.GELU)
        self.conv = nn.Conv2d(num_paths * bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=True)

        nn.init.constant_(self.expand.weight, 0)
        if self.expand.bias is not None:
            nn.init.constant_(self.expand.bias, 0)

    def forward(self, x: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        uncertainty: [B, 1, H, W]（必须提供）
        """
        B, C, H, W = x.shape
        # assert H >= self.r and W >= self.r, "feature spatial size should be >= window size r"

        # 通道压缩 & 尺度对齐到 r 的倍数
        x_b = self.reduce(x)

        r = self.r
        # 小于 r：上采样到 r；否则走原逻辑（floor 到 r 的倍数）
        if H < r or W < r:
            H_pad = max(H, r)
            W_pad = max(W, r)
        else:
            H_pad = (H // r) * r
            W_pad = (W // r) * r

        # 兜底，避免意外为 0
        H_pad = max(H_pad, r)
        W_pad = max(W_pad, r)

        x_resized = F.interpolate(x_b, size=(H_pad, W_pad), mode="bilinear", align_corners=False)
        # x_b = self.reduce(x)
        # H_pad, W_pad = (H // self.r) * self.r, (W // self.r) * self.r
        # x_resized = F.interpolate(x_b, size=(H_pad, W_pad), mode='bilinear', align_corners=False)


        # UA -> 两路不确定度（对齐到同尺度）
        u_resized = F.interpolate(uncertainty, size=(H_pad, W_pad), mode='bilinear', align_corners=False)
        u_temp, u_bias = self.ua(u_resized)             # raw maps, [B,1,H_pad,W_pad] ×2

        # 窗口注意力（门在 SDPA 之后按 head 应用）
        x_local = self.window_attention(x_resized, u_temp=u_temp, u_bias=u_bias)
        paths = [x_resized, x_local]

        # Grid 注意力（若启用）
        if self.use_grid_attention:
            x_grid = self.grid_attention(x_resized, u_temp=u_temp, u_bias=u_bias)
            paths.append(x_grid)

        # 融合到原尺度
        paths = [F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False) for p in paths]
        feat = torch.cat(paths, dim=1)
        feat = self.res_blk(feat)
        feat = self.conv(feat)

        out = x + self.expand(feat)
        return out

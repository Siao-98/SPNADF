import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.NonLoaclBlock import LocalCrossWindowAttention
from typing import List

try:
    from mmcv.ops import ModulatedDeformConv2d
    MMCV_AVAILABLE = True
except ImportError:
    MMCV_AVAILABLE = False
    print("Warning: mmcv.ops.ModulatedDeformConv2d not found. DCN alignment will be disabled.")


class ConvLayer(nn.Module):
    """
    卷积 + (可选)Norm + (可选)ReLU
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm: str = None,
                 activation: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            padding_mode="reflect",
                            bias=(norm is None))]

        if norm is not None:
            norm = norm.lower()
            if norm == 'batch':
                layers.append(nn.BatchNorm2d(out_channels))
            elif norm == 'instance':
                layers.append(nn.InstanceNorm2d(out_channels))
            elif norm == 'group':
                # 选一个不超过 8 且能整除 out_channels 的 group 数
                num_groups = min(8, out_channels)
                while out_channels % num_groups != 0:
                    num_groups -= 1
                layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
            else:
                raise ValueError(f"Unsupported norm type: {norm}")

        if activation:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    ResNet-style 残差块 + CBAM:
        Conv-ReLU-Conv -> +identity -> CBAM(channel+spatial) -> ReLU
    通道数不变
    """
    def __init__(self, in_channels, out_channels,
                 norm: str = None,
                 reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        assert in_channels == out_channels, "ResidualBlock 要求输入输出通道一致"

        self.conv1 = ConvLayer(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1,
                               norm=norm, activation=True)
        # 第二个 conv 不加 ReLU，在残差 + CBAM 之后统一 ReLU
        self.conv2 = ConvLayer(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1,
                               norm=norm, activation=False)
        self.relu = nn.ReLU(inplace=True)

        # ===== CBAM: Channel Attention =====
        self.ca_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(out_channels // reduction, 1)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
        )

        # ===== CBAM: Spatial Attention =====
        padding = (spatial_kernel - 1) // 2
        self.sa_conv = nn.Conv2d(2, 1,
                                 kernel_size=spatial_kernel,
                                 padding=padding,
                                 bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity        # 残差相加，做为 CBAM 的输入
        identity2 = out
        # ----- Channel Attention -----
        avg_pool = self.ca_avg_pool(out)
        max_pool = self.ca_max_pool(out)
        ca_avg = self.ca_mlp(avg_pool)
        ca_max = self.ca_mlp(max_pool)
        ca = self.sigmoid(ca_avg + ca_max)   # (B,C,1,1)
        out = out * ca

        # ----- Spatial Attention -----
        sa_avg = torch.mean(out, dim=1, keepdim=True)      # (B,1,H,W)
        sa_max, _ = torch.max(out, dim=1, keepdim=True)    # (B,1,H,W)
        sa = torch.cat([sa_avg, sa_max], dim=1)            # (B,2,H,W)
        sa = self.sigmoid(self.sa_conv(sa))                # (B,1,H,W)
        out = out * sa

        # 统一 ReLU
        out = self.relu(identity2+out)
        return out


# ======================
# ConvLSTM / ConvGRU Cell
# ======================

class ConvLSTMCell(nn.Module):
    """
    单步 ConvLSTM，输入/隐藏都是 4D: (N, C, H, W)
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True)

    def forward(self, x, state):
        """
        x: (N, C_in, H, W)
        state: (h, c)，如果为 None 则按 x 形状初始化为 0
        """
        if state is None:
            size_h = [x.size(0), self.hidden_channels, x.size(2), x.size(3)]
            h = x.new_zeros(size_h)
            c = x.new_zeros(size_h)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)  # (N, C_in + C_h, H, W)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class ConvGRUCell(nn.Module):
    """
    单步 ConvGRU，输入/隐藏都是 4D
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.conv_zr = nn.Conv2d(input_channels + hidden_channels,
                                 2 * hidden_channels,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 bias=True)
        self.conv_h = nn.Conv2d(input_channels + hidden_channels,
                                hidden_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=True)

    def forward(self, x, h_prev):
        """
        x: (N, C_in, H, W)
        h_prev: (N, C_h, H, W) 或 None
        """
        if h_prev is None:
            h_prev = x.new_zeros(x.size(0), self.hidden_channels,
                                 x.size(2), x.size(3))

        combined = torch.cat([x, h_prev], dim=1)
        zr = self.conv_zr(combined)
        z, r = torch.chunk(zr, 2, dim=1)

        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        combined_h = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_h))

        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next


# ==============================================================================
# HiddenStateAligner: 只使用 DCN 自己预测的 offset（无金字塔先验）
# ==============================================================================
class HiddenStateAligner(nn.Module):
    """
    隐状态对齐器：
    使用当前帧特征 x_t 与上一帧隐状态 h_prev 作为 guide，直接预测 DCN offset + mask，
    不再依赖多尺度 offset 先验。
    """

    def __init__(self, feature_channels, hidden_channels, kernel_size=3, deform_groups=4,
                 max_residue_limit=3.0):
        super().__init__()
        if not MMCV_AVAILABLE:
            raise ImportError("mmcv is not available.")

        self.deform_groups = deform_groups
        self.kernel_size = kernel_size
        self.max_residue_limit = float(max_residue_limit)

        guide_channels = feature_channels + hidden_channels
        out_channels = deform_groups * 3 * kernel_size * kernel_size

        self.conv_offset_mask_align = nn.Conv2d(
            guide_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.dcn = ModulatedDeformConv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            deform_groups=deform_groups
        )

        nn.init.zeros_(self.conv_offset_mask_align.weight)
        nn.init.zeros_(self.conv_offset_mask_align.bias)

    def forward(self, x_t, h_prev):
        if h_prev is None:
            return None

        guide = torch.cat([x_t, h_prev], dim=1)
        out = self.conv_offset_mask_align(guide)

        k = self.kernel_size
        G = self.deform_groups
        offset_channels = 2 * k * k * G

        delta_o = out[:, :offset_channels, :, :]
        mask_raw = out[:, offset_channels:, :, :]

        # 限制 offset 幅度
        delta_o = torch.tanh(delta_o) * self.max_residue_limit

        offset = delta_o
        mask = 2.0 * torch.sigmoid(mask_raw)

        h_aligned = self.dcn(h_prev, offset, mask)

        return  h_aligned


class RefreshBlock(nn.Module):
    """
    G(h, x): 用当前输入特征 x 对递归输出 h 做残差纠偏
    """
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, 1, 1),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        delta = self.fuse(torch.cat([h, x], dim=1))
        return F.relu(h + delta, inplace=True)



# ==============================================================================
# RecurrentConvLayer: 集成 DCN 对齐（无多尺度金字塔）
# ==============================================================================
class RecurrentConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=2,
                 padding=2,
                 recurrent_block_type: str = 'convlstm',
                 norm: str = None,
                 downsample_mode: str = 'conv',
                 use_dcn_align: bool = False,
                 dcn_deform_groups: int = 4,
                 dcn_max_offset: float = 3.0,
                 refresh_interval: int = 5,  # 默认 5 帧刷新
                 refresh_hidden: int = 64,
                 ):
        super().__init__()

        assert downsample_mode in ['conv', 'avgpool']
        self.downsample_mode = downsample_mode
        conv_stride = 1 if downsample_mode == 'avgpool' else stride

        self.conv = ConvLayer(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=conv_stride,
                              padding=padding,
                              norm=norm,
                              activation=True)

        if downsample_mode == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

        rtype = recurrent_block_type.lower()
        self.is_lstm = (rtype == 'convlstm')

        if self.is_lstm:
            self.cell = ConvLSTMCell(out_channels, out_channels, kernel_size=3, padding=1)
        elif rtype == 'convgru':
            self.cell = ConvGRUCell(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            raise ValueError(f"Unsupported recurrent_block_type: {recurrent_block_type}")

        # DCN 对齐
        self.use_dcn_align = use_dcn_align and MMCV_AVAILABLE
        if self.use_dcn_align:
            self.aligner = HiddenStateAligner(
                feature_channels=out_channels,
                hidden_channels=out_channels,
                kernel_size=3,
                deform_groups=dcn_deform_groups,
                max_residue_limit=dcn_max_offset
            )
        else:
            self.aligner = None

        self.refresh_interval = int(refresh_interval)
        self.refresh_block = RefreshBlock(out_channels, hidden=refresh_hidden) if self.refresh_interval > 0 else None
    def forward(self, x, prev_state, step_idx: int = None):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)

        h_prev = None
        c_prev = None
        if prev_state is not None:
            if self.is_lstm:
                h_prev, c_prev = prev_state
            else:
                h_prev = prev_state

        # 仅使用 DCN 对齐上一帧隐状态
        if self.use_dcn_align and h_prev is not None:
            h_aligned = self.aligner(x, h_prev)
        else:
            h_aligned = h_prev

        # ---- RNN step ----
        if self.is_lstm:
            state_to_rnn = (h_aligned, c_prev) if h_aligned is not None else None
            h_new, (h_new, c_new) = self.cell(x, state_to_rnn)  # 显式得到 c_new
        else:
            h_new = self.cell(x, h_aligned)
            c_new = None

        # ---- refresh on keyframes ----
        do_refresh = (
                self.refresh_block is not None
                and step_idx is not None
                and (step_idx % self.refresh_interval == 0)
                and (step_idx > 0)  # 推荐：不在 t=0 刷新；想 t=0 也刷新就删掉
        )

        if do_refresh:
            h_ref = self.refresh_block(h_new, x)
            h_new = h_ref
            if self.is_lstm:
                state_new = (h_ref, c_new)  # 只刷新 h，c 保持
            else:
                state_new = h_ref
        else:
            if self.is_lstm:
                state_new = (h_new, c_new)
            else:
                state_new = h_new

        return h_new, state_new


class UpsampleConvLayer(nn.Module):
    """
    上采样 ×2 (bilinear) + ConvLayer
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 padding=2,
                 norm: str = None):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              norm=norm,
                              activation=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class CrossModalBlock(nn.Module):
    """
    双分支同尺度特征交互模块
    """
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.delta = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feat_p, feat_t):
        x = torch.cat([feat_p, feat_t], dim=1)  # (N, 2C, H, W)
        j = self.fuse(x)                        # (N, C, H, W)
        delta = self.delta(j)                  # (N, 2C, H, W)
        delta_p, delta_t = torch.chunk(delta, 2, dim=1)
        out_p = feat_p + delta_p
        out_t = feat_t + delta_t
        return out_p, out_t


class UncGuidedModulatedDCNBlock(nn.Module):
    """
    Uncertainty-Guided Deformable Residual Block (基于 mmcv 的 ModulatedDeformConv2d)
    输入:
        x : (B, C, H, W)   特征
        U : (B, 1, H, W)   不确定度 (建议事先归一化到 [0,1]，越大越不确定)
    输出:
        out : (B, C, H, W)  修复后的特征
    """
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 deform_groups: int = 1,
                 tau: float = 0.5,    # gate 阈值, 作用在 [0,1] 的 U 上
                 beta: float = 10.0   # gate 斜率
                 ):
        super().__init__()

        self.tau = tau
        self.beta = beta

        kH = kW = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups

        guide_channels = channels + 1   # [x, U] 拼起来当作 guide

        # ---- 预测 offset + mask 的 1 个 conv ----
        self.conv_offset_mask = nn.Conv2d(
            guide_channels,
            deform_groups * 3 * kH * kW,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        # ---- mmcv 的 ModulatedDeformConv2d 主体 ----
        self.dcn = ModulatedDeformConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
            bias=True
        )

        # DCN 之后再接两层普通 conv 做残差
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        # ---- 初始化 ----
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

        nn.init.kaiming_normal_(self.dcn.weight, mode='fan_out', nonlinearity='relu')
        if self.dcn.bias is not None:
            nn.init.zeros_(self.dcn.bias)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, U):
        """
        x: (B, C, H, W)
        U: (B, 1, H, W)  建议已经缩放到 [0,1]
        """
        U = torch.clamp(U, 0.0, 1.0)

        B, C, H, W = x.shape
        kH = kW = self.dcn.kernel_size[0]

        # 1) 用 [x, U] 预测 offset + mask
        guide = torch.cat([x, U], dim=1)  # (B, C+1, H, W)
        out = self.conv_offset_mask(guide)  # (B, deform_groups*3*kH*kW, H, W)

        o_channels = self.deform_groups * 2 * kH * kW
        offset = out[:, :o_channels, :, :]
        mask   = out[:, o_channels:, :, :]

        # mask ∈ (0, 2)
        mask = 2.0 * torch.sigmoid(mask)

        # 2) mmcv ModulatedDeformConv2d
        z = self.dcn(x, offset, mask)  # (B, C, H, W)

        # 3) 残差分支
        r = self.relu(self.conv1(z))
        r = self.conv2(r)

        # 4) gate：只对高不确定像素做强修复
        gate = torch.sigmoid(self.beta * (U - self.tau))   # (B,1,H,W)
        r = r * gate                                       # 广播到 C 通道

        # 5) 残差相加
        out = x + r
        out = self.relu(out)
        return out


class UncertaintyGatedPhotonHead(nn.Module):
    """
    不确定度引导的门控光子头（当前在 DualUNetRecurrent 中未直接调用，可按需接入）。
    Input: (B, 3, H, W)
        - Ch0: Raw noisy photon counts (0/1)
        - Ch1: Signal probability
        - Ch2: Uncertainty map
    Output: (B, base_num_channels, H, W) Denoised features
    """
    def __init__(self, base_num_channels, norm=None):
        super().__init__()

        # 1. 特征分支：处理原始含噪数据 (Channel 0)
        self.raw_feature_conv = nn.Sequential(
            nn.Conv2d(1, base_num_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # 2. 门控分支：处理引导信息 (Channels 1 & 2)
        gate_mid_channels = base_num_channels // 2
        self.gate_generator = nn.Sequential(
            nn.Conv2d(2, gate_mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_mid_channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

        # 3. 后处理卷积
        self.post_gate_conv = ConvLayer(
            base_num_channels,
            base_num_channels,
            kernel_size=5, stride=1, padding=2,
            norm=norm, activation=True
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x_raw = x[:, 0:1, :, :]       # (B, 1, H, W)
        x_guide = x[:, 1:3, :, :]     # (B, 2, H, W)

        raw_feats = self.raw_feature_conv(x_raw) # (B, C, H, W)
        gate_mask = self.gate_generator(x_guide) # (B, 1, H, W)

        gated_feats = raw_feats * gate_mask      # (B, C, H, W)
        out = self.post_gate_conv(gated_feats)   # (B, C, H, W)
        return out


class AttentionSkipFusion(nn.Module):
    """
    Attention Gate Fusion
    - 用注意力过滤 Encoder 特征，再与 Decoder 特征做残差式融合。
    """
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        inter_channels = in_channels // reduction

        # 1. W_x: 处理 Encoder 特征
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(inter_channels, affine=True)
        )

        # 2. W_g: 处理 Decoder 特征 (Gating Signal)
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(inter_channels, affine=True)
        )

        # 3. 生成 Attention Map
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

        # 4. 最终融合层
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_dec, x_enc):
        """
        x_dec: Decoder feature (cleaner)
        x_enc: Encoder feature (noisy)
        """
        g_x = self.W_x(x_enc)
        g_g = self.W_g(x_dec)

        act = F.relu(g_x + g_g)
        attn = self.psi(act)

        x_enc_weighted = x_enc * attn
        out = x_dec + x_enc_weighted
        out = self.final_conv(out)
        return out

class URCAFuseHead(nn.Module):
    """
    URCA-FuseHead (bidirectional, non-shared projections) + 轻量不确定度校准
    - 路由默认 & 固定：w_in = cat([F_E, F_P], dim=1)
    - P→E: Q_E, K_P, V_P   ;  E→P: Q_P, K_E, V_E
    - logits 加 log(m_cal+eps) 与可学习的空间核偏置
    输入: photon_tensor = cat([P, E, U], 1)
      P: (B, p_ch, H, W), E: (B, e_ch, H, W), U: (B,1,H,W)∈[0,1]
    输出: (B, out_channels, H, W)
    """
    def __init__(self,
                 p_ch: int=1,
                 e_ch: int=2,
                 out_channels: int=32,
                 c_mid: int = 32,
                 c_proj: int = 16,
                 ksize: int = 5):
        super().__init__()
        self.p_ch, self.e_ch = p_ch, e_ch
        self.ksize = ksize
        pad = ksize // 2
        self.eps = 1e-6

        # ——— 两分支浅编码（不共享）———
        def enc(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_c, out_c, 1, 1, 0, bias=True),
                nn.SiLU(inplace=True),
            )
        self.encE = enc(e_ch, c_mid)        # 测量分支
        self.encP = enc(p_ch + 1, c_mid)    # 先验分支，输入 [P,U]

        # ——— 轻量不确定度校准 m_cal(U) ∈ (0,1) ———
        self.m_calib = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 3, 1, 1), nn.Sigmoid()
        )

        # ——— P→E: Q_E, K_P, V_P（独立参数）———
        self.q_E = nn.Conv2d(c_mid, c_proj, 1, 1, 0, bias=False)
        self.k_P = nn.Conv2d(c_mid, c_proj, 1, 1, 0, bias=False)
        self.v_P = nn.Conv2d(c_mid, c_proj, 1, 1, 0, bias=False)

        # ——— E→P: Q_P, K_E, V_E（独立参数）———
        self.q_P = nn.Conv2d(c_mid, c_proj, 1, 1, 0, bias=False)
        self.k_E = nn.Conv2d(c_mid, c_proj, 1, 1, 0, bias=False)
        self.v_E = nn.Conv2d(c_mid, c_proj, 1, 1, 0, bias=False)

        # ——— 两个方向的空间核偏置（不共享）———
        K2 = ksize * ksize
        self.spatial_bias_PE = nn.Parameter(torch.zeros(K2))  # P→E
        self.spatial_bias_EP = nn.Parameter(torch.zeros(K2))  # E→P

        # ——— 路由器（固定用特征输入）———
        r_in = 2 * c_mid  # cat([F_E, F_P])
        self.router = nn.Sequential(
            nn.Conv2d(r_in, 32, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1, 1, 0), nn.Sigmoid()
        )
        with torch.no_grad():
            self.router[-2].bias.zero_()

            # ——— 输出整合 + 测量线性残差 ———
        self.out_conv = nn.Conv2d(c_proj, out_channels, 1, 1, 0)
        self.proj_E  = nn.Conv2d(e_ch, out_channels, 1, 1, 0)

        # ——— unfold for local windows ———
        self.unfold = nn.Unfold(kernel_size=ksize, padding=pad)

    @staticmethod
    def _local_attend(Q, K_unf, V_unf, m_unf, spatial_bias, eps):
        """
        Q:     (B, C', H, W)
        K_unf: (B, C'*K2, H*W)
        V_unf: (B, C'*K2, H*W)
        m_unf: (B, K2,    H*W) —— 校准后的可信度块
        """
        B, Cp, H, W = Q.shape
        HW = H * W
        K2 = m_unf.shape[1]

        Qf = Q.view(B, Cp, HW).permute(0, 2, 1)               # (B, HW, C')
        Kf = K_unf.view(B, Cp, K2, HW).permute(0, 3, 1, 2)    # (B, HW, C', K2)
        Vf = V_unf.view(B, Cp, K2, HW).permute(0, 3, 1, 2)    # (B, HW, C', K2)
        mf = m_unf.permute(0, 2, 1)                           # (B, HW, K2)

        # 点积打分 + log(m_cal) + 空间核
        scores = (Qf.unsqueeze(-1) * Kf).sum(dim=2) / (Cp ** 0.5)     # (B, HW, K2)
        scores = scores + mf.clamp_min(eps).log() + spatial_bias.view(1, 1, K2)

        A = torch.softmax(scores, dim=-1)                              # (B, HW, K2)
        Y = (Vf * A.unsqueeze(2)).sum(dim=-1)                          # (B, HW, C')
        return Y.permute(0, 2, 1).contiguous().view(B, Cp, H, W)       # (B, C', H, W)

    def forward(self, photon_tensor: torch.Tensor):
        """
        photon_tensor = cat([P, E, U], 1)
        """
        B, C, H, W = photon_tensor.shape
        assert C == self.p_ch + self.e_ch + 1

        # 拆分
        P = photon_tensor[:, :self.p_ch]
        E = photon_tensor[:, self.p_ch:self.p_ch + self.e_ch]
        U = photon_tensor[:, -1:].clamp(0, 1)

        # 可信度 m = 1 - U，经轻量卷积校准
        m = 1.0 /(U+1e-3)
        m_cal = self.m_calib(m)               # (B,1,H,W)
        m_unf  = self.unfold(m_cal)           # (B, K2, H*W)

        # 编码
        F_E = self.encE(E)                     # (B, Cmid, H, W)
        F_P = self.encP(torch.cat([P, m_cal], 1))  # (B, Cmid, H, W)

        # ---------- P→E ----------
        Q_PE = self.q_E(F_E)
        K_PE = self.k_P(F_P)
        V_PE = self.v_P(F_P)
        F_EP = self._local_attend(Q_PE, self.unfold(K_PE), self.unfold(V_PE),
                                  m_unf, self.spatial_bias_PE, self.eps)

        # ---------- E→P ----------
        Q_EP = self.q_P(F_P)
        K_EP = self.k_E(F_E)
        V_EP = self.v_E(F_E)
        F_PE = self._local_attend(Q_EP, self.unfold(K_EP), self.unfold(V_EP),
                                  m_unf, self.spatial_bias_EP, self.eps)

        # 路由（固定用特征）：w = σ(CNN([F_E, F_P]))
        w_in = torch.cat([F_E, F_P], dim=1)
        w = self.router(w_in)                 # (B,1,H,W)

        F_mix = w * F_EP + (1.0 - w) * F_PE
        out = self.out_conv(F_mix) + self.proj_E(E)   # 线性残差
        return out


class DualUNetRecurrent(nn.Module):
    """
    双分支循环 UNet:
    - photon 分支: 处理光子计数，输出强度特征
    - tof 分支:    处理 ToF 特征，输出深度特征

    Encoder: 每层 ConvRNN (支持 DCN 对齐) + CrossModalBlock 交互；
    Decoder: AttentionSkipFusion + ResidualBlock + LCWA + Upsample
    full-res tail: skip + ResidualBlock + LCWA + 1×1 Conv
    """

    def __init__(self,
                 photon_input_channels: int,
                 tof_input_channels: int,
                 skip_type: str = 'sum',
                 recurrent_block_type: str = 'convlstm',
                 num_encoders: int = 4,
                 base_num_channels: int = 32,
                 num_residual_blocks: int = 2,
                 norm: str = None,
                 use_upsample_conv: bool = True,
                 downsample_mode: str = 'conv',
                 use_deep_supervision: bool = False,
                 lcwa_r: List[int] = (4, 4, 4, 8),
                 lcwa_bottleneck_ratio: List[int] = (8, 4, 2, 1),
                 lcwa_num_heads: int = 4,
                 use_grid_attention: List[bool] = (False, True, True, True),
                 # DCN 相关参数：仅使用 DCN 自身的 offset 预测
                 use_dcn_align: bool = False,
                 dcn_deform_groups_list: List[int] = None):
        super().__init__()

        assert skip_type in ['sum', 'concat']
        self.skip_type = skip_type
        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.norm = norm
        self.use_upsample_conv = use_upsample_conv
        self.downsample_mode = downsample_mode
        self.is_lstm = (recurrent_block_type.lower() == "convlstm")
        self.use_deep_supervision = use_deep_supervision

        self.lcwa_r = list(lcwa_r)
        self.lcwa_bottleneck_ratio = list(lcwa_bottleneck_ratio)
        self.lcwa_num_heads = lcwa_num_heads
        self.use_grid_attention = list(use_grid_attention)

        # DCN 对齐开关
        self.use_dcn_align = use_dcn_align and MMCV_AVAILABLE

        # 每层 deform_groups 配置，从浅到深
        if dcn_deform_groups_list is None:
            default_groups = [4, 4, 4, 4]
            self.dcn_deform_groups_list = default_groups[:num_encoders]
        else:
            assert len(dcn_deform_groups_list) == num_encoders, \
                f"dcn_deform_groups_list length {len(dcn_deform_groups_list)} != num_encoders {num_encoders}"
            self.dcn_deform_groups_list = list(dcn_deform_groups_list)

        self.photon_head = URCAFuseHead(out_channels=base_num_channels)

        # ConvLayer(photon_input_channels,
        #           base_num_channels,
        #           kernel_size=5, stride=1, padding=2,
        #           norm=norm, activation=True)
        self.tof_head = ConvLayer(tof_input_channels,
                                  base_num_channels,
                                  kernel_size=5, stride=1, padding=2,
                                  norm=norm, activation=True)

        # ----- 通道配置 -----
        encoder_input_sizes = [base_num_channels * (2 ** i) for i in range(num_encoders)]
        encoder_output_sizes = [base_num_channels * (2 ** (i + 1)) for i in range(num_encoders)]
        self.max_num_channels = base_num_channels * (2 ** num_encoders)  # 最深

        # ----- Encoders + Cross-modal fusions -----
        self.photon_encoders = nn.ModuleList()
        self.tof_encoders = nn.ModuleList()
        self.encoder_fusions = nn.ModuleList()
        max_offset = [2,1,0.5,0.25]
        for idx, (in_c, out_c) in enumerate(zip(encoder_input_sizes, encoder_output_sizes)):
            g = self.dcn_deform_groups_list[idx]
            # 保守检查：out_channels 必须能被 deform_groups 整除
            assert out_c % g == 0, \
                f"Encoder level {idx}: out_channels={out_c} not divisible by deform_groups={g}"

            self.photon_encoders.append(
                RecurrentConvLayer(in_c, out_c,
                                   kernel_size=5, stride=2, padding=2,
                                   recurrent_block_type=recurrent_block_type,
                                   norm=norm,
                                   downsample_mode=downsample_mode,
                                   use_dcn_align=self.use_dcn_align,
                                   dcn_deform_groups=g,
                                   dcn_max_offset = max_offset[idx]
                                   )
            )
            self.tof_encoders.append(
                RecurrentConvLayer(in_c, out_c,
                                   kernel_size=5, stride=2, padding=2,
                                   recurrent_block_type=recurrent_block_type,
                                   norm=norm,
                                   downsample_mode=downsample_mode,
                                   use_dcn_align=self.use_dcn_align,
                                   dcn_deform_groups=g,
                                   dcn_max_offset=max_offset[idx]
                                   )
            )
            self.encoder_fusions.append(CrossModalBlock(out_c))

        # ----- Bottleneck residual blocks 两条分支各自一套 -----
        self.photon_resblocks = nn.ModuleList([
            ResidualBlock(self.max_num_channels, self.max_num_channels, norm=norm,spatial_kernel=3)
            for _ in range(num_residual_blocks)
        ])
        self.tof_resblocks = nn.ModuleList([
            ResidualBlock(self.max_num_channels, self.max_num_channels, norm=norm,spatial_kernel=3)
            for _ in range(num_residual_blocks)
        ])

        # Bottleneck 交互
        self.bottleneck_fusion = CrossModalBlock(self.max_num_channels)

        # Bottleneck LCWA
        self.photon_bottleneck_lcwas = nn.ModuleList([
            LocalCrossWindowAttention(
                n_feat=self.max_num_channels,
                r=self.lcwa_r[0],
                num_heads=self.lcwa_num_heads,
                mlp_ratio=4.0,
                bottleneck_ratio=self.lcwa_bottleneck_ratio[0],
                use_grid_attention=self.use_grid_attention[0]
            ) for _ in range(0)
        ])

        self.tof_bottleneck_lcwas = nn.ModuleList([
            LocalCrossWindowAttention(
                n_feat=self.max_num_channels,
                r=self.lcwa_r[0],
                num_heads=self.lcwa_num_heads,
                mlp_ratio=4.0,
                bottleneck_ratio=self.lcwa_bottleneck_ratio[0],
                use_grid_attention=self.use_grid_attention[0]
            ) for _ in range(0)
        ])

        # ----- Decoders -----
        decoder_input_sizes = list(reversed(encoder_output_sizes))  # [max, ..., 2*base]

        self.photon_decoders = nn.ModuleList()
        self.tof_decoders = nn.ModuleList()

        self.photon_skip_refine = nn.ModuleList()
        self.tof_skip_refine = nn.ModuleList()

        self.photon_lcwa_scales = nn.ModuleList()
        self.tof_lcwa_scales = nn.ModuleList()

        self.photon_fusions = nn.ModuleList()
        self.tof_fusions = nn.ModuleList()

        for idx, in_c in enumerate(decoder_input_sizes):
            dec_in = in_c
            dec_out = in_c // 2

            self.photon_fusions.append(AttentionSkipFusion(dec_in))
            self.tof_fusions.append(AttentionSkipFusion(dec_in))

            self.photon_skip_refine.append(
                ResidualBlock(dec_in, dec_in, norm=None)
            )
            self.tof_skip_refine.append(
                ResidualBlock(dec_in, dec_in, norm=None)
            )

            self.photon_lcwa_scales.append(
                LocalCrossWindowAttention(
                    n_feat=dec_in,
                    r=self.lcwa_r[idx],
                    num_heads=self.lcwa_num_heads,
                    mlp_ratio=4.0,
                    attn_drop=0.0,
                    drop=0.0,
                    drop_path=0.0,
                    bottleneck_ratio=self.lcwa_bottleneck_ratio[idx],
                    use_grid_attention=self.use_grid_attention[idx]
                )
            )
            self.tof_lcwa_scales.append(
                LocalCrossWindowAttention(
                    n_feat=dec_in,
                    r=self.lcwa_r[idx],
                    num_heads=self.lcwa_num_heads,
                    mlp_ratio=4.0,
                    attn_drop=0.0,
                    drop=0.0,
                    drop_path=0.0,
                    bottleneck_ratio=self.lcwa_bottleneck_ratio[idx],
                    use_grid_attention=self.use_grid_attention[idx]
                )
            )

            if use_upsample_conv:
                self.photon_decoders.append(
                    UpsampleConvLayer(dec_in, dec_out,
                                      kernel_size=5, padding=2, norm=norm)
                )
                self.tof_decoders.append(
                    UpsampleConvLayer(dec_in, dec_out,
                                      kernel_size=5, padding=2, norm=norm)
                )
            else:
                self.photon_decoders.append(
                    nn.ConvTranspose2d(dec_in, dec_out,
                                       kernel_size=4, stride=2, padding=1)
                )
                self.tof_decoders.append(
                    nn.ConvTranspose2d(dec_in, dec_out,
                                       kernel_size=4, stride=2, padding=1)
                )

        pred_in_channels = base_num_channels

        # ----- full-res tail -----
        self.photon_tail_fusion = AttentionSkipFusion(base_num_channels)
        self.tof_tail_fusion = AttentionSkipFusion(base_num_channels)

        self.photon_lcwa_tail = LocalCrossWindowAttention(
            n_feat=pred_in_channels,
            r=self.lcwa_r[-1],
            num_heads=self.lcwa_num_heads,
            mlp_ratio=4.0,
            attn_drop=0.0,
            drop=0.0,
            drop_path=0.0,
            bottleneck_ratio=self.lcwa_bottleneck_ratio[-1],
            use_grid_attention=self.use_grid_attention[-1]
        )

        self.tof_lcwa_tail = LocalCrossWindowAttention(
            n_feat=pred_in_channels,
            r=self.lcwa_r[-1],
            num_heads=self.lcwa_num_heads,
            mlp_ratio=4.0,
            attn_drop=0.0,
            drop=0.0,
            drop_path=0.0,
            bottleneck_ratio=self.lcwa_bottleneck_ratio[-1],
            use_grid_attention=self.use_grid_attention[-1]
        )

        self.photon_tail_refine = ResidualBlock(pred_in_channels, pred_in_channels, norm=None)
        self.tof_tail_refine = ResidualBlock(pred_in_channels, pred_in_channels, norm=None)

        self.photon_tail_conv = ConvLayer(pred_in_channels,
                                          base_num_channels,
                                          kernel_size=1, stride=1, padding=0,
                                          norm=norm, activation=True)
        self.tof_tail_conv = ConvLayer(pred_in_channels,
                                       base_num_channels,
                                       kernel_size=1, stride=1, padding=0,
                                       norm=norm, activation=True)
        self.register_buffer("_step_counter", torch.zeros((), dtype=torch.long), persistent=False)
    def apply_skip(self, x, skip):
        if self.skip_type == 'sum':
            return x + skip
        else:  # 'concat'
            return torch.cat([x, skip], dim=1)

    def forward(self,
                photon_tensor,
                tof_tensor,
                prev_states_photon=None,
                prev_states_tof=None):
        if prev_states_photon is None and prev_states_tof is None:
            self._step_counter.zero_()
        step_idx = int(self._step_counter.item())
        # 注意：这里直接用 photon_tensor 最后一通道构建 U，具体与你的数据格式要对齐
        U_map = photon_tensor[:, -1:]
        U = U_map #/ (U_map.max() + 1e-6)

        # ----- Head -----
        h_p = self.photon_head(photon_tensor)  # (B, C, H, W)
        h_t = self.tof_head(tof_tensor)
        head_p = h_p
        head_t = h_t

        if prev_states_photon is None:
            prev_states_photon = [None] * self.num_encoders
        if prev_states_tof is None:
            prev_states_tof = [None] * self.num_encoders

        assert len(prev_states_photon) == self.num_encoders
        assert len(prev_states_tof) == self.num_encoders

        blocks_p = []
        blocks_t = []
        new_states_p = []
        new_states_t = []

        # ----- Encoders -----
        for i in range(self.num_encoders):
            enc_p = self.photon_encoders[i]
            enc_t = self.tof_encoders[i]
            fusion = self.encoder_fusions[i]

            # 仅使用 DCN 的对齐（不再传入 offset_prior）
            h_p, state_p = enc_p(h_p, prev_states_photon[i], step_idx=step_idx)
            h_t, state_t = enc_t(h_t, prev_states_tof[i], step_idx=step_idx)

            h_p, h_t = fusion(h_p, h_t)

            blocks_p.append(h_p)
            blocks_t.append(h_t)
            new_states_p.append(state_p)
            new_states_t.append(state_t)

        # ----- Bottleneck -----
        for rb in self.photon_resblocks:
            h_p = rb(h_p)
        for rb in self.tof_resblocks:
            h_t = rb(h_t)

        h_p, h_t = self.bottleneck_fusion(h_p, h_t)

        for blk_p, blk_t in zip(self.photon_bottleneck_lcwas, self.tof_bottleneck_lcwas):
            h_p = blk_p(h_p, uncertainty=U)
            h_t = blk_t(h_t, uncertainty=U)

        # ===== Decoder，多尺度特征收集 =====
        feat_I_scales = []  # 从粗到细的强度特征（最后一个是 full-res）
        feat_D_scales = []  # 从粗到细的深度特征

        for i in range(self.num_encoders):
            skip_idx = self.num_encoders - 1 - i

            # photon 分支
            skip_p = blocks_p[skip_idx]
            h_p = self.photon_fusions[i](h_p, skip_p)
            h_p = self.photon_skip_refine[i](h_p)
            h_p = self.photon_lcwa_scales[i](h_p, uncertainty=U)
            h_p = self.photon_decoders[i](h_p)

            # tof 分支
            skip_t = blocks_t[skip_idx]
            h_t = self.tof_fusions[i](h_t, skip_t)
            h_t = self.tof_skip_refine[i](h_t)
            h_t = self.tof_lcwa_scales[i](h_t, uncertainty=U)
            h_t = self.tof_decoders[i](h_t)

            if i < self.num_encoders - 1:
                feat_I_scales.append(h_p)
                feat_D_scales.append(h_t)

        # ===== full-res tail =====
        h_p = self.photon_tail_fusion(h_p, head_p)
        h_p = self.photon_lcwa_tail(h_p, uncertainty=U)
        h_p = self.photon_tail_refine(h_p)

        h_t = self.tof_tail_fusion(h_t, head_t)
        h_t = self.tof_lcwa_tail(h_t, uncertainty=U)
        h_t = self.tof_tail_refine(h_t)

        feat_I_main = self.photon_tail_conv(h_p)
        feat_D_main = self.tof_tail_conv(h_t)

        feat_I_scales.append(feat_I_main)
        feat_D_scales.append(feat_D_main)
        self._step_counter += 1
        # 返回：从最粗到最细（最后一个 full-res）
        return feat_I_scales, feat_D_scales, new_states_p, new_states_t


class IntensityDepthRecurrent(nn.Module):
    """
    顶层模型：
    - 输入: photon_tensor, tof_tensor
    - 输出:
        I_deep_list: 多尺度强度预测 (每个元素 ∈ (0,1))
        D_deep_list: 多尺度深度预测 (每个元素 ∈ (0,Dmax))
    """
    def __init__(self,
                 photon_input_channels: int,
                 tof_input_channels: int,
                 Dmax: float = 66.6,
                 skip_type: str = 'sum',
                 recurrent_block_type: str = 'convlstm',
                 num_encoders: int = 4,
                 base_num_channels: int = 32,
                 num_residual_blocks: int = 2,
                 norm: str = 'batch',
                 use_upsample_conv: bool = True,
                 use_depth_residual: bool = False,
                 max_depth_residual: float = 66.6,
                 downsample_mode: str = 'avgpool',
                 use_deep_supervision: bool = False,
                 use_ms_depth_residual: bool = False,
                 eps: float = 1e-3):
        super().__init__()

        self.Dmax = float(Dmax)
        self.use_depth_residual = use_depth_residual
        self.max_depth_residual = float(max_depth_residual)
        self.eps = float(eps)
        self.num_encoders = num_encoders
        self.use_deep_supervision = use_deep_supervision
        self.use_ms_depth_residual = use_ms_depth_residual

        # 每层 DCN deform_groups 配置（从浅到深）
        dcn_groups = [1, 2, 4, 8][:num_encoders]

        # backbone
        self.backbone = DualUNetRecurrent(
            photon_input_channels=photon_input_channels,
            tof_input_channels=tof_input_channels,
            skip_type=skip_type,
            recurrent_block_type=recurrent_block_type,
            num_encoders=num_encoders,
            base_num_channels=base_num_channels,
            num_residual_blocks=num_residual_blocks,
            norm=norm,
            use_upsample_conv=use_upsample_conv,
            downsample_mode=downsample_mode,
            use_deep_supervision=use_deep_supervision,
            use_dcn_align=True,
            dcn_deform_groups_list=dcn_groups
        )

        # decoder 各层输出通道，从深到浅（与 feat_*_scales 对应）
        dec_out_channels = [
            base_num_channels * (2 ** i)
            for i in range(num_encoders - 1, -1, -1)
        ]  # e.g. [256, 128, 64, 32] for base=32,num_encoders=4

        self.num_scales = num_encoders  # feat_I_scales 的长度

        # 强度 head
        self.intensity_heads = nn.ModuleList([
            ConvLayer(ch, 1, kernel_size=3, stride=1, padding=1,
                      norm=None, activation=False)
            for ch in dec_out_channels
        ])
        self.intensity_activation = nn.Sigmoid()
        conv_main_I = self.intensity_heads[-1].block[0]
        torch.nn.init.zeros_(conv_main_I.weight)
        torch.nn.init.constant_(conv_main_I.bias, 0.0)

        # 深度 head
        self.depth_heads = nn.ModuleList([
            ConvLayer(ch, 1, kernel_size=3, stride=1, padding=1,
                      norm=None, activation=False)
            for ch in dec_out_channels
        ])
        if use_depth_residual:
            self.depth_activation = nn.Tanh()
        else:
            self.depth_activation = nn.Sigmoid()

    def forward(self,
                photon_tensor,
                tof_tensor,
                prev_states_photon=None,
                prev_states_tof=None,
                depth_baseline=None,
                intensity_baseline = None
                ):

        feat_I_scales, feat_D_scales, new_states_p, new_states_t = self.backbone(
            photon_tensor, tof_tensor,
            prev_states_photon=prev_states_photon,
            prev_states_tof=prev_states_tof
        )
        B, _, H, W = photon_tensor.shape

        I_deep_list = []
        D_deep_list = []

        assert len(feat_I_scales) == self.num_scales

        for i, (fI, fD) in enumerate(zip(feat_I_scales, feat_D_scales)):
            # ---------- 多尺度强度 ----------
            I_raw = self.intensity_heads[i](fI)  # 作为 Δlogit（未约束）

            if intensity_baseline is not None:
                H_i, W_i = fI.shape[-2], fI.shape[-1]
                I_base = F.interpolate(intensity_baseline, size=(H_i, W_i),
                                       mode='bilinear', align_corners=False)
                I_base = I_base.clamp(self.eps, 1.0 - self.eps)
                base_logit = torch.logit(I_base)
                delta = torch.tanh(I_raw) * 3.0
                I_i = torch.sigmoid(base_logit + delta)
            else:
                I_i = torch.sigmoid(I_raw)
            I_deep_list.append(I_i)

            # ---------- 多尺度深度 ----------
            depth_raw_i = self.depth_heads[i](fD)

            if self.use_ms_depth_residual and self.use_depth_residual and depth_baseline is not None:
                H_i, W_i = fD.shape[-2], fD.shape[-1]
                depth_base_i = F.interpolate(depth_baseline,
                                             size=(H_i, W_i),
                                             mode='bilinear',
                                             align_corners=False)

                delta_norm_i = self.depth_activation(depth_raw_i)  # (-1,1)
                delta_D_i = delta_norm_i * self.max_depth_residual
                D_i_low = depth_base_i + delta_D_i
                D_i_low = torch.clamp(D_i_low,
                                      min=self.eps,
                                      max=self.Dmax - self.eps)
                D_i = D_i_low

            else:
                d_norm_i = self.depth_activation(depth_raw_i)  # (0,1) / (-1,1)
                D_i = d_norm_i * self.Dmax

            D_i_up = F.interpolate(D_i, size=(H, W),
                                   mode='bilinear', align_corners=False)
            D_deep_list.append(D_i_up)
        return I_deep_list, D_deep_list, new_states_p, new_states_t


if __name__ == "__main__":
    import time

    photon_bins = 1
    tof_bins = 2
    H, W = 128, 128
    batch_size = 4
    Dmax = 66.6

    infer_warmup_steps = 10
    infer_num_runs = 5
    infer_steps_per_run = 20

    train_warmup_steps = 5
    train_num_runs = 3
    train_steps_per_run = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    photon_x = torch.randn(batch_size, photon_bins, H, W, device=device)
    tof_x = torch.randn(batch_size, tof_bins, H, W, device=device)

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def detach_recurrent_states(states, is_lstm: bool):
        if states is None:
            return None
        new_states = []
        for s in states:
            if s is None:
                new_states.append(None)
                continue
            if is_lstm:
                h, c = s
                new_states.append((h.detach(), c.detach()))
            else:
                new_states.append(s.detach())
        return new_states

    for rtype in ["convlstm", "convgru"]:
        print("\n========================================")
        print(f"Testing IntensityDepthRecurrent, recurrent_block_type = {rtype}")
        print("========================================")

        is_lstm = (rtype == "convlstm")

        model = IntensityDepthRecurrent(
            photon_input_channels=photon_bins,
            tof_input_channels=tof_bins,
            Dmax=Dmax,
            skip_type='sum',
            recurrent_block_type=rtype,
            num_encoders=4,
            base_num_channels=32,
            num_residual_blocks=2,
            norm='instance',
            use_upsample_conv=True,
            use_depth_residual=False
        ).to(device)

        print("Trainable params:", count_params(model))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # ========== 推理: warmup ==========
        model.eval()
        prev_states_p = None
        prev_states_t = None
        with torch.no_grad():
            for _ in range(infer_warmup_steps):
                I_list, D_list, prev_states_p, prev_states_t = model(
                    photon_x, tof_x,
                    prev_states_photon=prev_states_p,
                    prev_states_tof=prev_states_t,
                    depth_baseline=None
                )
                I_hat = I_list[-1]
                D_hat = D_list[-1]

        # ========== 推理: 多次平均 ==========
        model.eval()
        total_infer_time = 0.0
        total_infer_steps = infer_num_runs * infer_steps_per_run

        with torch.no_grad():
            for run in range(infer_num_runs):
                prev_states_p = None
                prev_states_t = None
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()

                for _ in range(infer_steps_per_run):
                    I_list, D_list, prev_states_p, prev_states_t = model(
                        photon_x, tof_x,
                        prev_states_photon=prev_states_p,
                        prev_states_tof=prev_states_t,
                        depth_baseline=None
                    )
                    I_hat = I_list[-1]
                    D_hat = D_list[-1]

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                dt = t1 - t0
                total_infer_time += dt
                print(f"[Inference Run {run+1}/{infer_num_runs}] "
                      f"steps={infer_steps_per_run}, time={dt:.4f}s")

        avg_infer_time_per_step = total_infer_time / total_infer_steps
        infer_fps = batch_size / avg_infer_time_per_step
        print(f"[Inference AVG] steps={total_infer_steps}, "
              f"avg_time/step={avg_infer_time_per_step*1000:.3f} ms, "
              f"FPS(batch-level)={infer_fps:.2f}")
        print(f"Sample I_hat range: ({I_hat.min().item():.4f}, {I_hat.max().item():.4f})")
        print(f"Sample D_hat range: ({D_hat.min().item():.4f}, {D_hat.max().item():.4f})")

        # ========== 训练: warmup ==========
        model.train()
        prev_states_p = None
        prev_states_t = None
        for _ in range(train_warmup_steps):
            optimizer.zero_grad()
            prev_states_p = detach_recurrent_states(prev_states_p, is_lstm)
            prev_states_t = detach_recurrent_states(prev_states_t, is_lstm)

            I_list, D_list, prev_states_p, prev_states_t = model(
                photon_x, tof_x,
                prev_states_photon=prev_states_p,
                prev_states_tof=prev_states_t,
                depth_baseline=None
            )
            I_hat = I_list[-1]
            D_hat = D_list[-1]

            target_I = torch.zeros_like(I_hat)
            target_D = torch.zeros_like(D_hat)
            loss_I = criterion(I_hat, target_I)
            loss_D = criterion(D_hat, target_D)
            loss = loss_I + loss_D
            loss.backward()
            optimizer.step()

        # ========== 训练: 多次平均 ==========
        model.train()
        total_train_time = 0.0
        total_train_steps = train_num_runs * train_steps_per_run
        last_loss = None

        try:
            for run in range(train_num_runs):
                prev_states_p = None
                prev_states_t = None
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()

                for _ in range(train_steps_per_run):
                    optimizer.zero_grad()
                    prev_states_p = detach_recurrent_states(prev_states_p, is_lstm)
                    prev_states_t = detach_recurrent_states(prev_states_t, is_lstm)

                    I_list, D_list, prev_states_p, prev_states_t = model(
                        photon_x, tof_x,
                        prev_states_photon=prev_states_p,
                        prev_states_tof=prev_states_t,
                        depth_baseline=None
                    )
                    I_hat = I_list[-1]
                    D_hat = D_list[-1]

                    target_I = torch.zeros_like(I_hat)
                    target_D = torch.zeros_like(D_hat)
                    loss_I = criterion(I_hat, target_I)
                    loss_D = criterion(D_hat, target_D)
                    loss = loss_I + loss_D
                    loss.backward()
                    optimizer.step()
                    last_loss = loss.item()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                dt = t1 - t0
                total_train_time += dt
                print(f"[Train Run {run+1}/{train_num_runs}] "
                      f"steps={train_steps_per_run}, time={dt:.4f}s, "
                      f"last_loss={last_loss:.6f}")

            avg_train_time_per_step = total_train_time / total_train_steps
            train_fps = batch_size / avg_train_time_per_step
            print(f"[Train AVG] steps={total_train_steps}, "
                  f"avg_time/step={avg_train_time_per_step*1000:.3f} ms, "
                  f"FPS(batch-level)={train_fps:.2f}")
            print(f"Last training loss: {last_loss:.6f}")
            print("Training finished without runtime errors.")
        except Exception as e:
            print("!!! Training raised an exception !!!")
            print(e)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from mamba_ssm import Mamba as SSMMamba
except ImportError:  # pragma: no cover - fallback for non-mamba envs
    SSMMamba = None


def extract_feature_entry(entry):
    """Normalize a diffusion feature entry to a tensor feature."""
    if isinstance(entry, dict):
        feat = entry.get('feat', entry.get('feature', entry.get('tensor')))
        return feat
    if isinstance(entry, (tuple, list)):
        return entry[0]
    return entry


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


def sinusoidal_timestep_embedding(timesteps, dim, max_period=10000):
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps], dtype=torch.float32)
    timesteps = timesteps.float().view(-1)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device).float() / max(half, 1))
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeConditionedAffine(nn.Module):
    def __init__(self, channels, time_dim, bias=False):
        super().__init__()
        self.to_affine = nn.Sequential(
            nn.Linear(time_dim, channels * 2, bias=True),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2, bias=True),
        )
        self.bias = bias

    def forward(self, x, time_emb):
        affine = self.to_affine(time_emb).view(time_emb.shape[0], -1, 1, 1)
        shift, scale = affine.chunk(2, dim=1)
        return x * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift


class DualMedicalPromptConditioning(nn.Module):
    """Generate structure/function/shared prompt maps from clean latent streams."""

    def __init__(self, channels, bias=False):
        super().__init__()
        self.structure_proj = nn.Sequential(
            ConvNormAct(channels * 3, channels, kernel_size=1, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.function_proj = nn.Sequential(
            ConvNormAct(channels * 3, channels, kernel_size=1, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.shared_proj = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )

    def forward(self, z_s, z_f):
        shared = self.shared_proj(torch.cat([z_s, z_f], dim=1))
        structure = self.structure_proj(torch.cat([z_s, z_s - z_f, shared], dim=1))
        function = self.function_proj(torch.cat([z_f, z_f - z_s, shared], dim=1))
        return {
            'structure': structure,
            'function': function,
            'shared': shared,
        }


class RestormerStyleGDFN(nn.Module):
    def __init__(self, channels, bias=False, ffn_expansion=1.5):
        super().__init__()
        hidden = max(int(channels * ffn_expansion), channels)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class LightweightLocalMixer(nn.Module):
    """Shape-preserving local mixer used when a stage disables Mamba."""

    def __init__(self, channels, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.block(x)


class Mamba2DMixer(nn.Module):
    """2D CNN + true Mamba mixer with bidirectional sequence adaptation."""

    def __init__(self, channels, bias=False, d_state=16, d_conv=4, expand=2, bidirectional=True):
        super().__init__()
        self.has_mamba = SSMMamba is not None
        self.fallback_hidden = max(channels // 2, 8)
        self.bidirectional = bidirectional
        self.pre_norm = nn.LayerNorm(channels)
        self.local_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
        )
        if self.has_mamba:
            self.mamba = SSMMamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bias=bias,
            )
            self.mamba_rev = SSMMamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bias=bias,
            ) if bidirectional else None
        else:
            self.mamba = None
            self.mamba_rev = None
        self.post_norm = nn.LayerNorm(channels)
        self.fallback_norm = nn.LayerNorm(channels)
        self.fallback_in_proj = nn.Linear(channels, self.fallback_hidden * 2, bias=True)
        self.fallback_dwconv = nn.Conv1d(
            self.fallback_hidden,
            self.fallback_hidden,
            kernel_size=3,
            padding=1,
            groups=self.fallback_hidden,
            bias=bias,
        )
        self.fallback_out_proj = nn.Sequential(
            nn.Linear(self.fallback_hidden, channels, bias=True),
            nn.GELU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.merge_proj = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )

    def _mamba_forward(self, seq):
        out = self.mamba(self.pre_norm(seq))
        if not self.bidirectional:
            return out
        rev = torch.flip(seq, dims=[1])
        rev = self.mamba_rev(self.pre_norm(rev))
        rev = torch.flip(rev, dims=[1])
        return 0.5 * (out + rev)

    def _fallback_forward(self, seq):
        seq = self.fallback_norm(seq)
        u, v = self.fallback_in_proj(seq).chunk(2, dim=-1)
        token = F.silu(u).transpose(1, 2)
        token = self.fallback_dwconv(token).transpose(1, 2)
        return self.fallback_out_proj(F.silu(token) * v)

    def forward(self, x):
        b, c, h, w = x.shape
        local = self.local_path(x)
        seq = x.flatten(2).transpose(1, 2).contiguous()
        if x.is_cuda and self.has_mamba:
            try:
                token = self._mamba_forward(seq)
            except RuntimeError:
                token = self._fallback_forward(seq)
        else:
            token = self._fallback_forward(seq)
        token = self.post_norm(token)
        token = token.transpose(1, 2).reshape(b, c, h, w)
        return self.merge_proj(torch.cat([local, token], dim=1))


class SharedMultiDomainCore(nn.Module):
    def __init__(
        self,
        channels,
        time_dim=None,
        bias=False,
        role='structure',
        disable_mamba=False,
        use_mamba=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_bidirectional=True,
        ffn_expansion=1.5,
    ):
        super().__init__()
        if role not in ('structure', 'function'):
            raise ValueError("role must be 'structure' or 'function'.")
        self.role = role
        self.use_mamba = bool(use_mamba) and not disable_mamba
        self.disable_mamba = not self.use_mamba
        self.time_affine = TimeConditionedAffine(channels, time_dim, bias=bias) if time_dim is not None else None
        self.prompt_proj = nn.Sequential(
            nn.LazyConv2d(channels, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
        )
        self.bank_context_proj = nn.Sequential(
            nn.LazyConv2d(channels, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
        )
        self.bank_context_to_bank = nn.Sequential(
            nn.LazyConv2d(channels * 2, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=bias),
        )
        self.local_mixer = ConvNormAct(channels, channels, kernel_size=3, bias=bias)
        if self.use_mamba:
            self.mamba_mixer = Mamba2DMixer(
                channels,
                bias=bias,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                bidirectional=mamba_bidirectional,
            )
        else:
            self.mamba_mixer = LightweightLocalMixer(channels, bias=bias)
        self.branch_fuse = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        # Cross-branch gating: each branch modulates the other before fusion
        self.local_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )
        self.mamba_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Sequential(
            ConvNormAct(channels * 2, channels, kernel_size=1, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.ffn = RestormerStyleGDFN(channels, bias=bias, ffn_expansion=ffn_expansion)

    def forward(self, x, prompt_map, time_emb=None, bank_context=None, role_scale=None, interaction_scale=None):
        if prompt_map.shape[-2:] != x.shape[-2:]:
            prompt_map = F.interpolate(prompt_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if self.time_affine is not None and time_emb is not None:
            x = self.time_affine(x, time_emb)
        prompt = self.prompt_proj(prompt_map)
        if bank_context is not None:
            if bank_context.shape[-2:] != x.shape[-2:]:
                bank_context = F.interpolate(bank_context, size=x.shape[-2:], mode='bilinear', align_corners=False)
            bank_context = self.bank_context_proj(bank_context)
            prompt = prompt + 0.1 * bank_context
        if role_scale is None:
            role_scale = x.new_full((x.shape[0], 1, 1, 1), 0.5)
        elif role_scale.dim() == 2:
            role_scale = role_scale.view(role_scale.shape[0], 1, 1, 1)
        elif role_scale.dim() == 1:
            role_scale = role_scale.view(role_scale.shape[0], 1, 1, 1)
        role_scale = role_scale.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
        interaction_scale = role_scale if interaction_scale is None else interaction_scale
        if interaction_scale.dim() == 2:
            interaction_scale = interaction_scale.view(interaction_scale.shape[0], 1, 1, 1)
        elif interaction_scale.dim() == 1:
            interaction_scale = interaction_scale.view(interaction_scale.shape[0], 1, 1, 1)
        interaction_scale = interaction_scale.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
        x_prompt = x + prompt
        if self.role == 'structure':
            local_feat = self.local_mixer(x_prompt)
            mamba_feat = self.mamba_mixer(local_feat + 0.5 * prompt)
            # Cross-branch gating: each branch modulates the other
            local_gate = self.local_gate(local_feat)
            mamba_gate = self.mamba_gate(mamba_feat)
            local_modulated = local_feat * mamba_gate
            mamba_modulated = mamba_feat * local_gate
            primary = local_modulated * (0.55 + 0.45 * role_scale)
            secondary = mamba_modulated * (0.35 + 0.65 * (1.0 - role_scale))
            fused = self.branch_fuse(torch.cat([primary, secondary], dim=1))
        else:
            mamba_feat = self.mamba_mixer(x_prompt)
            local_feat = self.local_mixer(mamba_feat + 0.5 * prompt)
            # Cross-branch gating: each branch modulates the other
            local_gate = self.local_gate(local_feat)
            mamba_gate = self.mamba_gate(mamba_feat)
            local_modulated = local_feat * mamba_gate
            mamba_modulated = mamba_feat * local_gate
            primary = mamba_modulated * (0.55 + 0.45 * role_scale)
            secondary = local_modulated * (0.35 + 0.65 * (1.0 - role_scale))
            fused = self.branch_fuse(torch.cat([primary, secondary], dim=1))
        out = self.fuse_proj(torch.cat([x, fused], dim=1))
        out = out + (0.7 + 0.3 * interaction_scale) * self.ffn(out)
        if bank_context is not None:
            out = out + (0.03 + 0.07 * interaction_scale) * bank_context
        return x + out, {
            'domain_weights': None,
            'fused_domain': fused,
            'local_domain': local_feat,
            'mamba_domain': mamba_feat,
            'prompt_domain': prompt,
        }


class DualStreamRestorationBlock(nn.Module):
    """Dual-stream denoising block with asymmetric structure/function cores."""

    def __init__(
        self,
        channels,
        time_dim=None,
        bias=False,
        disable_mamba=False,
        use_mamba=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_bidirectional=True,
        ffn_expansion=1.5,
    ):
        super().__init__()
        self.use_mamba = bool(use_mamba) and not disable_mamba
        self.structure_core = SharedMultiDomainCore(
            channels,
            time_dim=time_dim,
            bias=bias,
            role='structure',
            disable_mamba=disable_mamba,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_bidirectional=mamba_bidirectional,
            ffn_expansion=ffn_expansion,
        )
        self.function_core = SharedMultiDomainCore(
            channels,
            time_dim=time_dim,
            bias=bias,
            role='function',
            disable_mamba=disable_mamba,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_bidirectional=mamba_bidirectional,
            ffn_expansion=ffn_expansion,
        )
        self.time_to_common = TimeConditionedAffine(channels, time_dim, bias=bias) if time_dim is not None else None
        self.shared_exchange = nn.Sequential(
            ConvNormAct(channels * 3, channels, kernel_size=1, bias=bias),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
        )
        self.bank_context_proj = nn.Sequential(
            nn.LazyConv2d(channels, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
        )
        self.bank_context_to_bank = nn.Sequential(
            nn.LazyConv2d(channels * 2, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=bias),
        )
        self.shared_gate = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.merge_proj = nn.Sequential(
            ConvNormAct(channels * 2, channels * 2, kernel_size=1, bias=bias),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, bias=bias),
        )
        self.structure_ffn = RestormerStyleGDFN(channels, bias=bias, ffn_expansion=ffn_expansion)
        self.function_ffn = RestormerStyleGDFN(channels, bias=bias, ffn_expansion=ffn_expansion)
        self.bank_ffn = RestormerStyleGDFN(channels * 2, bias=bias, ffn_expansion=ffn_expansion)

    @staticmethod
    def _expand_scale(scale, reference):
        if scale is None:
            return reference.new_full((reference.shape[0], 1, 1, 1), 0.5)
        if not torch.is_tensor(scale):
            scale = reference.new_tensor(scale)
        if scale.dim() == 0:
            scale = scale.view(1, 1, 1, 1).repeat(reference.shape[0], 1, 1, 1)
        elif scale.dim() == 1:
            scale = scale.view(-1, 1, 1, 1)
        elif scale.dim() == 2:
            scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)
        return scale.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _build_role_schedule(step_phase, reference):
        """Build an explicit early-weak / mid-strong / late-weak interaction schedule."""
        if step_phase is None:
            step_phase = reference.new_full((reference.shape[0], 1, 1, 1), 0.5)
        elif step_phase.dim() == 1:
            step_phase = step_phase.view(-1, 1, 1, 1)
        step_phase = step_phase.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)

        # Structural stream is slightly stronger in later steps, functional stream in earlier steps.
        structure_scale = 0.25 + 0.75 * step_phase
        function_scale = 0.25 + 0.75 * (1.0 - step_phase)

        # Strongest around the middle; clearly weaker at both ends.
        interaction_peak = torch.sin(math.pi * step_phase).pow(2)
        interaction_scale = 0.08 + 0.92 * interaction_peak

        # Directional exchange follows the same phase trend but remains asymmetric.
        s_to_f = 0.18 + 0.82 * step_phase
        f_to_s = 0.18 + 0.82 * (1.0 - step_phase)

        # Keep the direct difference exchange conservative near the ends.
        exchange_scale = interaction_scale.pow(1.35)
        return structure_scale, function_scale, interaction_scale, exchange_scale, s_to_f, f_to_s, step_phase

    def forward(self, x_s, x_f, prompt_state, time_emb=None, bank_context=None, step_phase=None):
        p_s = prompt_state['structure']
        p_f = prompt_state['function']
        p_shared = prompt_state['shared']

        if bank_context is not None:
            if bank_context.shape[-2:] != x_s.shape[-2:]:
                bank_context = F.interpolate(bank_context, size=x_s.shape[-2:], mode='bilinear', align_corners=False)
            bank_context = self.bank_context_proj(bank_context)

        structure_scale, function_scale, interaction_scale, exchange_scale, s_to_f, f_to_s, step_phase = self._build_role_schedule(
            step_phase,
            x_s,
        )

        s_feat, s_aux = self.structure_core(
            x_s,
            p_s + p_shared,
            time_emb=time_emb,
            bank_context=bank_context,
            role_scale=structure_scale,
            interaction_scale=interaction_scale,
        )
        f_feat, f_aux = self.function_core(
            x_f,
            p_f + p_shared,
            time_emb=time_emb,
            bank_context=bank_context,
            role_scale=function_scale,
            interaction_scale=interaction_scale,
        )

        common_context = self.shared_exchange(torch.cat([0.5 * (s_feat + f_feat), s_feat - f_feat, p_shared], dim=1))
        if bank_context is not None:
            common_context = common_context + (0.05 + 0.05 * interaction_scale) * bank_context
        if self.time_to_common is not None and time_emb is not None:
            common_context = self.time_to_common(common_context, time_emb)
        common_context = common_context * interaction_scale

        shared_gate = self.shared_gate(torch.cat([s_feat, f_feat, common_context, p_shared], dim=1))
        s_next = s_feat + (0.25 + 0.75 * function_scale) * shared_gate[:, 0:1] * common_context
        f_next = f_feat + (0.25 + 0.75 * structure_scale) * shared_gate[:, 1:2] * common_context
        # Keep direct stream-to-stream exchange conservative so the two branches remain distinguishable.
        cross_delta = torch.tanh(f_feat - s_feat)
        s_next = s_next + exchange_scale * f_to_s * cross_delta
        f_next = f_next - exchange_scale * s_to_f * cross_delta
        s_next = s_next + (0.65 + 0.35 * structure_scale) * self.structure_ffn(s_next)
        f_next = f_next + (0.65 + 0.35 * function_scale) * self.function_ffn(f_next)
        merged = self.merge_proj(torch.cat([s_next, f_next], dim=1))
        merged = merged + (0.55 + 0.45 * interaction_scale) * self.bank_ffn(merged)
        bank_feat = merged
        if bank_context is not None:
            bank_feat = bank_feat + (0.02 + 0.06 * interaction_scale) * self.bank_context_to_bank(bank_context)
        return s_next, f_next, merged, {
            'structure_domain': s_aux['fused_domain'],
            'function_domain': f_aux['fused_domain'],
            'bank_feat': bank_feat,
            'shared_context': common_context,
            'role_schedule': {
                'structure_scale': structure_scale,
                'function_scale': function_scale,
                'interaction_scale': interaction_scale,
                'exchange_scale': exchange_scale,
                's_to_f': s_to_f,
                'f_to_s': f_to_s,
                'step_phase': step_phase,
            },
        }


class DiffusionProcessFeatureBank(nn.Module):
    """Selective multi-scale process feature bank used by DSDFuse."""

    def __init__(
        self,
        out_channels,
        select_stages=None,
        max_features=3,
        summary_mode='average',
        enabled=True,
        bias=False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.select_stages = set(select_stages or [1, 2, 3])
        self.max_features = max(int(max_features), 1)
        self.summary_mode = summary_mode
        self.enabled = bool(enabled)
        self.bias = bias
        self.projections = nn.ModuleDict()

    def _ensure_projection(self, stage_idx, device):
        key = str(stage_idx)
        if key not in self.projections:
            self.projections[key] = nn.LazyConv2d(self.out_channels, kernel_size=1, bias=self.bias)
        self.projections[key] = self.projections[key].to(device)
        return self.projections[key]

    @staticmethod
    def _stage_to_int(stage_idx):
        if isinstance(stage_idx, str):
            stage_idx = stage_idx.replace('stage', '')
        return int(stage_idx)

    def forward(self, features, target_size=None):
        if not self.enabled or not features:
            return []
        items = features.items() if isinstance(features, dict) else enumerate(features)
        bank = []
        for stage_idx, feat in items:
            try:
                stage_idx = self._stage_to_int(stage_idx)
            except (TypeError, ValueError):
                continue
            if stage_idx not in self.select_stages:
                continue
            feat = extract_feature_entry(feat)
            proj = self._ensure_projection(stage_idx, feat.device)(feat)
            if target_size is not None and proj.shape[-2:] != target_size:
                proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=False)
            bank.append(proj)
            if len(bank) >= self.max_features:
                break
        return bank

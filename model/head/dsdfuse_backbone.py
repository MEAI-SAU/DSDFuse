import torch
import torch.nn as nn
import torch.nn.functional as F
from model.head.dsdfuse_blocks import (
    ConvNormAct,
    DiffusionProcessFeatureBank,
    DualMedicalPromptConditioning,
    DualStreamRestorationBlock,
    sinusoidal_timestep_embedding,
)


class DualStreamDenoisingBackbone(nn.Module):
    """DSDFuse denoising backbone with dual-stream collaborative denoising."""

    def __init__(
        self,
        inp_channels=None,
        out_channels=None,
        dim=16,
        num_blocks=None,
        num_channel=None,
        in_channels=None,
        feat_channels=None,
        bias=False,
        num_train_timesteps=1000,
        disable_mamba=False,
        mamba_stage_mask=None,
        mamba_d_state=8,
        mamba_d_conv=3,
        mamba_expand=1,
        mamba_bidirectional=False,
        ffn_expansion=1.5,
        process_bank_enabled=True,
        process_bank_select_stages=None,
        process_bank_max_features=3,
        process_bank_summary_mode='average',
        process_bank_collect_after='rb',
    ):
        super().__init__()
        if inp_channels is None:
            inp_channels = in_channels
        if inp_channels is None:
            raise ValueError('inp_channels or in_channels must be provided.')
        if out_channels is None:
            raise ValueError('out_channels must be provided.')
        if feat_channels is not None:
            num_channel = feat_channels
        if inp_channels % 2 != 0 or out_channels % 2 != 0:
            raise ValueError('DualStreamDenoisingBackbone expects even input/output channels.')

        num_blocks = num_blocks or [2, 2, 2, 2]
        num_channel = num_channel or [dim, dim * 2, dim * 4, dim * 8]
        if mamba_stage_mask is None:
            mamba_stage_mask = [False, False, True, True]
        mamba_stage_mask = list(mamba_stage_mask)
        if len(num_blocks) < 4:
            raise ValueError('num_blocks should provide four stage depths.')
        if len(num_channel) < 4:
            raise ValueError('num_channel/feat_channels should provide four stage widths.')
        if len(mamba_stage_mask) < 4:
            raise ValueError('mamba_stage_mask should provide four stage flags.')

        self.stream_in_channels = inp_channels // 2
        self.stream_out_channels = out_channels // 2
        self.stage_dims = list(num_channel[:4])
        self.time_dim = self.stage_dims[0] * 4
        self.num_train_timesteps = max(int(num_train_timesteps), 1)
        self.disable_mamba = disable_mamba
        self.mamba_stage_mask = [bool(v) for v in mamba_stage_mask[:4]]
        self.effective_mamba_stage_mask = [
            (not self.disable_mamba) and flag for flag in self.mamba_stage_mask
        ]
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_bidirectional = mamba_bidirectional
        self.ffn_expansion = ffn_expansion
        self.process_bank_enabled = bool(process_bank_enabled)
        self.process_bank_select_stages = list(process_bank_select_stages or [1, 2, 3])
        self.process_bank_max_features = max(int(process_bank_max_features), 1)
        self.process_bank_summary_mode = process_bank_summary_mode
        self.process_bank_collect_after = process_bank_collect_after

        self.prompt_conditioning = DualMedicalPromptConditioning(self.stream_in_channels, bias=bias)
        self.prompt_state = None
        self.time_embed = nn.Sequential(
            nn.Linear(self.stage_dims[0], self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.stage_time_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.time_dim, stage_dim),
                nn.GELU(),
                nn.Linear(stage_dim, stage_dim),
            )
            for stage_dim in self.stage_dims
        ])

        self.input_proj_s = ConvNormAct(self.stream_in_channels, self.stage_dims[0], kernel_size=3, bias=bias)
        self.input_proj_f = ConvNormAct(self.stream_in_channels, self.stage_dims[0], kernel_size=3, bias=bias)

        self.prompt_adapters = nn.ModuleList([
            nn.LazyConv2d(stage_dim, kernel_size=1, bias=bias) for stage_dim in self.stage_dims
        ])

        self.encoder_stages = nn.ModuleList()
        self.down_s = nn.ModuleList()
        self.down_f = nn.ModuleList()
        for stage_idx in range(3):
            blocks = nn.ModuleList([
                DualStreamRestorationBlock(
                    self.stage_dims[stage_idx],
                    time_dim=self.stage_dims[stage_idx],
                    bias=bias,
                    disable_mamba=disable_mamba,
                    use_mamba=self.effective_mamba_stage_mask[stage_idx],
                    mamba_d_state=mamba_d_state,
                    mamba_d_conv=mamba_d_conv,
                    mamba_expand=mamba_expand,
                    mamba_bidirectional=mamba_bidirectional,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(num_blocks[stage_idx])
            ])
            self.encoder_stages.append(blocks)
            self.down_s.append(nn.Conv2d(self.stage_dims[stage_idx], self.stage_dims[stage_idx + 1], kernel_size=2, stride=2))
            self.down_f.append(nn.Conv2d(self.stage_dims[stage_idx], self.stage_dims[stage_idx + 1], kernel_size=2, stride=2))

        self.latent_stage = nn.ModuleList([
            DualStreamRestorationBlock(
                self.stage_dims[3],
                time_dim=self.stage_dims[3],
                bias=bias,
                disable_mamba=disable_mamba,
                use_mamba=self.effective_mamba_stage_mask[3],
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_bidirectional=mamba_bidirectional,
                ffn_expansion=ffn_expansion,
            )
            for _ in range(num_blocks[3])
        ])

        self.up_s = nn.ModuleList()
        self.up_f = nn.ModuleList()
        self.skip_reduce_s = nn.ModuleList()
        self.skip_reduce_f = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for stage_idx in range(2, -1, -1):
            in_dim = self.stage_dims[stage_idx + 1]
            out_dim = self.stage_dims[stage_idx]
            self.up_s.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, bias=bias),
                    nn.PixelShuffle(2),
                )
            )
            self.up_f.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, bias=bias),
                    nn.PixelShuffle(2),
                )
            )
            self.skip_reduce_s.append(ConvNormAct(out_dim * 2, out_dim, kernel_size=1, bias=bias))
            self.skip_reduce_f.append(ConvNormAct(out_dim * 2, out_dim, kernel_size=1, bias=bias))
            self.decoder_stages.append(
                nn.ModuleList([
                    DualStreamRestorationBlock(
                        out_dim,
                        time_dim=out_dim,
                        bias=bias,
                        disable_mamba=disable_mamba,
                        use_mamba=self.effective_mamba_stage_mask[stage_idx],
                        mamba_d_state=mamba_d_state,
                        mamba_d_conv=mamba_d_conv,
                        mamba_expand=mamba_expand,
                        mamba_bidirectional=mamba_bidirectional,
                        ffn_expansion=ffn_expansion,
                    )
                    for _ in range(num_blocks[stage_idx])
                ])
            )

        self.bank = DiffusionProcessFeatureBank(
            out_channels,
            select_stages=self.process_bank_select_stages,
            max_features=self.process_bank_max_features,
            summary_mode=self.process_bank_summary_mode,
            enabled=self.process_bank_enabled,
            bias=bias,
        )
        self.output_proj_s = nn.Conv2d(self.stage_dims[0], self.stream_out_channels, kernel_size=1, bias=True)
        self.output_proj_f = nn.Conv2d(self.stage_dims[0], self.stream_out_channels, kernel_size=1, bias=True)
        self.use_bank_feedback = True

    def set_stream_context(self, z_s, z_f):
        self.prompt_state = self.prompt_conditioning(z_s, z_f)

    def _resize_prompt(self, prompt_map, stage_idx, target_size):
        if prompt_map.shape[-2:] != target_size:
            prompt_map = F.interpolate(prompt_map, size=target_size, mode='bilinear', align_corners=False)
        return self.prompt_adapters[stage_idx](prompt_map)

    def _get_prompt_state(self, x_s, x_f):
        return self.prompt_conditioning(x_s, x_f)

    def _build_stage_time_embeddings(self, t, device, batch_size):
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device)
        if t.dim() == 0:
            t = t[None]
        if t.numel() == 1 and batch_size > 1:
            t = t.repeat(batch_size)
        base = sinusoidal_timestep_embedding(t.to(device), self.stage_dims[0]).to(dtype=torch.float32)
        base = self.time_embed(base)
        return [proj(base) for proj in self.stage_time_proj]

    def _build_step_phase(self, t, device, batch_size):
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device)
        if t.dim() == 0:
            t = t[None]
        if t.numel() == 1 and batch_size > 1:
            t = t.repeat(batch_size)
        phase = t.to(device=device, dtype=torch.float32)
        denom = float(max(self.num_train_timesteps - 1, 1))
        phase = torch.clamp(phase / denom, 0.0, 1.0)
        return phase.view(-1, 1, 1, 1)

    @staticmethod
    def _match_bank_context(bank_context, target_size):
        if bank_context is None:
            return None
        if bank_context.shape[-2:] != target_size:
            bank_context = F.interpolate(bank_context, size=target_size, mode='bilinear', align_corners=False)
        return bank_context

    def forward(self, x, t, device):
        struct_latent, func_latent = torch.chunk(x, 2, dim=1)
        base_prompt = self._get_prompt_state(struct_latent, func_latent)
        use_bank_feedback = bool(getattr(self, 'use_bank_feedback', True))
        stage_time_embeddings = self._build_stage_time_embeddings(
            t,
            x.device if torch.is_tensor(x) else device,
            batch_size=x.shape[0],
        )
        step_phase = self._build_step_phase(
            t,
            x.device if torch.is_tensor(x) else device,
            batch_size=x.shape[0],
        )

        s = self.input_proj_s(struct_latent)
        f = self.input_proj_f(func_latent)
        skips = []
        bank_feats = {}
        running_bank = None

        for stage_idx, blocks in enumerate(self.encoder_stages):
            prompt_state = {
                key: self._resize_prompt(value, stage_idx, s.shape[-2:])
                for key, value in base_prompt.items()
            }
            time_emb = stage_time_embeddings[stage_idx]
            for block in blocks:
                s, f, _, aux = block(
                    s,
                    f,
                    prompt_state,
                    time_emb=time_emb,
                    bank_context=self._match_bank_context(running_bank, s.shape[-2:]) if use_bank_feedback else None,
                    step_phase=step_phase,
                )
                if use_bank_feedback:
                    running_bank = aux['bank_feat']
            if use_bank_feedback and stage_idx in self.process_bank_select_stages:
                bank_feats[stage_idx] = 0.5 * (s + f)
            skips.append((s, f))
            s = self.down_s[stage_idx](s)
            f = self.down_f[stage_idx](f)

        latent_prompt = {
            key: self._resize_prompt(value, 3, s.shape[-2:])
            for key, value in base_prompt.items()
        }
        latent_time = stage_time_embeddings[3]
        for block in self.latent_stage:
            s, f, _, aux = block(
                s,
                f,
                latent_prompt,
                time_emb=latent_time,
                bank_context=self._match_bank_context(running_bank, s.shape[-2:]) if use_bank_feedback else None,
                step_phase=step_phase,
            )
            if use_bank_feedback:
                running_bank = aux['bank_feat']
        if use_bank_feedback and 3 in self.process_bank_select_stages:
            bank_feats[3] = 0.5 * (s + f)

        for rev_idx, blocks in enumerate(self.decoder_stages):
            stage_idx = 2 - rev_idx
            s = self.up_s[rev_idx](s)
            f = self.up_f[rev_idx](f)
            skip_s, skip_f = skips[stage_idx]
            if s.shape[-2:] != skip_s.shape[-2:]:
                s = F.interpolate(s, size=skip_s.shape[-2:], mode='bilinear', align_corners=False)
            if f.shape[-2:] != skip_f.shape[-2:]:
                f = F.interpolate(f, size=skip_f.shape[-2:], mode='bilinear', align_corners=False)
            s = self.skip_reduce_s[rev_idx](torch.cat([s, skip_s], dim=1))
            f = self.skip_reduce_f[rev_idx](torch.cat([f, skip_f], dim=1))
            prompt_state = {
                key: self._resize_prompt(value, stage_idx, s.shape[-2:])
                for key, value in base_prompt.items()
            }
            time_emb = stage_time_embeddings[stage_idx]
            for block in blocks:
                s, f, _, aux = block(
                    s,
                    f,
                    prompt_state,
                    time_emb=time_emb,
                    bank_context=self._match_bank_context(running_bank, s.shape[-2:]) if use_bank_feedback else None,
                    step_phase=step_phase,
                )
                if use_bank_feedback:
                    running_bank = aux['bank_feat']

        eps_s = self.output_proj_s(s)
        eps_f = self.output_proj_f(f)
        model_output = torch.cat([eps_s, eps_f], dim=1)
        middle_feat = self.bank(bank_feats, target_size=x.shape[-2:]) if use_bank_feedback else []
        return model_output, middle_feat

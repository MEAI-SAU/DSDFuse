# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from model.diffusers.schedulers.scheduling_ddim import DDIMScheduler
from model.diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from model.diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from model.diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from model.diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from model.diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from model.diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from model.diffusers.schedulers.scheduling_pndm import PNDMScheduler
from model.diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from model.head.dsdfuse_backbone import DualStreamDenoisingBackbone
from model.head.dsdfuse_head import (
    MultiScaleReliabilityGuidedStructureFunctionFusionHead,
)
from model.head.dsdfuse_blocks import ConvNormAct
from model.loss import loss


class StructureAnchoredRefinement(nn.Module):
    """Lightweight structure-anchored residual refinement."""

    def __init__(self, init_gate_bias=-1.5):
        super().__init__()
        self.feature_block = nn.Sequential(
            ConvNormAct(4, 16, kernel_size=3, bias=True),
            ConvNormAct(16, 16, kernel_size=3, bias=True),
        )
        self.gate_predictor = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_refine = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
        )
        self.structure_scale = nn.Parameter(torch.tensor(0.1))
        self.functional_scale = nn.Parameter(torch.tensor(0.05))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.constant_(self.gate_predictor[-2].bias, init_gate_bias)

    def forward(self, raw_fusion, structure_anchor, functional_source):
        if structure_anchor.shape[-2:] != raw_fusion.shape[-2:]:
            structure_anchor = nn.functional.interpolate(
                structure_anchor,
                size=raw_fusion.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        if functional_source.shape[-2:] != raw_fusion.shape[-2:]:
            functional_source = nn.functional.interpolate(
                functional_source,
                size=raw_fusion.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        gate_input = torch.cat([
            raw_fusion,
            structure_anchor,
            functional_source,
            raw_fusion - structure_anchor,
        ], dim=1)
        feat = self.feature_block(gate_input)
        gate = self.gate_predictor(feat)
        residual = self.residual_refine(feat)

        fused = (
            structure_anchor
            + self.structure_scale * gate * (raw_fusion - structure_anchor)
            + self.functional_scale * (1.0 - gate) * (raw_fusion - functional_source)
            + self.residual_scale * gate * residual
        )
        return fused, gate
class SimpleFusionHead(nn.Module):
    """Simple fusion head for baseline ablation: concat + 1x1 conv + residual."""
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.act = nn.GELU()

    def forward(self, x_s, x_f, diffusion_features=None, guidance=None):
        fused = torch.cat([x_s, x_f], dim=1)
        fused = self.conv1(fused)
        fused = self.act(fused)
        residual = self.conv2(fused)
        return fused + residual


class SimpleBackbone(nn.Module):
    """Simple residual convolution block for wo_backbone ablation."""
    def __init__(self, inp_channels, out_channels, dim, num_blocks, num_channel, bias=False, return_middle_feat=False):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.return_middle_feat = return_middle_feat

        # Encoder
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, (blocks, channels) in enumerate(zip(num_blocks, num_channel)):
            stage = nn.ModuleList()
            in_ch = num_channel[i] if i > 0 else inp_channels
            for _ in range(blocks):
                stage.append(nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=bias),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                ))
                in_ch = channels
            self.encoder_stages.append(stage)
            if i < len(num_blocks) - 1:
                self.downsamples.append(nn.Conv2d(channels, num_channel[i+1], kernel_size=2, stride=2, bias=bias))

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for i in range(len(num_blocks) - 2, -1, -1):
            in_dim = num_channel[i + 1]
            out_dim = num_channel[i]
            self.upsamples.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, bias=bias),
                nn.PixelShuffle(2),
            ))
            stage = nn.ModuleList()
            for _ in range(num_blocks[i]):
                stage.append(nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=bias),
                    nn.GELU(),
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=bias),
                ))
            self.decoder_stages.append(stage)

        self.output_proj = nn.Conv2d(num_channel[0], out_channels, kernel_size=1, bias=True)

    def forward(self, x, t=None, device=None):
        # Encoder
        skips = []
        middle_feat = []
        for stage_idx, stage in enumerate(self.encoder_stages):
            for block in stage:
                x = block(x)
            skips.append(x)
            if self.return_middle_feat:
                middle_feat.append(x)
            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)

        # Decoder
        for rev_idx, stage in enumerate(self.decoder_stages):
            stage_idx = len(self.encoder_stages) - 2 - rev_idx
            x = self.upsamples[rev_idx](x)
            if x.shape[-2:] != skips[stage_idx].shape[-2:]:
                x = F.interpolate(x, size=skips[stage_idx].shape[-2:], mode='bilinear', align_corners=False)
            for block in stage:
                x = block(x)
            if self.return_middle_feat:
                middle_feat.append(x)

        x = self.output_proj(x)
        if self.return_middle_feat:
            middle_feat.append(x)
        return x, middle_feat


class FusionPipeline(nn.Module):
    def __init__(
        self,
        LDM_Enc,
        LDM_Dec,
        sample_selected,
        model_selected,
        feat_channels,
        inference_steps=5,
        num_train_timesteps=1000,
        mode='Max',
        fusion_task='MEF',
        channel_emdin=8,
        num_blocks=[4, 4, 4, 4],
        heads=[1, 2, 4, 8],
        bias=False,
        LayerNorm_type='WithBias',
        efficiency=None,
        ablation=None,
    ):
        super().__init__()
        self.mode = mode
        self.LDM_Enc = LDM_Enc
        self.LDM_Dec = LDM_Dec
        self.sample_selected = sample_selected
        self.fusion_task = fusion_task
        self.model_selected = str(model_selected)
        self.diffusion_inference_steps = inference_steps
        self.efficiency = efficiency or {}
        self.ablation = ablation or {}
        self.ablation_mode = self.ablation.get('mode', 'full')
        self.disable_feature_bank = self.ablation_mode in ('wo_bank', 'wo_feature_bank')

        self.structure_residual_cfg = {}
        lightweight_dsb_cfg = {}
        process_bank_cfg = {}
        ms_rgsf_cfg = {}
        if isinstance(efficiency, dict):
            self.structure_residual_cfg = efficiency.get('structure_gated_residual', {}) or {}
            lightweight_dsb_cfg = efficiency.get('lightweight_dsb', {}) or {}
            process_bank_cfg = efficiency.get('process_bank', {}) or {}
            ms_rgsf_cfg = efficiency.get('ms_rgsf_head', efficiency.get('rgsf_head', {})) or {}
        if not isinstance(lightweight_dsb_cfg, dict):
            lightweight_dsb_cfg = {}
        if not isinstance(process_bank_cfg, dict):
            process_bank_cfg = {}
        if not isinstance(ms_rgsf_cfg, dict):
            ms_rgsf_cfg = {}
        self.lightweight_dsb_cfg = lightweight_dsb_cfg
        self.process_bank_cfg = process_bank_cfg
        self.ms_rgsf_cfg = ms_rgsf_cfg
        
        # Ablation: wo_refinement - disable structure-aware reliability refinement
        self.use_structure_gated_residual = bool(self.structure_residual_cfg.get('enabled', True))
        if self.ablation_mode == 'wo_refinement':
            self.use_structure_gated_residual = False
        elif self.ablation_mode == 'baseline':
            self.use_structure_gated_residual = False
        
        self.structure_residual_weight = float(self.structure_residual_cfg.get('loss_weight', 0.0))
        self.structure_residual = StructureAnchoredRefinement(
            init_gate_bias=float(self.structure_residual_cfg.get('init_gate_bias', -1.5))
        )
        self.last_codec_state = None

        if self.model_selected.lower() not in ('dsdfuse', 'dssffuse'):
            raise NotImplementedError('This cleaned branch only keeps the DSDFuse architecture.')

        # Ablation: wo_backbone - use simple residual blocks instead of multi-domain backbone
        if self.ablation_mode == 'baseline':
            self.model = SimpleBackbone(
                inp_channels=channel_emdin * 2,
                out_channels=channel_emdin * 2,
                dim=feat_channels[0],
                num_blocks=num_blocks,
                num_channel=feat_channels,
                bias=bias,
                return_middle_feat=False,
            )
        elif self.ablation_mode == 'wo_backbone':
            self.model = SimpleBackbone(
                inp_channels=channel_emdin * 2,
                out_channels=channel_emdin * 2,
                dim=feat_channels[0],
                num_blocks=num_blocks,
                num_channel=feat_channels,
                bias=bias,
                return_middle_feat=True,
            )
        else:
            disable_mamba = self.ablation_mode == 'wo_mamba'
            mamba_stage_mask = lightweight_dsb_cfg.get('mamba_stage_mask', [False, False, True, True])
            mamba_d_state = int(lightweight_dsb_cfg.get('mamba_d_state', 8))
            mamba_d_conv = int(lightweight_dsb_cfg.get('mamba_d_conv', 3))
            mamba_expand = int(lightweight_dsb_cfg.get('mamba_expand', 1))
            mamba_bidirectional = bool(lightweight_dsb_cfg.get('mamba_bidirectional', False))
            ffn_expansion = float(lightweight_dsb_cfg.get('ffn_expansion', 1.5))
            process_bank_enabled = bool(process_bank_cfg.get('enabled', True))
            process_bank_select_stages = process_bank_cfg.get(
                'select_stages',
                ms_rgsf_cfg.get('bank_select_stages', [1, 2, 3]),
            )
            process_bank_max_features = int(process_bank_cfg.get('max_features', 3))
            process_bank_summary_mode = str(process_bank_cfg.get('summary_mode', 'average'))
            process_bank_collect_after = str(process_bank_cfg.get('collect_after', 'rb'))
            self.model = DualStreamDenoisingBackbone(
                inp_channels=channel_emdin * 2,
                out_channels=channel_emdin * 2,
                dim=feat_channels[0],
                num_blocks=num_blocks,
                num_channel=feat_channels,
                bias=bias,
                num_train_timesteps=num_train_timesteps,
                disable_mamba=disable_mamba,
                mamba_stage_mask=mamba_stage_mask,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_bidirectional=mamba_bidirectional,
                ffn_expansion=ffn_expansion,
                process_bank_enabled=process_bank_enabled,
                process_bank_select_stages=process_bank_select_stages,
                process_bank_max_features=process_bank_max_features,
                process_bank_summary_mode=process_bank_summary_mode,
                process_bank_collect_after=process_bank_collect_after,
            )
        if hasattr(self.model, 'use_bank_feedback'):
            self.model.use_bank_feedback = not self.disable_feature_bank

        # Ablation: wo_fusion_head - use simple fusion head
        if self.ablation_mode == 'baseline':
            self.fusion = SimpleFusionHead(
                in_channels=channel_emdin,
                out_channels=channel_emdin,
                bias=bias,
            )
        elif self.ablation_mode == 'wo_fusion_head':
            self.fusion = SimpleFusionHead(
                in_channels=channel_emdin,
                out_channels=channel_emdin,
                bias=bias,
            )
        else:
            self.fusion = MultiScaleReliabilityGuidedStructureFunctionFusionHead(
                in_channels=channel_emdin,
                out_channels=channel_emdin,
                dim=int(ms_rgsf_cfg.get('dim', feat_channels[0])),
                max_bank_features=int(ms_rgsf_cfg.get('max_bank_features', 3)),
                bank_aggregation=str(ms_rgsf_cfg.get('bank_aggregation', 'scale_softmax')),
                guidance_weight=float(ms_rgsf_cfg.get('guidance_weight', 0.2)),
                temperature=float(ms_rgsf_cfg.get('temperature', 1.0)),
                use_depthwise_refine=bool(ms_rgsf_cfg.get('use_depthwise_refine', True)),
                use_concat_out=bool(ms_rgsf_cfg.get('use_concat_out', False)),
                bias=bias,
            )

        self.scheduler = self._build_scheduler(num_train_timesteps)
        self.pipeline = DenoisePipeline(self.model, self.scheduler, self.sample_selected)
        self.last_modulation_stats = {}

    def _build_scheduler(self, num_train_timesteps):
        if self.sample_selected == 'DDIM':
            return DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        if self.sample_selected == 'ddp-solver':
            return DPMSolverSinglestepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'ddp-solver++':
            return DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Deis':
            return DEISMultistepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Unipc':
            return UniPCMultistepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'LMS':
            return LMSDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Heun':
            return HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'PNDM':
            return PNDMScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Euler':
            return EulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        raise ValueError(f'Unsupported sampler: {self.sample_selected}')

    def set_loss(self, device):
        self.dif_loss = nn.MSELoss().to(device)
        if self.fusion_task in ('MEF', 'MFF'):
            self.fusion_loss = loss.Fusion_loss(mode=self.mode, lambda1=10, lambda2=20, lambda3=20).to(device)
        else:
            self.fusion_loss = loss.Fusion_loss(mode=self.mode, lambda1=10, lambda2=40, lambda3=40).to(device)

    @staticmethod
    def _get_device(module):
        if module is None:
            return None
        if hasattr(module, 'device'):
            return module.device
        if hasattr(module, 'parameters'):
            for param in module.parameters():
                return param.device
        return None

    def _move_to_device(self, module, device):
        if module is None:
            return None
        module_device = self._get_device(module)
        if module_device is None or module_device != device:
            module = module.to(device)
        return module

    def _sync_runtime_modules(self, device):
        self.LDM_Enc = self._move_to_device(self.LDM_Enc, device)
        self.LDM_Dec = self._move_to_device(self.LDM_Dec, device)
        self.model = self._move_to_device(self.model, device)
        self.fusion = self._move_to_device(self.fusion, device)
        self.structure_residual = self._move_to_device(self.structure_residual, device)
        self.pipeline.model = self.model

    def _prepare_latent_inputs(self, struct_feat, func_feat):
        return torch.cat((struct_feat, func_feat), dim=1)

    @staticmethod
    def _summarize_modulation_stats(modulation_stats):
        if not modulation_stats:
            return {}
        summary = {}
        for key, value in modulation_stats.items():
            if torch.is_tensor(value):
                detached = value.detach()
                if detached.numel() == 1:
                    summary[key] = float(detached.item())
                else:
                    summary[key] = {
                        'shape': tuple(detached.shape),
                        'mean': float(detached.mean().item()),
                        'std': float(detached.std(unbiased=False).item()) if detached.numel() > 1 else 0.0,
                    }
            else:
                summary[key] = value
        return summary

    def test_Fusion(self, x_in, device):
        structure_y = x_in[:, :1]
        function_y = x_in[:, 1:]
        with torch.no_grad():
            batch_size = structure_y.shape[0]
            dtype = structure_y.dtype
            self._sync_runtime_modules(device)
            struct_feat, func_feat, codec_state = self.LDM_Enc(structure_y, function_y)
            self.last_codec_state = codec_state
            latent = self._prepare_latent_inputs(struct_feat, func_feat)
            latent_result, _, _, middle_feat = self.pipeline(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                image=latent,
                num_inference_steps=self.diffusion_inference_steps,
                return_dict=False,
            )
            self.last_modulation_stats = {}
            
            # Ablation: wo_bank - use only the last denoised latent feature, not the feature bank
            if self.disable_feature_bank:
                middle_feat = []
            
            fusion_result = self.fusion(struct_feat, func_feat, diffusion_features=middle_feat, guidance=latent_result)
            fusion_result = self.LDM_Dec(fusion_result, codec_state)
            if self.use_structure_gated_residual and self.structure_residual is not None:
                fusion_result, _ = self.structure_residual(fusion_result, structure_y, function_y)
        return fusion_result

    def forward(self, structure_y, function_y):
        batch_size = structure_y.shape[0]
        device = structure_y.device
        dtype = structure_y.dtype

        self._sync_runtime_modules(device)
        struct_feat, func_feat, codec_state = self.LDM_Enc(structure_y, function_y)
        self.last_codec_state = codec_state
        latent = self._prepare_latent_inputs(struct_feat, func_feat)

        latent_result, latent_noise, pre_noise, middle_feat = self.pipeline(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            image=latent,
            num_inference_steps=self.diffusion_inference_steps,
            return_dict=False,
        )
        self.last_modulation_stats = {}

        dif_loss = self.dif_loss(pre_noise, latent_noise)
        
        # Ablation: wo_bank - use only the last denoised latent feature, not the feature bank
        if self.disable_feature_bank:
            middle_feat = []
        
        fusion_result = self.fusion(struct_feat, func_feat, diffusion_features=middle_feat, guidance=latent_result)
        fusion_result = self.LDM_Dec(fusion_result, codec_state)
        residual_gate = None
        if self.use_structure_gated_residual and self.structure_residual is not None:
            fusion_result, residual_gate = self.structure_residual(fusion_result, structure_y, function_y)
        fusion_loss, loss_structure, loss_l1, loss_ssim = self.fusion_loss(structure_y, function_y, fusion_result)
        loss_residual_gate = fusion_result.new_tensor(0.0)
        if residual_gate is not None and self.structure_residual_weight > 0:
            loss_residual_gate = residual_gate.mean()
        total_loss = fusion_loss + dif_loss + self.structure_residual_weight * loss_residual_gate
        return {
            'Fusion': fusion_result,
            'loss': total_loss,
            'loss_structure': loss_structure,
            'loss_gradient': loss_structure,
            'loss_l1': loss_l1,
            'loss_SSIM': loss_ssim,
            'dif_loss': dif_loss,
            'loss_residual_gate': loss_residual_gate,
            'modulation_stats': self.last_modulation_stats,
        }


class DenoisePipeline:
    def __init__(self, model, scheduler, sample_selected):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.sample_selected = sample_selected

    def _run_model(self, image, t, device):
        model_output, middle_feat = self.model(image, t, device)
        return model_output, middle_feat

    def __call__(
        self,
        batch_size,
        device,
        dtype,
        image,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        return_dict: bool = True,
    ) -> Union[Dict, Tuple]:
        if generator is not None and generator.device.type != device.type and device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{device}`, so that `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{device}")` instead.'
            )
            raise RuntimeError("generator.device == 'cpu'", "0.11.0", message)

        self.scheduler.set_timesteps(num_inference_steps)
        noise = torch.randn_like(image, dtype=dtype, device=device) if generator is None else \
                torch.randn(image.shape, dtype=dtype, device=device, generator=generator)
        model_output = noise
        all_middle_feats = []

        if self.sample_selected == 'DDIM':
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=device,
                                     generator=generator).long()
            image = self.scheduler.add_noise(image, noise, timesteps).to(device)
            for t in self.scheduler.timesteps:
                model_output, step_middle_feat = self._run_model(image, t, device)
                if step_middle_feat:
                    all_middle_feats = list(step_middle_feat)
                image = self.scheduler.step(
                    model_output,
                    t,
                    image,
                    eta=eta,
                    use_clipped_model_output=True,
                    generator=generator,
                )['prev_sample']
        elif self.sample_selected in ('ddp-solver', 'ddp-solver++', 'Deis', 'Unipc', 'PNDM'):
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            for t in self.scheduler.timesteps:
                model_output, step_middle_feat = self._run_model(image, t, device)
                if step_middle_feat:
                    all_middle_feats = list(step_middle_feat)
                image = self.scheduler.step(model_output, t, image)['prev_sample']
        elif self.sample_selected == 'Heun':
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            for t in self.scheduler.timesteps:
                image = self.scheduler.scale_model_input(image, t)
                model_output, step_middle_feat = self._run_model(image, t, device)
                if step_middle_feat:
                    all_middle_feats = list(step_middle_feat)
                image = self.scheduler.step(model_output, t, image)['prev_sample']
        elif self.sample_selected == 'LMS':
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            for t in self.scheduler.timesteps:
                image = self.scheduler.scale_model_input(image, t)
                model_output, step_middle_feat = self._run_model(image, t, device)
                if step_middle_feat:
                    all_middle_feats = list(step_middle_feat)
                image = self.scheduler.step(model_output, t, image)['prev_sample']
        elif self.sample_selected == 'Euler':
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            local_generator = torch.manual_seed(0)
            for t in self.scheduler.timesteps:
                image = self.scheduler.scale_model_input(image, t)
                model_output, step_middle_feat = self._run_model(image, t, device)
                if step_middle_feat:
                    all_middle_feats = list(step_middle_feat)
                image = self.scheduler.step(model_output, t, image, generator=local_generator)['prev_sample']
        else:
            raise ValueError(f'Unsupported sampler: {self.sample_selected}')

        middle_feat = all_middle_feats

        if not return_dict:
            return (image, noise, model_output, middle_feat)
        return {'images': image, 'noise': noise}


# Backward-compatible aliases for older experiment scripts and checkpoints.
Fusion_Pipiline = FusionPipeline
DenoisePipiline = DenoisePipeline

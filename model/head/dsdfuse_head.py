import torch
import torch.nn as nn
import torch.nn.functional as F

from model.head.dsdfuse_blocks import extract_feature_entry


class MultiScaleProcessBankAggregation(nn.Module):
    """Aggregate selected multi-scale process features into process evidence."""

    def __init__(
        self,
        dim,
        max_bank_features=3,
        aggregation='scale_softmax',
        guidance_weight=0.2,
        bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.max_bank_features = max(int(max_bank_features), 1)
        self.aggregation = aggregation
        self.guidance_weight = float(guidance_weight)
        self.bias = bias
        self.projections = nn.ModuleList()
        self.scale_score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 1, kernel_size=1, bias=True),
        )
        self.guidance_proj = nn.LazyConv2d(dim, kernel_size=1, bias=bias)

    @staticmethod
    def _normalize_features(features):
        if features is None:
            return []
        if isinstance(features, dict):
            return list(features.values())
        return list(features)

    def _ensure_projection(self, index, device):
        while len(self.projections) <= index:
            self.projections.append(nn.LazyConv2d(self.dim, kernel_size=1, bias=self.bias))
        self.projections[index] = self.projections[index].to(device)
        return self.projections[index]

    def forward(self, diffusion_features, target_size, fallback, guidance=None):
        features = self._normalize_features(diffusion_features)[:self.max_bank_features]
        if not features:
            B_ms = fallback
        else:
            projected = []
            scores = []
            for idx, feat in enumerate(features):
                feat = extract_feature_entry(feat)
                proj = self._ensure_projection(idx, feat.device)(feat)
                if proj.shape[-2:] != target_size:
                    proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=False)
                projected.append(proj)
                scores.append(self.scale_score(proj))

            if self.aggregation == 'average' or len(projected) == 1:
                B_ms = sum(projected) / float(len(projected))
            else:
                weights = torch.softmax(torch.cat(scores, dim=1), dim=1)
                B_ms = sum(weights[:, idx:idx + 1] * feat for idx, feat in enumerate(projected))

        if guidance is not None and self.guidance_weight > 0:
            g = self.guidance_proj(guidance)
            if g.shape[-2:] != target_size:
                g = F.interpolate(g, size=target_size, mode='bilinear', align_corners=False)
            B_ms = B_ms + self.guidance_weight * g
        return B_ms


class MultiScaleReliabilityGuidedStructureFunctionFusionHead(nn.Module):
    """MS-RGSF Head: reliability-guided fusion of structure, function, and process bank."""

    def __init__(
        self,
        in_channels,
        out_channels,
        dim=16,
        max_bank_features=3,
        bank_aggregation='scale_softmax',
        guidance_weight=0.2,
        temperature=1.0,
        use_depthwise_refine=True,
        use_concat_out=False,
        bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.temperature = max(float(temperature), 1e-6)
        self.use_concat_out = bool(use_concat_out)

        self.S_proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1, bias=bias),
            nn.GELU(),
        )
        self.F_proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1, bias=bias),
            nn.GELU(),
        )
        self.bank_aggregator = MultiScaleProcessBankAggregation(
            dim=dim,
            max_bank_features=max_bank_features,
            aggregation=bank_aggregation,
            guidance_weight=guidance_weight,
            bias=bias,
        )
        self.reliability_estimator = nn.Sequential(
            nn.Conv2d(dim * 5, dim, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, 3, kernel_size=1, bias=True),
        )
        groups = dim if use_depthwise_refine else 1
        self.residual_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=groups, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )
        out_in_channels = dim * 3 if self.use_concat_out else dim
        self.out_proj = nn.Conv2d(out_in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, structure_feat, function_feat, diffusion_features=None, guidance=None):
        S = self.S_proj(structure_feat)
        F_feat = self.F_proj(function_feat)
        target_size = S.shape[-2:]

        B_ms = self.bank_aggregator(
            diffusion_features,
            target_size=target_size,
            fallback=0.5 * (S + F_feat),
            guidance=guidance,
        )
        D = torch.abs(S - F_feat)
        C = S * F_feat
        R = self.reliability_estimator(torch.cat([S, F_feat, B_ms, D, C], dim=1))
        R = torch.softmax(R / self.temperature, dim=1)
        R_s, R_f, R_b = R[:, 0:1], R[:, 1:2], R[:, 2:3]

        z0 = R_s * S + R_f * F_feat + R_b * B_ms
        z = z0 + self.residual_refine(z0)
        if self.use_concat_out:
            z = torch.cat([z, S, F_feat], dim=1)
        z_fuse = self.out_proj(z)
        return z_fuse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry import generate_sdf_field, sample_sdf_at_points, render_density_field, \
    excluded_volume_loss_vectorized, ligand_self_repulsion_loss, ligand_protein_shell_loss
from torch_scatter import scatter_mean, scatter_sum

def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
    if mode == "none":
        offset = 0.0
        pass
    elif mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset

class GeometryEnergyGuidedLoss(nn.Module):
    def __init__(self, renderer, energy_projector, render_loss_fn, cfg, pos_norm):
        super().__init__()
        self.renderer = renderer
        self.energy_projector = energy_projector
        self.render_loss_fn = render_loss_fn

        # 从配置文件中读取参数
        # self.weight_sdf = cfg.sdf_weight
        # self.weight_field = cfg.graph_field_weight
        self.weight_steric = cfg.geom_energy_cfg.steric_weight
        self.weight_self = cfg.geom_energy_cfg.self_repulsion_weight
        self.weight_shell = cfg.geom_energy_cfg.shell_weight
        self.weight_render = cfg.geom_energy_cfg.render_weight

        self.pos_norm = pos_norm

        self.r_max = cfg.r_max
        self.knn = cfg.knn
        self.probe = cfg.geom_energy_cfg.probe


    def forward(self, batch, pred):
        offset = pred["offset"]
        pred_world = pred["pred_ligand_pos"] + offset[batch["ligand_element_batch"]]

        protein_pos, ligand_pos, _ = center_pos(batch["protein_pos"], batch["ligand_pos"], batch["protein_element_batch"], batch["ligand_element_batch"])

        pred_w_pos = pred_world * torch.tensor(
            self.pos_norm, dtype=torch.float32, device=offset.device
        )
        pred_pos = pred["pred_ligand_pos"] * torch.tensor(
            self.pos_norm, dtype=torch.float32, device=protein_pos.device
        )

        protein_pos = protein_pos * torch.tensor(
            self.pos_norm, dtype=torch.float32, device=protein_pos.device
        )
        # ligand_pos = ligand_pos * torch.tensor(
        #     self.pos_norm, dtype=torch.float32, device=protein_pos.device
        # )

        protein_w_pos = batch["protein_pos"] * torch.tensor(
            self.pos_norm, dtype=torch.float32, device=protein_pos.device
        )
        ligand_w_pos = batch["ligand_pos"] * torch.tensor(
            self.pos_norm, dtype=torch.float32, device=protein_pos.device
        )

        # === 1. 几何场约束 (SDF + Graph Field) ===
        # loss_sdf, loss_field = self.compute_sdf_graph_field_loss(pred_pos, ligand_pos, protein_pos)

        # total_loss = self.weight_sdf * loss_sdf + self.weight_field * loss_field

        # # # ===== steric clash + self-repulsion =====
        protein_batch = batch["protein_element_batch"]
        ligand_batch = batch["ligand_element_batch"]

        loss_steric = excluded_volume_loss_vectorized(
            ligand_pos=pred_pos, ligand_batch=ligand_batch,
            protein_pos=protein_pos, protein_batch=protein_batch,
            protein_Z=batch.get("protein_Z", None),  # 如果没有就 None
            r_max=self.r_max, probe=self.probe, margin=0.0, reduction="mean", clamp_max=10.0
        )

        loss_self = ligand_self_repulsion_loss(
            ligand_pos=pred_pos, ligand_batch=ligand_batch,
            k=self.knn, d_min=1.2, power=2, clamp_max=10.0
        )

        loss_shell = ligand_protein_shell_loss(
            ligand_pos=pred_pos, ligand_batch=ligand_batch,
            protein_pos=protein_pos, protein_batch=protein_batch,
            d0=3.5, sigma=0.8)

        total_loss = self.weight_steric * loss_steric + self.weight_self * loss_self + self.weight_shell * loss_shell

        # === 2. 渲染监督 ===
        pred_views = self.renderer(
            protein_pos=protein_w_pos, protein_types=batch["protein_atom_feature"],
            batch_protein=batch["protein_element_batch"],
            ligand_pos=pred_w_pos, ligand_types=pred['ligand_v_recon'],
            batch_ligand=batch["ligand_element_batch"],
        )
        # === 3. 能量投影反应相互作用 ===
        physics_pred = self.energy_projector(
            protein_w_pos, batch["protein_atom_feature"], batch["protein_element_batch"],
            pred_w_pos, pred['ligand_v_recon'], batch["ligand_element_batch"]
        )
        pred_views['energy'] = physics_pred["energy_maps"]

        with torch.no_grad():
            tgt_views = self.renderer(
                protein_pos=protein_w_pos, protein_types=batch["protein_atom_feature"],
                batch_protein=batch["protein_element_batch"],
                ligand_pos=ligand_w_pos, ligand_types=batch["ligand_atom_feature_full"],
                batch_ligand=batch["ligand_element_batch"], target=True
            )

        render_losses = self.render_loss_fn(pred_views, tgt_views)
        total_loss += self.weight_render * render_losses["total"]


        return {
            'total': total_loss,
            'energy_clash': render_losses["energy_clash"],
            'energy_same_sign': render_losses["energy_same_sign"],
            'energy_hbond': render_losses["energy_hbond"],
            'energy_hydro': render_losses["energy_hydro"],
            'energy_ionic': render_losses["energy_ionic"],
            'energy_view_var': render_losses["energy_view_var"],
            'loss_steric': loss_steric,
            'loss_self_repulsion': loss_self,
            'loss_density': render_losses["loss_density"],
            'loss_depth': render_losses["loss_depth"],
            'loss_energy': render_losses["loss_energy"],
        }

    # def compute_sdf_graph_field_loss(self, pred_pos, ligand_pos, protein_pos):
    #     sdf_protein = generate_sdf_field(protein_pos)
    #     sdf_pred = sample_sdf_at_points(pred_pos, sdf_protein)
    #     sdf_tgt = sample_sdf_at_points(ligand_pos, sdf_protein)
    #     loss_sdf = F.l1_loss(sdf_pred, sdf_tgt, reduction="mean")
    #     field_pred = render_density_field(pred_pos, sigma=1.0)
    #     field_true = render_density_field(ligand_pos, sigma=1.0)
    #
    #     def _norm(v):
    #         m = v.mean()
    #         s = v.std().clamp_min(1e-6)
    #         return (v - m) / s
    #
    #     field_pred = _norm(field_pred)
    #     field_true = _norm(field_true)
    #
    #     loss_field = F.mse_loss(field_pred, field_true, reduction='mean')
    #
    #     return loss_sdf, loss_field


class DifferentiableRenderLoss(nn.Module):
    """
    Computes combined L1 + SSIM loss between predicted and target render views.
    """
    def __init__(self, weight_density=1.0, weight_depth=0.5, weight_energy=0.5):
        super().__init__()
        self.weight_density = weight_density
        self.weight_depth = weight_depth
        self.weight_energy = weight_energy

    def fg_weighted_charb_loss(self, pred, tgt, thr=1e-4, eps=1e-3, alpha=2.0):
        """
        Weight pixel error by foreground strength, so it doesn't vanish.
        weight = ((tgt + pred)/2) ^ alpha  (only where foreground exists)
        """
        if pred.shape != tgt.shape:
            raise ValueError("pred/tgt shape mismatch")

        fg = ((tgt > thr) | (pred > thr)).float()
        w = ((tgt.clamp_min(0) + pred.clamp_min(0)) * 0.5).pow(alpha) * fg

        diff = pred - tgt
        loss = torch.sqrt(diff * diff + eps * eps)

        return (loss * w).sum() / w.sum().clamp_min(1.0)

    def soft_dice_loss(self, pred, tgt, thr=1e-4, eps=1e-6):
        # pred,tgt: (B,V,H,W) or (V,H,W), assume >=0
        # foreground mask (union) to avoid background domination
        mask = ((tgt > thr) | (pred > thr)).float()
        p = pred * mask
        t = tgt * mask
        inter = (p * t).sum(dim=(-2, -1))
        denom = p.sum(dim=(-2, -1)) + t.sum(dim=(-2, -1))
        dice = (2.0 * inter + eps) / (denom + eps)
        return (1.0 - dice).mean()

    def forward(self, pred_views, tgt_views):
        loss = 0.0

        loss_charbonnier = self.fg_weighted_charb_loss(pred_views["density"], tgt_views["density"], thr=1e-4, eps=1e-3)
        loss_dice = self.soft_dice_loss(pred_views["density"], tgt_views["density"], thr=1e-4, eps=1e-6)
        loss_density = 10.0 * loss_charbonnier + 0.4 * loss_dice
        loss += self.weight_density * loss_density
        occ_mask = (tgt_views["density"] > 1e-4).float()
        occ_e = ((pred_views["density"] > 1e-4) | (tgt_views["density"] > 1e-4)).float()

        charb_loss = self.fg_weighted_charb_loss(pred_views['depth'], tgt_views['depth'], thr=1e-4, eps=1e-3)
        grad_loss= mask_gradient_loss(pred_views['depth'], tgt_views['depth'], occ_mask)
        # Sparse multi-view consistency: 只用 0, mid, last 三帧，节省算力
        if pred_views['depth'].size(1) >= 3:
            mid = pred_views['depth'].size(1) // 2
            pred_pair1 = pred_views['depth'][:, 0] - pred_views['depth'][:, mid]
            pred_pair2 = pred_views['depth'][:, mid] - pred_views['depth'][:, -1]
            tgt_pair1 = tgt_views['depth'][:, 0] - tgt_views['depth'][:, mid]
            tgt_pair2 = tgt_views['depth'][:, mid] - tgt_views['depth'][:, -1]
            m1 = occ_mask[:, 0] * occ_mask[:, mid]
            m2 = occ_mask[:, mid] * occ_mask[:, -1]
            consistency_loss = 0.5 * (masked_l1(pred_pair1, tgt_pair1, m1) + masked_l1(pred_pair2, tgt_pair2, m2))
        elif pred_views['depth'].size(1) == 2:
            m = occ_mask[:, 1] * occ_mask[:, 0]
            consistency_loss = masked_l1(pred_views['depth'][:, 1] - pred_views['depth'][:, 0],
                         tgt_views['depth'][:, 1] - tgt_views['depth'][:, 0], m)

        else:
            consistency_loss = torch.tensor(0.0)
        loss_depth = 0.3 * charb_loss + 0.3 * grad_loss + 0.4 * consistency_loss

        loss += self.weight_depth * loss_depth

        # use union-occupancy for energy
        loss_energy, stats = compute_render_quality_loss(pred_views["energy"])

        loss += self.weight_energy * loss_energy

        return {"total": loss,
                "energy_clash": stats["render_clash"],
                "energy_same_sign": stats["render_same_sign"],
                "energy_hbond": stats["render_hbond_reward"],
                "energy_hydro": stats["render_hydrophobic_reward"],
                "energy_ionic": stats["render_ionic_reward"],
                "energy_view_var": stats["render_view_var"],
                "loss_density": loss_density if 'loss_density' in locals() else torch.tensor(0.0),
                "loss_depth": loss_depth if 'loss_depth' in locals() else torch.tensor(0.0),
                "loss_energy": loss_energy if 'loss_energy' in locals() else torch.tensor(0.0),
                }


def mask_gradient_loss(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred, tgt: (V,H,W) or (B,V,H,W) 目前按 (V,H,W) 使用
    """
    diff = (pred - tgt).abs()

    # dx (along W)
    diff_dx = diff[..., :, 1:] - diff[..., :, :-1]
    mask_dx = mask[..., :, 1:] * mask[..., :, :-1]
    loss_dx = (diff_dx.abs() * mask_dx).sum() / mask_dx.sum().clamp_min(1.0)

    # dy (along H)
    diff_dy = diff[..., 1:, :] - diff[..., :-1, :]
    mask_dy = mask[..., 1:, :] * mask[..., :-1, :]
    loss_dy = (diff_dy.abs() * mask_dy).sum() / mask_dy.sum().clamp_min(1.0)

    return loss_dx + loss_dy

def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return ((pred - tgt).abs() * mask).sum() / denom


def compute_render_quality_loss(energy_maps):
    """
    energy_maps: [B, V, C, G, G]
    C order:
        0 hbond_donor
        1 hbond_acceptor
        2 hydrophobic
        3 positive_ionic
        4 negative_ionic
        5 same_sign
        6 clash
    """
    maps = energy_maps

    # favorable channels
    hbond_map = maps[:, :, 0] + maps[:, :, 1]   # [B,V,G,G]
    hydro_map = maps[:, :, 2]
    ionic_map = maps[:, :, 3] + maps[:, :, 4]

    # unfavorable channels
    same_sign_map = maps[:, :, 5]
    clash_map = maps[:, :, 6]

    # ---- spatial pooling ----
    hbond_view = hbond_map.mean(dim=(-2, -1))       # [B,V]
    hydro_view = hydro_map.mean(dim=(-2, -1))
    ionic_view = ionic_map.mean(dim=(-2, -1))
    same_sign_view = same_sign_map.mean(dim=(-2, -1))
    clash_view = clash_map.mean(dim=(-2, -1))

    # ---- saturated favorable rewards ----
    hbond_reward = 1.0 - torch.exp(-1.2 * hbond_view)
    hydro_reward = 1.0 - torch.exp(-0.8 * hydro_view)
    ionic_reward = 1.0 - torch.exp(-1.0 * ionic_view)

    # ---- penalties ----
    # keep clash slightly harder
    clash_penalty = clash_view
    same_sign_penalty = same_sign_view

    clash_mean = clash_penalty.mean()
    same_mean = same_sign_penalty.mean()

    # ---- badness-controlled reward gate ----
    # if clash / same-sign is still high, reward should almost not contribute
    badness = clash_mean + 0.5 * same_mean
    reward_gate = torch.sigmoid(15.0 * (0.1 - badness))

    # ---- favorable summary ----
    favorable_reward = (
        1.0 * hbond_reward.mean()
        + 0.5 * hydro_reward.mean()
        + 0.6 * ionic_reward.mean()
    )

    # cap reward so it can only refine, not dominate
    favorable_reward = torch.clamp(favorable_reward, max=0.3)

    # ---- multi-view consistency ----
    # still useful, but keep weak
    view_var_hbond = hbond_view.var(dim=1).mean()
    view_var_hydro = hydro_view.var(dim=1).mean()
    view_var_ionic = ionic_view.var(dim=1).mean()
    view_var = (view_var_hbond + view_var_hydro + view_var_ionic) / 3.0

    # ---- penalty-dominant render loss ----
    loss = (
        2.0 * clash_mean
        + 0.8 * same_mean
        - reward_gate * favorable_reward
        + 0.05 * view_var
    )
    stats = {
        "render_clash": clash_penalty.mean().detach(),
        "render_same_sign": same_sign_penalty.mean().detach(),
        "render_hbond_reward": hbond_reward.mean().detach(),
        "render_hydrophobic_reward": hydro_reward.mean().detach(),
        "render_ionic_reward": ionic_reward.mean().detach(),
        "render_view_var": view_var.detach(),
    }
    return loss, stats


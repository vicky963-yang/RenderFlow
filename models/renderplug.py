import torch
import torch.nn as nn
import torch.nn.functional as F
from models.energy_pro import map_pred_to_radius_charge, map_protein_to_radius_charge
from torch_geometric.utils import to_dense_batch

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1,  False): 0,
    (6,  False): 1,
    (6,  True):  2,
    (7,  False): 3,
    (7,  True):  4,
    (8,  False): 5,
    (8,  True):  6,
    (9,  False): 7,
    (15, False): 8,
    (15, True):  9,
    (16, False): 10,
    (16, True):  11,
    (17, False): 12,
}

ligand_radius_by_index = [
    1.20,  # 0: H
    1.70,  # 1: C (non-aromatic)
    1.70,  # 2: C (aromatic)
    1.55,  # 3: N (non-aromatic)
    1.55,  # 4: N (aromatic)
    1.52,  # 5: O (non-aromatic)
    1.52,  # 6: O (aromatic, rare)
    1.47,  # 7: F
    1.80,  # 8: P (non-aromatic)
    1.80,  # 9: P (aromatic, rare)
    1.80,  # 10: S (non-aromatic)
    1.80,  # 11: S (aromatic)
    1.75,  # 12: Cl
]

ligand_charge_by_index = [
    +0.05,  # 0: H (slightly positive)
     0.00,  # 1: C non-aromatic
     0.00,  # 2: C aromatic
    -0.25,  # 3: N non-aromatic
    -0.30,  # 4: N aromatic
    -0.35,  # 5: O non-aromatic
    -0.35,  # 6: O aromatic (rare)
    -0.10,  # 7: F
    +0.10,  # 8: P non-aromatic
    +0.10,  # 9: P aromatic
    +0.05,  # 10: S non-aromatic
    +0.05,  # 11: S aromatic
    -0.05,  # 12: Cl
]


class TetSphereRenderer(nn.Module):
    """
    TetSphere-leaning splatting, fully tensorized, no Python loops:

    - FOV: centered ligand, radius = scene_radius_A (Å)
    - Image: G x G
    - Primitive: sphere footprint (sigma_px derived from vdW radius)
    - Kernel: separable Gaussian in screen space
    - Accumulation: einsum / batched matmul -> density and depth

    Inputs are ragged (pos + batch); internally packed to dense (B,Nmax,3).
    """

    def __init__(
        self,
        cfg,
        n_views: int = 8,
        image_size: int = 64,
        scene_radius_A: float = 10.0,     # your 10 Å radius FOV
        sigma_gain: float = 1.0,          # footprint gain
        eps: float = 1e-6,
        normalize_1d: bool = True,        # make each point integrate ~1 on grid
    ):
        super().__init__()
        self.cfg = cfg
        self.n_views = n_views
        self.G = image_size
        self.scene_radius_A = float(scene_radius_A)
        self.sigma_gain = float(sigma_gain)
        self.eps = float(eps)
        self.normalize_1d = bool(normalize_1d)

        # Pixel coordinate grid: 0..G-1
        x = torch.arange(self.G, dtype=torch.float32)
        self.register_buffer("pix", x)  # [G]

    def _rotation_mats(self, device):
        V = self.n_views
        angles = torch.linspace(0, 2 * torch.pi * (1 - 1 / V), steps=V, device=device)
        c, s = torch.cos(angles), torch.sin(angles)
        R = torch.stack([
            torch.stack([c, torch.zeros_like(c), s], dim=-1),
            torch.stack([c.new_zeros(V), c.new_ones(V), c.new_zeros(V)], dim=-1),
            torch.stack([-s, torch.zeros_like(c), c], dim=-1),
        ], dim=-2)  # [V,3,3]
        return R

    def _world_to_pixel(self, coords_bvnc):
        """
        coords_bvnc: [B,V,N,3] in Å (centered)
        returns x_pix, y_pix: [B,V,N] float in [0, G-1]
        """
        G = self.G
        # normalized to [-1,1]
        xn = coords_bvnc[..., 0] / self.scene_radius_A
        yn = coords_bvnc[..., 1] / self.scene_radius_A
        # map to pixel
        x_pix = (xn * (G / 2.0) + (G - 1) / 2.0).clamp(0.0, G - 1.0)
        y_pix = (yn * (G / 2.0) + (G - 1) / 2.0).clamp(0.0, G - 1.0)
        return x_pix, y_pix

    def _sigmaA_to_sigmapx(self, radius_A):
        """
        radius_A: [B,N] or [B,V,N]
        sigma_px = gain * radius_A * (G / (2*R))
        """
        return self.sigma_gain * radius_A * (self.G / (2.0 * self.scene_radius_A))

    def _gauss_1d(self, center_pix, sigma_pix):
        """
        center_pix: [B,V,N]
        sigma_pix:  [B,V,N]
        return g:   [B,V,N,G]
        """
        # pix: [G] -> broadcast
        pix = self.pix.to(center_pix.device)  # [G]
        # dx: [B,V,N,G]
        dx = pix.view(1, 1, 1, self.G) - center_pix.unsqueeze(-1)
        inv2s2 = 1.0 / (2.0 * sigma_pix.clamp_min(1e-3).unsqueeze(-1) ** 2)
        g = torch.exp(-(dx * dx) * inv2s2)  # [B,V,N,G]

        if self.normalize_1d:
            g = g / g.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return g

    def _pack_dense(self, pos, batch, feat_or_none=None, max_nodes=None):
        """
        Pack ragged -> dense using to_dense_batch.
        pos: [N,3]
        batch: [N]
        feat_or_none: [N,C] or None
        return:
          pos_d:  [B,Nmax,3]
          mask_d: [B,Nmax] bool
          feat_d: [B,Nmax,C] or None
        """

        pos_d, mask_d = to_dense_batch(pos, batch, max_num_nodes=max_nodes)  # [B,Nmax,3], [B,Nmax]
        if feat_or_none is None:
            return pos_d, mask_d, None
        feat_d, _ = to_dense_batch(feat_or_none, batch, max_num_nodes=max_nodes)
        return pos_d, mask_d, feat_d

    def forward(
        self,
        protein_pos, protein_types, batch_protein,
        ligand_pos, ligand_types, batch_ligand,
        target: bool = False,
        return_separate: bool = False,
    ):
        """
        protein_feat: [Np,27]
        ligand_types:
          - target=True:  [Nl] int
          - target=False: [Nl,13] logits/probs

        Returns:
          dict with
            density: [B,V,G,G]
            depth:   [B,V,G,G]
          optionally:
            density_pro, density_lig, depth_pro, depth_lig
        """
        device = ligand_pos.device
        V = self.n_views

        if target:
            ligand_types = F.one_hot(ligand_types, num_classes=len(ligand_radius_by_index)).float()
        # --- radii/charges (protein from feature, ligand from probs) ---
        ligand_radius, _ = map_pred_to_radius_charge(ligand_types, ligand_radius_by_index, ligand_charge_by_index,
                                                     use_soft=True)
        max_ligand_nodes = len(ligand_radius_by_index)
        prot_r, _ = map_protein_to_radius_charge(protein_types)  # [Np]

        # --- pack to dense (B,Nmax,*) ---
        p_pos, p_mask, p_feat = self._pack_dense(
            protein_pos, batch_protein, feat_or_none=protein_types, max_nodes=self.cfg.model.protein_atom_feature_dim
        )
        l_pos, l_mask, _ = self._pack_dense(
            ligand_pos, batch_ligand, feat_or_none=None, max_nodes=max_ligand_nodes
        )
        # pack radii too
        p_r, _ = to_dense_batch(prot_r, batch_protein, max_num_nodes=self.cfg.model.protein_atom_feature_dim)  # [B,Npmax]
        l_r, _ = to_dense_batch(ligand_radius, batch_ligand, max_num_nodes=max_ligand_nodes)     # [B,Nlmax]

        # B = l_pos.size(0)

        # --- rotations ---
        R = self._rotation_mats(device)  # [V,3,3]

        # rotate: [B,N,3] -> [B,V,N,3]
        p_coords = torch.einsum("bnc,vdc->bvnd", p_pos, R)  # [B,V,Np,3]
        l_coords = torch.einsum("bnc,vdc->bvnd", l_pos, R)  # [B,V,Nl,3]

        # project to pixel centers
        px, py = self._world_to_pixel(p_coords)  # [B,V,Np]
        lx, ly = self._world_to_pixel(l_coords)  # [B,V,Nl]

        # sigma in pixels from vdW radius (TetSphere-ish sphere footprint)
        p_sigma = self._sigmaA_to_sigmapx(p_r).unsqueeze(1).expand(-1, V, -1)  # [B,V,Np]
        l_sigma = self._sigmaA_to_sigmapx(l_r).unsqueeze(1).expand(-1, V, -1)  # [B,V,Nl]

        # 1D kernels
        gx_p = self._gauss_1d(px, p_sigma)  # [B,V,Np,G]
        gy_p = self._gauss_1d(py, p_sigma)  # [B,V,Np,G]
        gx_l = self._gauss_1d(lx, l_sigma)  # [B,V,Nl,G]
        gy_l = self._gauss_1d(ly, l_sigma)  # [B,V,Nl,G]

        # masks -> float amplitude
        ap = p_mask.float().unsqueeze(1)  # [B,1,Np]
        al = l_mask.float().unsqueeze(1)  # [B,1,Nl]

        # density = sum_n (a_n * gx[:,n,g] * gy[:,n,h])
        # implement as einsum over N (fully tensorized)
        density_pro = torch.einsum("bvnG,bvnH->bvGH", gx_p * ap.unsqueeze(-1), gy_p)  # [B,V,G,G]
        density_lig = torch.einsum("bvnG,bvnH->bvGH", gx_l * al.unsqueeze(-1), gy_l)  # [B,V,G,G]
        density = density_pro + density_lig

        # depth numerator: sum_n (a_n * z_n * gx * gy)
        pz = p_coords[..., 2]  # [B,V,Np]
        lz = l_coords[..., 2]  # [B,V,Nl]
        depth_num_pro = torch.einsum("bvnG,bvnH->bvGH", gx_p * (ap * pz).unsqueeze(-1), gy_p)
        depth_num_lig = torch.einsum("bvnG,bvnH->bvGH", gx_l * (al * lz).unsqueeze(-1), gy_l)
        depth_num = depth_num_pro + depth_num_lig
        depth = depth_num / density.clamp_min(self.eps)

        # # optional per-view normalization (keep as you like)
        # dens_flat = density.view(B, V, -1)
        # max_per_view = dens_flat.max(dim=-1, keepdim=True)[0].clamp_min(self.eps)
        # density = (dens_flat / max_per_view).view(B, V, G, G)

        out = {"density": density, "depth": depth, "energy": None}
        if return_separate:
            out.update({
                "density_pro": density_pro,
                "density_lig": density_lig,
                "depth_pro": depth_num_pro / density_pro.clamp_min(self.eps),
                "depth_lig": depth_num_lig / density_lig.clamp_min(self.eps),
            })
        return out




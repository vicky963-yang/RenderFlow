import math
import torch
import torch.nn as nn
import torch.nn.functional as F


ligand_radius_by_index = [
    1.20,  # 0: H
    1.70,  # 1: C (non-aromatic)
    1.70,  # 2: C (aromatic)
    1.55,  # 3: N (non-aromatic)
    1.55,  # 4: N (aromatic)
    1.52,  # 5: O (non-aromatic)
    1.52,  # 6: O (aromatic)
    1.47,  # 7: F
    1.80,  # 8: P
    1.80,  # 9: P(aromatic)
    1.80,  # 10: S
    1.80,  # 11: S(aromatic)
    1.75,  # 12: Cl
]

ligand_charge_by_index = [
    +0.05,  # H
     0.00,  # C
     0.00,  # C(ar)
    -0.25,  # N
    -0.30,  # N(ar)
    -0.35,  # O
    -0.35,  # O(ar)
    -0.10,  # F
    +0.10,  # P
    +0.10,  # P(ar)
    +0.05,  # S
    +0.05,  # S(ar)
    -0.05,  # Cl
]

protein_element_radius = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    16: 1.80,
    34: 1.90,
}
protein_element_charge = {
    1:  +0.05,
    6:   0.00,
    7:  -0.35,
    8:  -0.45,
    16: +0.10,
    34: +0.15,
}

AA_NAME_SYM = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}
AA_ORDER = list(AA_NAME_SYM.values())

AA_CHARGE = [
    0, 0, -1, -1, 0, 0, 0.1, 0, 1, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0
]


def map_protein_to_radius_charge(protein_feat):
    """
    protein_feat: [N,27] = element(6) + amino_acid(20) + backbone(1)
    """
    element_onehot = protein_feat[:, :6]
    aa_onehot = protein_feat[:, 6:26]
    is_backbone = protein_feat[:, 26:27]

    element_idx = element_onehot.argmax(dim=-1)
    elements = [1, 6, 7, 8, 16, 34]

    radius_lut = torch.tensor(
        [protein_element_radius[z] for z in elements],
        device=protein_feat.device,
        dtype=torch.float32
    )
    charge_lut = torch.tensor(
        [protein_element_charge[z] for z in elements],
        device=protein_feat.device,
        dtype=torch.float32
    )

    radius = radius_lut[element_idx]
    charge = charge_lut[element_idx]

    aa_charge_vec = torch.tensor(AA_CHARGE, device=protein_feat.device, dtype=torch.float32)
    aa_charge_term = (aa_onehot * aa_charge_vec).sum(dim=-1)

    charge = charge + aa_charge_term
    radius = radius * (1.0 + 0.1 * is_backbone.squeeze(-1))
    charge = charge * (1.0 + 0.2 * is_backbone.squeeze(-1))

    return radius, charge


def map_pred_to_radius_charge(pred_ligand_v, radius_vec, charge_vec, use_soft=True):
    radius_vec = torch.as_tensor(radius_vec, device=pred_ligand_v.device, dtype=torch.float32)
    charge_vec = torch.as_tensor(charge_vec, device=pred_ligand_v.device, dtype=torch.float32)

    if use_soft:
        ligand_radius = pred_ligand_v.float() @ radius_vec
        ligand_charge = pred_ligand_v.float() @ charge_vec
    else:
        idx = pred_ligand_v.argmax(dim=-1)
        ligand_radius = radius_vec[idx]
        ligand_charge = charge_vec[idx]

    return ligand_radius, ligand_charge


class SparsePharmacophoreEnergyRenderer(nn.Module):
    CHANNELS = [
        "hbond_donor",
        "hbond_acceptor",
        "hydrophobic",
        "positive_ionic",
        "negative_ionic",
        "same_sign",
        "clash",
    ]

    def __init__(
        self,
        cfg,
        n_views: int = 8,
        image_size: int = 64,
        scene_radius_A: float = 10.0,
        cutoff_A: float = 6.0,
        sigma_gain: float = 1.0,
        alpha_scale: float = 0.35,
        eps: float = 1e-6,
        normalize_1d: bool = True,
        use_alpha_blend_approx: bool = True,
        gaussian_truncate: float = 3.0,
        use_depth_attenuation: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_views = n_views
        self.G = image_size
        self.scene_radius_A = float(scene_radius_A)
        self.cutoff_A = float(cutoff_A)
        self.sigma_gain = float(sigma_gain)
        self.alpha_scale = float(alpha_scale)
        self.eps = float(eps)
        self.normalize_1d = bool(normalize_1d)
        self.use_alpha_blend_approx = bool(use_alpha_blend_approx)
        self.gaussian_truncate = float(gaussian_truncate)
        self.use_depth_attenuation = bool(use_depth_attenuation)

        self.num_channels = len(self.CHANNELS)

        # protein priors aligned with protein_types[:, :6] = [H,C,N,O,S,Se]
        self.register_buffer(
            "PROT_HYDRO",
            torch.tensor([0.05, 1.00, 0.15, 0.00, 0.35, 0.25], dtype=torch.float32)
        )
        self.register_buffer(
            "PROT_DONOR",
            torch.tensor([0.00, 0.00, 0.80, 0.20, 0.10, 0.05], dtype=torch.float32)
        )
        self.register_buffer(
            "PROT_ACCEPTOR",
            torch.tensor([0.00, 0.00, 0.35, 1.00, 0.25, 0.15], dtype=torch.float32)
        )
        self.register_buffer(
            "AA_CHARGE_VEC",
            torch.tensor(AA_CHARGE, dtype=torch.float32)
        )

    # ------------------------------------------------------------
    # geometry utils
    # ------------------------------------------------------------
    def _rotation_mats(self, device):
        V = self.n_views
        angles = torch.linspace(0, 2 * torch.pi * (1 - 1 / V), steps=V, device=device)
        c, s = torch.cos(angles), torch.sin(angles)
        R = torch.stack([
            torch.stack([c, torch.zeros_like(c), s], dim=-1),
            torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)], dim=-1),
            torch.stack([-s, torch.zeros_like(c), c], dim=-1),
        ], dim=-2)  # [V,3,3]
        return R

    def _world_to_pixel(self, coords_vnc):
        """
        coords_vnc: [V,N,3]
        returns:
            x_pix, y_pix: [V,N]
        """
        G = self.G
        xn = coords_vnc[..., 0] / self.scene_radius_A
        yn = coords_vnc[..., 1] / self.scene_radius_A
        x_pix = (xn * (G / 2.0) + (G - 1) / 2.0).clamp(0.0, G - 1.0)
        y_pix = (yn * (G / 2.0) + (G - 1) / 2.0).clamp(0.0, G - 1.0)
        return x_pix, y_pix

    def _sigmaA_to_sigmapx(self, radius_A):
        return self.sigma_gain * radius_A * (self.G / (2.0 * self.scene_radius_A))

    # ------------------------------------------------------------
    # feature decoding
    # ------------------------------------------------------------
    def _decode_protein_features(self, protein_types):
        element_onehot = protein_types[:, :6]
        aa_onehot = protein_types[:, 6:26]
        is_backbone = protein_types[:, 26]

        elem_idx = element_onehot.argmax(dim=-1)

        prot_radius, prot_charge = map_protein_to_radius_charge(protein_types)

        prot_hydrophobic = self.PROT_HYDRO[elem_idx]
        prot_donor = self.PROT_DONOR[elem_idx]
        prot_acceptor = self.PROT_ACCEPTOR[elem_idx]

        aa_charge_term = (aa_onehot * self.AA_CHARGE_VEC.unsqueeze(0)).sum(dim=-1)

        acidic_ids = [2, 3]      # D, E
        basic_ids = [6, 8, 14]   # H, K, R

        acidic_mask = aa_onehot[:, acidic_ids].sum(dim=-1)
        basic_mask = aa_onehot[:, basic_ids].sum(dim=-1)

        prot_acceptor = prot_acceptor + 0.20 * acidic_mask + 0.15 * is_backbone
        prot_donor = prot_donor + 0.15 * basic_mask + 0.15 * is_backbone
        prot_charge = prot_charge + 0.35 * aa_charge_term

        return (
            prot_radius,
            prot_charge,
            prot_hydrophobic.clamp(0.0, 1.5),
            prot_donor.clamp(0.0, 1.5),
            prot_acceptor.clamp(0.0, 1.5),
        )

    def _decode_ligand_features(self, ligand_types, target=False):
        """
        target=True:  ligand_types [N] int
        target=False: ligand_types [N,13] probs/logits

        Returns:
            ligand_probs : [N,13]
            lig_radius   : [N]
            lig_charge   : [N]   # sharpened / contracted charge
            lig_hydro    : [N]
            lig_donor    : [N]
            lig_acceptor : [N]
        """
        if target:
            ligand_probs = F.one_hot(
                ligand_types, num_classes=len(ligand_radius_by_index)
            ).float()
        else:
            ligand_probs = ligand_types.float()
            # if logits accidentally passed in, this keeps it safer
            if ligand_probs.min() < 0 or ligand_probs.max() > 1.0 + 1e-4:
                ligand_probs = torch.softmax(ligand_probs, dim=-1)

        # ---- radius: soft average is acceptable ----
        radius_vec = torch.tensor(
            ligand_radius_by_index,
            device=ligand_probs.device,
            dtype=torch.float32,
        )
        lig_radius = ligand_probs @ radius_vec

        # ---- charge: first soft average, then shrink / sharpen ----
        # reduce fake "half-charged cloud" effect from uncertain typing
        charge_vec = torch.tensor(
            ligand_charge_by_index,
            device=ligand_probs.device,
            dtype=torch.float32,
        )
        raw_charge = ligand_probs @ charge_vec
        lig_charge = torch.tanh(2.2 * raw_charge)  # contracted effective charge

        # ---- masks ----
        hydro_mask = torch.zeros(13, device=ligand_probs.device, dtype=torch.float32)
        hydro_mask[[1, 2, 7, 8, 9, 10, 11, 12]] = 1.0

        donor_mask = torch.zeros(13, device=ligand_probs.device, dtype=torch.float32)
        donor_mask[[3, 4, 8, 10, 11]] = 1.0

        acceptor_mask = torch.zeros(13, device=ligand_probs.device, dtype=torch.float32)
        acceptor_mask[[3, 4, 5, 6, 7, 10, 11, 12]] = 1.0

        # ---- conservative soft feature mapping ----
        # compared with direct sum, tanh-compression is more robust
        raw_hydro = ligand_probs @ hydro_mask
        raw_donor = ligand_probs @ donor_mask
        raw_acceptor = ligand_probs @ acceptor_mask

        lig_hydrophobic = torch.tanh(1.4 * raw_hydro)
        lig_donor = torch.tanh(1.8 * raw_donor)
        lig_acceptor = torch.tanh(1.8 * raw_acceptor)

        return (
            ligand_probs,
            lig_radius,
            lig_charge,
            lig_hydrophobic.clamp(0.0, 1.0),
            lig_donor.clamp(0.0, 1.0),
            lig_acceptor.clamp(0.0, 1.0),
        )

    # ------------------------------------------------------------
    # interaction on a single complex
    # ------------------------------------------------------------
    def _build_single_complex_interaction_features(
            self,
            p_pos, p_radius, p_charge, p_hydro, p_donor, p_acceptor,
            l_pos, l_radius, l_charge, l_hydro, l_donor, l_acceptor,
    ):
        """
        Inputs:
            protein: [Np,*]
            ligand:  [Nl,*]
        Returns:
            feat: [Nl,C]

        Channel order:
            0 hbond_donor
            1 hbond_acceptor
            2 hydrophobic
            3 positive_ionic
            4 negative_ionic
            5 same_sign
            6 clash
        """
        device = l_pos.device
        dtype = l_pos.dtype

        if l_pos.size(0) == 0:
            return torch.zeros((0, self.num_channels), device=device, dtype=dtype)
        if p_pos.size(0) == 0:
            return torch.zeros((l_pos.size(0), self.num_channels), device=device, dtype=dtype)

        # [Nl, Np]
        dist = torch.cdist(l_pos, p_pos).clamp_min(1e-6)
        cutoff_mask = (dist < self.cutoff_A).to(dtype)

        # ---- vdw geometry ----
        vdw_touch = l_radius.unsqueeze(-1) + p_radius.unsqueeze(0)  # [Nl,Np]
        overlap = vdw_touch - dist

        # ---- hard clash ----
        # stronger than linear relu, but still stable for training
        clash_pair = F.relu(overlap + 0.12).square() * cutoff_mask

        # ---- safe gate ----
        # favorable interactions only count outside overlap region
        safe_gate = torch.sigmoid(10.0 * (dist - (vdw_touch + 0.08))) * cutoff_mask

        # ---- sharper distance windows ----
        # narrower than before, closer to docking-like local preference
        hbond_w = torch.exp(-0.5 * ((dist - 2.9) / 0.30) ** 2) * safe_gate
        hydro_w = torch.exp(-0.5 * ((dist - 3.8) / 0.75) ** 2) * safe_gate
        ionic_w = torch.exp(-0.5 * ((dist - 3.4) / 0.60) ** 2) * safe_gate

        # same-sign penalty should still see near-range charge conflicts
        same_sign_w = torch.exp(-0.5 * ((dist - 3.5) / 0.75) ** 2) * cutoff_mask

        # ---- charges ----
        l_posq = F.relu(l_charge)
        l_negq = F.relu(-l_charge)
        p_posq = F.relu(p_charge)
        p_negq = F.relu(-p_charge)

        # ---- hbond ----
        # donor/acceptor pairing remains directional at feature level
        hbond_donor_pair = (
                l_donor.unsqueeze(-1) * p_acceptor.unsqueeze(0) * hbond_w
        )  # [Nl,Np]

        hbond_acceptor_pair = (
                l_acceptor.unsqueeze(-1) * p_donor.unsqueeze(0) * hbond_w
        )  # [Nl,Np]

        # atom-level saturation before splat: prevents one atom from over-collecting
        feat_hbond_donor = 1.0 - torch.exp(-1.6 * hbond_donor_pair.sum(dim=-1))
        feat_hbond_acceptor = 1.0 - torch.exp(-1.6 * hbond_acceptor_pair.sum(dim=-1))

        # ---- hydrophobic ----
        hydro_pair = (
                l_hydro.unsqueeze(-1) * p_hydro.unsqueeze(0) * hydro_w
        )

        # saturate before splat, very important to stop atom-packing abuse
        feat_hydrophobic = 1.0 - torch.exp(-1.1 * hydro_pair.sum(dim=-1))

        # ---- ionic attraction ----
        pos_ionic_pair = (
                l_posq.unsqueeze(-1) * p_negq.unsqueeze(0) * ionic_w
        )
        neg_ionic_pair = (
                l_negq.unsqueeze(-1) * p_posq.unsqueeze(0) * ionic_w
        )

        feat_positive_ionic = 1.0 - torch.exp(-1.4 * pos_ionic_pair.sum(dim=-1))
        feat_negative_ionic = 1.0 - torch.exp(-1.4 * neg_ionic_pair.sum(dim=-1))

        # ---- same-sign electrostatic penalty ----
        same_sign_pair = (
                (
                        l_posq.unsqueeze(-1) * p_posq.unsqueeze(0)
                        + l_negq.unsqueeze(-1) * p_negq.unsqueeze(0)
                ) * same_sign_w
        )
        feat_same_sign = same_sign_pair.sum(dim=-1)

        # ---- clash penalty ----
        feat_clash = clash_pair.sum(dim=-1)

        feat = torch.stack([
            feat_hbond_donor,
            feat_hbond_acceptor,
            feat_hydrophobic,
            feat_positive_ionic,
            feat_negative_ionic,
            feat_same_sign,
            feat_clash,
        ], dim=-1)  # [Nl, 7]

        # ---- mild channel balancing ----
        if self.normalize_1d and feat.numel() > 0:
            scale = feat.detach().mean(dim=0, keepdim=True).clamp_min(1e-4)
            feat = feat / scale

        # ---- clip hotspots to avoid unstable splat domination ----
        feat = feat.clamp(min=0.0, max=6.0)

        return feat

    # ------------------------------------------------------------
    # sparse local splat for one view
    # ------------------------------------------------------------
    def _splat_all_views_vectorized(
            self,
            x_pix, y_pix, sigma_pix, feat, depth=None,
    ):
        """
        x_pix, y_pix: [V, N]
        sigma_pix:    [N]
        feat:         [N, C]
        depth:        [V, N] or None

        return:
            maps: [V, C, G, G]
        """
        device = feat.device
        dtype = feat.dtype
        V, N = x_pix.shape
        C = feat.shape[-1]
        G = self.G

        if N == 0:
            return torch.zeros((V, C, G, G), device=device, dtype=dtype)

        # ----- primitive amplitudes -----
        feat_eff = feat.unsqueeze(0).expand(V, -1, -1)  # [V,N,C]

        if self.use_alpha_blend_approx:
            favorable = (
                    feat[:, 0] + feat[:, 1]  # hbond donor/acceptor
                    + 0.7 * feat[:, 2]  # hydrophobic
                    + 0.8 * feat[:, 3] + 0.8 * feat[:, 4]  # ionic
            )
            unfavorable = 0.8 * feat[:, 5] + 1.2 * feat[:, 6]  # same_sign + clash
            alpha = torch.sigmoid(self.alpha_scale * (favorable - unfavorable))
            feat_eff = feat_eff * alpha.view(1, N, 1)

        if self.use_depth_attenuation and depth is not None:
            depth_attn = torch.sigmoid(-0.25 * depth).unsqueeze(-1)  # [V,N,1]
            feat_eff = feat_eff * depth_attn

        # ----- flatten all primitives over views -----
        P = V * N
        x = x_pix.reshape(P)  # [P]
        y = y_pix.reshape(P)  # [P]
        z = sigma_pix.unsqueeze(0).expand(V, -1).reshape(P).clamp_min(1e-3)  # [P]
        feat_eff = feat_eff.reshape(P, C)  # [P,C]

        # view id per primitive
        view_ids = torch.arange(V, device=device).unsqueeze(1).expand(V, N).reshape(P)  # [P]

        # ----- local patch offsets -----
        r = torch.ceil(self.gaussian_truncate * z).to(torch.long)  # [P]
        r_max = int(r.max().item())
        if r_max == 0:
            r_max = 1

        offs = torch.arange(-r_max, r_max + 1, device=device)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")  # [K,K]
        ox = ox.reshape(-1)  # [M]
        oy = oy.reshape(-1)  # [M]
        M = ox.numel()

        # patch coords for every primitive
        x0 = torch.floor(x).to(torch.long)  # [P]
        y0 = torch.floor(y).to(torch.long)

        xs = x0[:, None] + ox[None, :]  # [P,M]
        ys = y0[:, None] + oy[None, :]  # [P,M]

        # valid local radius mask
        local_valid = (ox[None, :].abs() <= r[:, None]) & (oy[None, :].abs() <= r[:, None])

        # image boundary mask
        in_bound = (xs >= 0) & (xs < G) & (ys >= 0) & (ys < G)

        valid = local_valid & in_bound  # [P,M]

        # ----- gaussian weights -----
        xs_f = xs.to(dtype)
        ys_f = ys.to(dtype)

        gx = torch.exp(-0.5 * ((xs_f - x[:, None]) / z[:, None]) ** 2)  # [P,M]
        gy = torch.exp(-0.5 * ((ys_f - y[:, None]) / z[:, None]) ** 2)  # [P,M]
        w = gx * gy
        w = w * valid.to(dtype)

        # normalize patch weights per primitive
        w = w / w.sum(dim=1, keepdim=True).clamp_min(self.eps)  # [P,M]

        # ----- flatten target index: view * G*G + y*G + x -----
        flat_idx = view_ids[:, None] * (G * G) + ys * G + xs  # [P,M]
        flat_idx = flat_idx.clamp(0, V * G * G - 1)

        # ----- scatter add -----
        out = torch.zeros((C, V * G * G), device=device, dtype=dtype)

        # [P,C,M]
        vals = feat_eff[:, :, None] * w[:, None, :]

        # scatter_add_ expects matching shape
        idx = flat_idx[:, None, :].expand(P, C, M).permute(1, 0, 2).reshape(C, -1)
        src = vals.permute(1, 0, 2).reshape(C, -1)

        out.scatter_add_(dim=1, index=idx, src=src)

        out = out.view(C, V, G, G).permute(1, 0, 2, 3).contiguous()  # [V,C,G,G]
        return out

    # ------------------------------------------------------------
    # render one complex
    # ------------------------------------------------------------
    def _render_single_complex(
        self,
        protein_pos, protein_types,
        ligand_pos, ligand_types,
        target=False,
    ):
        # decode per-atom chemistry
        p_radius, p_charge, p_hydro, p_donor, p_acceptor = self._decode_protein_features(protein_types)
        _, l_radius, l_charge, l_hydro, l_donor, l_acceptor = self._decode_ligand_features(ligand_types, target=target)

        # ligand-centered interaction feature
        feat = self._build_single_complex_interaction_features(
            protein_pos, p_radius, p_charge, p_hydro, p_donor, p_acceptor,
            ligand_pos, l_radius, l_charge, l_hydro, l_donor, l_acceptor,
        )  # [Nl,C]

        # rotate ligand only
        R = self._rotation_mats(ligand_pos.device)             # [V,3,3]
        l_coords = torch.einsum("nc,vdc->vnd", ligand_pos, R)  # [V,Nl,3]

        lx, ly = self._world_to_pixel(l_coords)  # [V,Nl], [V,Nl]
        lz = l_coords[..., 2]                    # [V,Nl]

        sigma_pix = self._sigmaA_to_sigmapx(l_radius)  # [Nl]

        maps = self._splat_all_views_vectorized(
            lx, ly, sigma_pix, feat, depth=lz)

        scale = maps.detach().abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        maps = torch.tanh(maps / (2.0 * scale))

        return maps

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(
        self,
        protein_pos, protein_types, batch_protein,
        ligand_pos, ligand_types, batch_ligand,
        target: bool = False,
    ):
        device = ligand_pos.device
        B = int(batch_ligand.max().item()) + 1

        all_maps = []
        for b in range(B):
            p_mask = (batch_protein == b)
            l_mask = (batch_ligand == b)

            maps_b = self._render_single_complex(
                protein_pos[p_mask],
                protein_types[p_mask],
                ligand_pos[l_mask],
                ligand_types[l_mask],
                target=target,
            )  # [V,C,G,G]

            all_maps.append(maps_b)

        energy_maps = torch.stack(all_maps, dim=0).to(device)  # [B,V,C,G,G]

        return {
            "energy_maps": energy_maps,
            "channel_names": self.CHANNELS,
        }



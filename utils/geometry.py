import torch
import torch.nn.functional as F

from torch_cluster import radius
from torch_scatter import scatter_min

from torch_geometric.nn import knn_graph

# Minimal vdW table; extend if you have more elements
_VDW = {1:1.20, 6:1.70, 7:1.55, 8:1.52, 9:1.47, 15:1.80, 16:1.80, 17:1.75, 35:1.85, 53:1.98}

def generate_sdf_field(atom_positions, grid_size=32):
    # 简单示例: 根据原子点生成SDF体素网格
    grid = torch.linspace(-16, 16, grid_size)
    X, Y, Z = torch.meshgrid(grid, grid, grid, indexing='ij')
    coords = torch.stack([X, Y, Z], dim=-1).to(atom_positions.device)
    # [G,G,G,1,3] - [1,1,1,N,3] -> [G,G,G,N,3]
    diff = coords.unsqueeze(-2) - atom_positions.view(1, 1, 1, -1, 3)
    dist = diff.norm(dim=-1)  # [G,G,G,N]

    # SDF = 对所有原子取最小距离
    sdf, _ = dist.min(dim=-1)  # [G,G,G]
    return sdf

def sample_sdf_at_points(points, sdf_field, box_radius=16.0):
    device = sdf_field.device
    points = points.to(device)

    G = sdf_field.shape[0]
    # grid_sample 的坐标需要在 [-1, 1]
    # 这里假设 sdf_field 覆盖 [-box_radius, box_radius]
    grid = (points / box_radius).view(1, 1, 1, -1, 3)  # [1,1,1,N,3]

    # [1,1,G,G,G]
    vol = sdf_field.view(1, 1, G, G, G)

    sdf = F.grid_sample(
        vol, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # [1,1,1,1,N]

    return sdf.view(-1)  # [N]

def render_density_field(atom_positions, sigma=1.0, grid_size=64, box_radius=16.0):
    device = atom_positions.device
    N = atom_positions.size(0)

    grid = torch.linspace(-box_radius, box_radius, grid_size, device=device)
    X, Y, Z = torch.meshgrid(grid, grid, grid, indexing='ij')  # [G,G,G]
    coords = torch.stack([X, Y, Z], dim=-1)  # [G,G,G,3]

    # [G,G,G,1,3] - [1,1,1,N,3] → [G,G,G,N,3]
    diff = coords[..., None, :] - atom_positions.view(1, 1, 1, N, 3)
    dist2 = (diff ** 2).sum(-1)  # [G,G,G,N]

    density = torch.exp(-dist2 / (2 * sigma ** 2)).sum(-1)  # [G,G,G]
    density = density / max(N, 1)
    return density / len(atom_positions)

def make_grid(center, half_extent, voxel_size, device, dtype=torch.float32):
    # 连续坐标网格（单位：Å）
    grid_size = int(torch.round(2*half_extent/voxel_size).item())
    # 确保为偶数或适配你的内核/下采样需求
    grid_min = center - half_extent
    # 生成 1D 轴坐标
    axes = [torch.linspace(grid_min[i],
                           grid_min[i] + voxel_size*(grid_size-1),
                           grid_size, device=device, dtype=dtype) for i in range(3)]
    X, Y, Z = torch.meshgrid(*axes, indexing='ij')
    coords = torch.stack([X, Y, Z], dim=-1)  # [G,G,G,3]
    return coords, grid_min, grid_size

# Schrödinger-like excluded volume
def vdw_radii(Z: torch.Tensor, default=1.70) -> torch.Tensor:
    r = torch.full((Z.numel(),), float(default), device=Z.device, dtype=torch.float32)
    for k, v in _VDW.items():
        r = torch.where(Z == k, torch.tensor(v, device=Z.device), r)
    return r

def excluded_volume_loss_vectorized(
    ligand_pos: torch.Tensor, ligand_batch: torch.Tensor,
    protein_pos: torch.Tensor, protein_batch: torch.Tensor,
    protein_Z: torch.Tensor = None,
    r_max: float = 10.0,
    probe: float = 1.4,
    margin: float = 0.0,
    reduction: str = "mean",
    clamp_max: float = 10.0,
):
    """
    Vectorized steric clash penalty:
    - Build ligand->protein neighbor pairs within `cutoff` per complex (bipartite radius)
    - z = dist - (r_vdw + probe + margin)
    - per ligand atom: z_min = min_i z
    - penalty = relu(-z_min)
    """

    device = ligand_pos.device
    n_lig = ligand_pos.size(0)

    # radii for protein atoms
    if protein_Z is None:
        pr = torch.full((protein_pos.size(0),), 1.7, device=device, dtype=torch.float32)
    else:
        pr = vdw_radii(protein_Z).to(device)

    # bipartite radius neighbors: pairs (j, i) s.t. ||x_j - y_i|| < cutoff
    # Here: x = ligand, y = protein
    # returns edge_index (2, E): [lig_idx; prot_idx]
    edge_index = radius(
        x=ligand_pos, y=protein_pos,
        r=r_max,
        batch_x=ligand_batch, batch_y=protein_batch,
        max_num_neighbors=256  # cap for safety; adjust if needed
    )
    lig_idx, prot_idx = edge_index[1], edge_index[0]

    if lig_idx.numel() == 0:
        # no neighbors found => no steric info; return zero to avoid NaN
        return ligand_pos.new_tensor(0.0)

    # compute z = dist - (radius+probe+margin)
    d = torch.norm(ligand_pos[lig_idx] - protein_pos[prot_idx], dim=-1)  # (E,)
    rad = pr[prot_idx] + probe + margin                                  # (E,)
    z = d - rad                                                          # (E,)

    # per ligand atom: min z over its neighbors
    z_min, _ = scatter_min(z, lig_idx, dim=0, dim_size=n_lig)            # (n_lig,)
    # If some ligand atoms had no neighbors (outside cutoff), z_min will be +inf
    z_min = torch.where(torch.isfinite(z_min), z_min, torch.zeros_like(z_min))

    # penetration depth
    pen = (-z_min).clamp(min=0.0, max=clamp_max)

    if reduction == "mean":
        return pen.mean()
    elif reduction == "sum":
        return pen.sum()
    else:
        return pen

# Ligand self-collapse penalty
def ligand_self_repulsion_loss(
    ligand_pos: torch.Tensor, ligand_batch: torch.Tensor,
    k: int = 16,
    d_min: float = 1.2,
    power: float = 2.0,
    clamp_max: float = 10.0,
):
    # kNN within each complex
    edge_index = knn_graph(ligand_pos, k=min(k, max(int(ligand_pos.size(0)-1), 1)),
                           batch=ligand_batch, flow="source_to_target")
    src, dst = edge_index
    d = torch.norm(ligand_pos[src] - ligand_pos[dst], dim=-1)
    pen = (d_min - d).clamp(min=0.0, max=clamp_max)
    return (pen ** power).mean()

def ligand_protein_shell_loss(
    ligand_pos,
    ligand_batch,
    protein_pos,
    protein_batch,
    d0=3.6,
    sigma=0.8,
):
    """
    ligand_pos   : [Nl,3]
    ligand_batch : [Nl]
    protein_pos  : [Np,3]
    protein_batch: [Np]

    Protein-centered coordinates are already assumed if center_pos(mode="protein") was used.
    """

    dist = torch.cdist(ligand_pos, protein_pos)   # [Nl, Np]

    same_batch = ligand_batch[:, None] == protein_batch[None, :]
    dist = dist.masked_fill(~same_batch, 1e6)

    min_dist = dist.min(dim=1).values   # [Nl]

    shell_reward = torch.exp(-0.5 * ((min_dist - d0) / sigma) ** 2)
    loss = 1.0 - shell_reward.mean()

    return loss
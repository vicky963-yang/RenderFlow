import torch

def get_hydrophobic_mask(atoms):
    # 简单模拟疏水原子掩码
    # 可结合 RDKit 根据元素类型标注
    return torch.ones(atoms.shape[0], device=atoms.device)

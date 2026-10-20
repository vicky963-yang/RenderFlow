import copy

import pytorch_lightning as pl
import torch
import math
from torch_scatter import scatter_sum, scatter_mean
import numpy as np

import utils.transforms as trans
from models.bfn4sbdd import BFN4SBDDScoreModel
from models.energy_pro import SparsePharmacophoreEnergyRenderer
from models.render_loss import GeometryEnergyGuidedLoss, DifferentiableRenderLoss
from models.renderplug import TetSphereRenderer
from evaluation.utils import atom_num
from models.score_model import log_sample_categorical, center_pos



class RenderDockBFNModule(pl.LightningModule):
    """
    Wrap ScorePosNet3D as diffusion generator backbone and add render-guided supervision.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = BFN4SBDDScoreModel(
            self.cfg.model,
            protein_atom_feature_dim=self.cfg.model.protein_atom_feature_dim,
            ligand_atom_feature_dim=self.cfg.model.ligand_atom_feature_dim,
            condition_time=True,
            sigma1_coord=self.cfg.model.sigma1_coord,
            beta1=self.cfg.model.beta1,
            use_discrete_t=self.cfg.model.use_discrete_t,
            discrete_steps=self.cfg.model.discrete_steps,
            t_min=self.cfg.model.t_min,
            node_indicator=self.cfg.model.node_indicator,
            time_emb_mode=self.cfg.model.time_emb_mode,
            time_emb_dim=self.cfg.model.time_emb_dim,
            center_pos_mode=self.cfg.model.center_pos_mode,
            pos_init_mode=self.cfg.model.pos_init_mode,
            destination_prediction=False,
            sampling_strategy=self.cfg.model.sampling_strategy,
        )

        self.renderer = TetSphereRenderer(
            self.cfg,
            n_views=self.cfg.model.geom_energy_cfg.renderer.n_views,
            image_size=self.cfg.model.geom_energy_cfg.renderer.image_size,
            scene_radius_A=self.cfg.model.geom_energy_cfg.renderer.scene_radius_A,  # your 10 Å radius FOV
            sigma_gain=self.cfg.model.geom_energy_cfg.renderer.sigma_gain,  # footprint gain
            eps=self.cfg.model.geom_energy_cfg.renderer.eps,
            normalize_1d=self.cfg.model.geom_energy_cfg.renderer.normalize_1d,
        )
        self.energy_projector = SparsePharmacophoreEnergyRenderer(
            self.cfg, **self.cfg.model.geom_energy_cfg.renderer
            )
        self.render_loss_fn = DifferentiableRenderLoss(
            **self.cfg.model.geom_energy_cfg.render_loss_fn
        )

        self.geom_energy_loss = GeometryEnergyGuidedLoss(
            renderer=self.renderer,
            energy_projector=self.energy_projector,
            render_loss_fn=self.render_loss_fn,
            cfg=cfg.model,
            pos_norm=self.cfg.data.normalizer_dict.pos

        )
        self.save_hyperparameters(self.cfg)


    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.optimizer.lr,
            weight_decay=self.cfg.training.optimizer.weight_decay,
            betas=(self.cfg.training.optimizer.beta1, self.cfg.training.optimizer.beta2,))
        if self.cfg.training.scheduler.type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=self.cfg.training.scheduler.max_iters,
                eta_min=self.cfg.training.scheduler.min_lr,
            )
            sched = {
                'scheduler': scheduler,
                'monitor': self.cfg.training.scheduler.monitor,
                'interval': self.cfg.training.scheduler.interval,
                'frequency': self.cfg.training.scheduler.frequency,
            }
            return {"optimizer": opt, "lr_scheduler": sched}
        elif self.cfg.training.scheduler.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=self.cfg.training.scheduler.factor,
                patience=self.cfg.training.scheduler.patience,
                min_lr=self.cfg.training.scheduler.min_lr
            )
            sched = {
                'scheduler': scheduler,
                'monitor': self.cfg.training.scheduler.monitor,
                'interval': self.cfg.training.scheduler.interval,
                'frequency': self.cfg.training.scheduler.frequency,
            }
            return {"optimizer": opt, "lr_scheduler": sched}

    def on_after_backward(self):
        if self.global_rank == 0:
            for name, p in self.named_parameters():
                if p.requires_grad and p.grad is None:
                    print(f"[UNUSED PARAM] {name}")

    def training_step(self, batch, batch_idx):
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )  # get the data from the batch


        with torch.no_grad():
            # add noise to protein_pos
            protein_noise = torch.randn_like(protein_pos) * self.cfg.training.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            # random rotation as data aug
            if self.cfg.training.random_rot:
                M = np.random.randn(3, 3)
                Q, __ = np.linalg.qr(M)
                Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
                gt_protein_pos = gt_protein_pos @ Q
                ligand_pos = ligand_pos @ Q

        num_graphs = batch_protein.max().item() + 1
        # !!!!!
        gt_protein_pos, ligand_pos, offset = center_pos(
            gt_protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.cfg.model.center_pos_mode,
        )
        t = torch.rand(
            [num_graphs, 1], dtype=ligand_pos.dtype, device=ligand_pos.device
        ).index_select(
            0, batch_ligand
        )  # different t for different molecules.

        if not self.cfg.model.use_discrete_t and not self.cfg.model.destination_prediction:
            t = torch.clamp(t, min=self.backbone.t_min)  # clamp t to [t_min,1]

        c_loss, d_loss, discretised_loss, coord_pred, p0_h = self.backbone.loss_one_step(
            t,
            protein_pos=gt_protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=ligand_pos,
            ligand_v=ligand_v,
            batch_ligand=batch_ligand,
        )

        # 2) 几何与能量约束
        diff_out = {
            "ligand_v_recon": p0_h,
            "pred_ligand_pos": coord_pred,
            "offset": offset,
        }
        render_losses = self.geom_energy_loss(batch, diff_out)

        loss = torch.mean(c_loss + self.cfg.training.v_loss_weight * d_loss + discretised_loss)
        total = loss + self.cfg.lambda_render * render_losses['total']

        # logs
        self.log_dict({
            "train/loss": total,
            "train/loss_c_ratio": c_loss.mean().item() / loss.item(),
            "train/loss_pos": c_loss.mean().item(),
            "train/loss_v": d_loss.mean().item(),
            "train/render_total": render_losses['total'],
            "train/render_clash": render_losses['energy_clash'],
            "train/render_same_sign": render_losses['energy_same_sign'],
            "train/render_hbond": render_losses['energy_hbond'],
            "train/render_hydro": render_losses['energy_hydro'],
            "train/render_ionic": render_losses['energy_ionic'],
            "train/render_view_var": render_losses["energy_view_var"],
            "train/render_steric": render_losses['loss_steric'],
            "train/render_self_repulsion": render_losses['loss_self_repulsion'],
            "train/render_depth": render_losses['loss_depth'],
            "train/render_density": render_losses['loss_density'],
            "train/render_energy": render_losses['loss_energy'],

        }, prog_bar=True, on_step=True, on_epoch=True, batch_size=int(batch["protein_element_batch"].max().item()+1),sync_dist=True)

        return total

    def validation_step(self, batch, batch_idx):
        out_data_list = self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref', desc=f'Val')
        return out_data_list

    def test_step(self, batch, batch_idx):
        out_data_list = []
        n_samples = self.cfg.evaluation.num_samples
        for _ in range(n_samples):
            batch_output = self.shared_sampling_step(batch, batch_idx,
                                                     sample_num_atoms=self.cfg.evaluation.sample_num_atoms,
                                                     desc=f'Test-{_}/{n_samples}')
            # for idx, item in enumerate(batch_output):
            out_data_list.append(batch_output)

        out_data_list_reorder = []
        for i in range(len(out_data_list[0])):
            for j in range(len(out_data_list)):
                out_data_list_reorder.append(out_data_list[j][i])
        return out_data_list_reorder

    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms, desc=''):
        # here we need to sample the molecules in the validation step
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )
        num_graphs = batch_protein.max().item() + 1  # B
        n_nodes = batch_ligand.size(0)  # N_lig
        assert num_graphs == len(batch), f"num_graphs: {num_graphs} != len(batch): {len(batch)}"

        # move protein center to origin & ligand correspondingly
        protein_pos, ligand_pos, offset = center_pos(
            protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.cfg.model.center_pos_mode,
        )  # TODO: ugly
        ligand_mask, ligand_v = None, None

        # determine the number of atoms in the ligand
        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(
                    data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size, self.cfg.data.path).astype(int))
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(
                protein_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=protein_pos.device)
        elif sample_num_atoms == 'prior_ref':
            assert hasattr(batch, 'ligand_pos') and hasattr(batch, 'ligand_atom_feature_full')
            ligand_num_atoms = []
            while len(ligand_num_atoms) < len(batch):
                data_id = len(ligand_num_atoms)
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(
                    data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                atom_nums = atom_num.sample_atom_num(pocket_size, self.cfg.data.path).astype(int)
                if atom_nums < len(data.ligand_pos):
                    continue
                ligand_num_atoms.append(atom_nums)
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(
                protein_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=protein_pos.device)
        elif sample_num_atoms == 'ref' or sample_num_atoms == 'inpainting_ref':
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(protein_pos.device)
            if sample_num_atoms == 'inpainting_ref':
                assert hasattr(batch, 'ligand_pos') and hasattr(batch, 'ligand_atom_feature_full') and hasattr(batch,
                                                                                                               'ligand_mask')
                ligand_mask = batch.ligand_mask
                ligand_pos = batch.ligand_pos
                ligand_v = batch.ligand_atom_feature_full
        elif sample_num_atoms == 'inpainting_prior':
            assert hasattr(batch, 'ligand_pos') and hasattr(batch, 'ligand_atom_feature_full') and hasattr(batch,
                                                                                                           'ligand_mask')
            ligand_num_atoms = []
            ligand_mask, ligand_pos, ligand_v = [], [], []
            while len(ligand_num_atoms) < len(batch):
                data_id = len(ligand_num_atoms)
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(
                    data.protein_pos.detach().cpu().numpy() * self.cfg.data.normalizer_dict.pos)
                atom_nums = atom_num.sample_atom_num(pocket_size, self.cfg.data.path).astype(int)
                if atom_nums < len(data.ligand_pos):
                    continue
                ligand_num_atoms.append(atom_nums)

                # update ligand_pos and ligand_v, ligand_mask
                data.ligand_pos = torch.cat(
                    [data.ligand_pos, torch.zeros(atom_nums - len(data.ligand_pos), 3, device=data.ligand_pos.device)])
                data.ligand_atom_feature_full = torch.cat([data.ligand_atom_feature_full,
                                                           torch.zeros(atom_nums - len(data.ligand_atom_feature_full),
                                                                       device=data.ligand_atom_feature_full.device)])
                data.ligand_mask = torch.cat([data.ligand_mask,
                                              torch.zeros(atom_nums - len(data.ligand_mask), dtype=torch.bool,
                                                          device=data.ligand_mask.device)])
                batch[data_id] = data
                # print(f"sampled {atom_nums} atoms for ligand {data_id} in inpainting_prior mode")
                assert len(data.ligand_pos) == atom_nums and len(data.ligand_atom_feature_full) == atom_nums and len(
                    data.ligand_mask) == atom_nums
                ligand_mask.append(data.ligand_mask)
                ligand_pos.append(data.ligand_pos)
                ligand_v.append(data.ligand_atom_feature_full)

            # print(f"ligand_num_atoms: {sum(ligand_num_atoms)}")
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(
                protein_pos.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=protein_pos.device)
            ligand_mask = torch.cat(ligand_mask, dim=0).bool()
            ligand_pos = torch.cat(ligand_pos, dim=0)
            ligand_v = torch.cat(ligand_v, dim=0).long()
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ligand_pos.device),
            ligand_num_atoms.cumsum(dim=0)
        ])

        # TODO reuse for visualization and test
        # forward pass to get the ligand sample
        theta_chain, sample_chain, y_chain = self.backbone.sample(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            sample_steps=self.cfg.evaluation.sample_steps,
            n_nodes=num_graphs,
            ligand_pos=ligand_pos,  # for inpainting
            ligand_v=ligand_v,  # for inpainting
            ligand_mask=ligand_mask,  # for inpainting
            EPS=self.cfg.evaluation.interpolate_coef,
            desc=desc,
        )

        # restore ligand to original position
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot = final[0] + offset[batch_ligand], final[1]

        # along with normalizer
        pred_pos = pred_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=protein_pos.device
        )
        out_batch = copy.deepcopy(batch)
        out_batch.protein_pos = out_batch.protein_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=protein_pos.device
        )

        pred_v = one_hot.argmax(dim=-1)
        # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
        pred_atom_type = trans.get_atomic_number_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        )  # List[int]

        # for visualization
        atom_type = [trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]  # List[int]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=protein_pos.device)  # [N_lig]

        # for reconstruction
        pred_aromatic = trans.is_aromatic_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        )  # List[bool]

        # print('[DEBUG]', num_graphs, len(ligand_cum_atoms))

        # add necessary dict to new pyg batch
        out_batch.x, out_batch.pos = atom_type, pred_pos
        out_batch.atom_type = torch.tensor(pred_atom_type, dtype=torch.long, device=protein_pos.device)
        out_batch.is_aromatic = torch.tensor(pred_aromatic, dtype=torch.long, device=protein_pos.device)
        # out_batch.mol = results

        _slice_dict = {
            "x": ligand_cum_atoms,
            "pos": ligand_cum_atoms,
            "atom_type": ligand_cum_atoms,
            "is_aromatic": ligand_cum_atoms,
            # "mol": out_batch._slice_dict["ligand_filename"],
        }
        _inc_dict = {
            "x": out_batch._inc_dict["ligand_element"],  # [0] * B, TODO: figure out what this is
            "pos": out_batch._inc_dict["ligand_pos"],
            "atom_type": out_batch._inc_dict["ligand_element"],
            "is_aromatic": out_batch._inc_dict["ligand_element"],
            # "mol": out_batch._inc_dict["ligand_filename"],
        }
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
        out_data_list = out_batch.to_data_list()
        return out_data_list

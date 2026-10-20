# import argparse
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
import concurrent.futures
import glob
# from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

# import datetime, pytz
from utils.config import load_config
from models.sbdd_train_bfn_loop import RenderDockBFNModule
from callbacks.basic import NormalizerCallback
from callbacks.bfn_validation_callback import (
    DockingTestCallback, DummyTestCallback
)

import utils.transforms as trans
from data.datasets.utils import PDBProtein, parse_sdf_file
from data.datasets.pl_data import ProteinLigandData, torchify_dict
from data.datasets.pl_data import FOLLOW_BATCH

import pytorch_lightning as pl

from pytorch_lightning import seed_everything

# from absl import logging
# import glob

from evaluation.utils import scoring_func
from evaluation.docking_vina import VinaDockingTask
from posecheck import PoseCheck
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Geometry


def get_dataloader_from_pdb(cfg):
    assert cfg.evaluation.protein_path is not None and cfg.evaluation.ligand_path is not None
    protein_fn, ligand_fn = cfg.evaluation.protein_path, cfg.evaluation.ligand_path

    # load protein and ligand
    protein = PDBProtein(protein_fn)
    ligand_dict = parse_sdf_file(ligand_fn)
    lig_pos = ligand_dict["pos"]

    print('[DEBUG] get_dataloader')
    print(lig_pos.shape, lig_pos.mean(axis=0))

    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, cfg.model.r_max)
    )
    pocket = PDBProtein(pdb_block_pocket)
    pocket_dict = pocket.to_dict_atom()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = protein_fn
    data.ligand_filename = ligand_fn

    # transform
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
    ]
    transform = Compose(transform_list)
    cfg.model.protein_atom_feature_dim = protein_featurizer.feature_dim
    cfg.model.ligand_atom_feature_dim = ligand_featurizer.feature_dim
    print(f"protein feature dim: {cfg.model.protein_atom_feature_dim}, " +
          f"ligand feature dim: {cfg.model.ligand_atom_feature_dim}")

    # dataloader
    collate_exclude_keys = ["ligand_nbh_list"]
    test_set = [transform(data)] * cfg.evaluation.num_samples
    cfg.evaluation.num_samples = 1
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )

    cfg.evaluation.docking_config.protein_root = os.path.dirname(os.path.abspath(protein_fn))
    print(f"protein root: {cfg.evaluation.docking_config.protein_root}")

    return test_loader


def call(protein_fn, ligand_fn, ckpt_path='./logs/Pocket_splat/checkpoints/20260328-035652/last.ckpt',
         num_samples=1000, sample_steps=100, sample_num_atoms='prior',
         beta1=1.5, sigma1_coord=0.03, seed=1234):

    cfg = load_config('./logs/Pocket_splat/test_outputs/20260328-141049/pocketsplatbfn.yaml')
    seed_everything(cfg.seed)

    cfg.evaluation.protein_path = protein_fn
    cfg.evaluation.ligand_path = ligand_fn
    cfg.test_ckpt_path = ckpt_path
    cfg.test_only = True
    cfg.no_wandb = True
    cfg.evaluation.num_samples = num_samples
    cfg.evaluation.sample_steps = sample_steps
    cfg.evaluation.sample_num_atoms = sample_num_atoms  # or 'prior'
    cfg.model.beta1 = beta1
    cfg.model.sigma1_coord = sigma1_coord
    cfg.seed = seed
    cfg.accounting.test_outputs_dir = './logs/Pocket_splat/test_outputs_v8'


    # print(f"The config of this process is:\n{cfg}")

    print(protein_fn, ligand_fn)
    test_loader = get_dataloader_from_pdb(cfg)
    # wandb_logger.log_hyperparams(cfg.todict())

    model = RenderDockBFNModule(cfg)

    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.training.epochs,
        check_val_every_n_epoch=cfg.training.ckpt_freq,
        devices=1,
        # logger=wandb_logger,
        num_sanity_val_steps=0,
        callbacks=[
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            DummyTestCallback(
                protein_fn=protein_fn,
            ),
        ],
    )

    trainer.test(model, dataloaders=test_loader, ckpt_path=cfg.test_ckpt_path)

def align_to_ref(mol, ref_mol):
    """将生成的 mol 平移到参考配体的中心"""
    ref_center = AllChem.ComputeCentroid(ref_mol.GetConformer())
    mol_center = AllChem.ComputeCentroid(mol.GetConformer())
    dx = ref_center.x - mol_center.x
    dy = ref_center.y - mol_center.y
    dz = ref_center.z - mol_center.z

    # 4. 手动更新坐标
    for i in range(mol.GetNumAtoms()):
        pos = mol.GetConformer().GetAtomPosition(i)
        new_pos = Geometry.Point3D(pos.x + dx, pos.y + dy, pos.z + dz)
        mol.GetConformer().SetAtomPosition(i, new_pos)
    return mol

class Metrics:
    def __init__(self, protein_fn, ref_ligand_fn, ligand_fn):
        self.protein_fn = protein_fn
        self.ref_ligand_fn = ref_ligand_fn
        self.ligand_fn = ligand_fn
        self.exhaustiveness = 16

    def vina_dock(self, mol):
        chem_results = {}

        try:
            # qed, logp, sa, lipinski, ring size, etc
            chem_results.update(scoring_func.get_chem(mol))
            chem_results['atom_num'] = mol.GetNumAtoms()

            # docking for actives
            ref_ligand = next(iter(Chem.SDMolSupplier(self.ref_ligand_fn)))
            mol = align_to_ref(mol, ref_ligand) # 关键一步：平移坐标
            # docking
            # vina_task = VinaDockingTask.from_generated_mol(
                # mol, ligand_filename=self.ref_ligand_fn, protein_root=os.path.dirname(self.protein_fn))
            vina_task = VinaDockingTask.from_generated_mol(
                mol, ligand_filename=self.ref_ligand_fn, protein_root=os.path.dirname(self.protein_fn), center=AllChem.ComputeCentroid(ref_ligand.GetConformer()))
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=self.exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=self.exhaustiveness)
            docking_results = vina_task.run(mode='dock', exhaustiveness=self.exhaustiveness)

            chem_results['vina_score'] = score_only_results[0]['affinity']
            chem_results['vina_minimize'] = minimize_results[0]['affinity']
            chem_results['vina_dock'] = docking_results[0]['affinity']
            # chem_results['vina_dock_pose'] = docking_results[0]['pose']
            return chem_results
        except Exception as e:
            print(e)

        return chem_results

    def pose_check(self, mol):
        pc = PoseCheck()

        pose_check_results = {}

        protein_ready = False
        try:
            pc.load_protein_from_pdb(self.protein_fn)
            protein_ready = True
        except ValueError as e:
            return pose_check_results

        ligand_ready = False
        try:
            pc.load_ligands_from_mols([mol])
            ligand_ready = True
        except ValueError as e:
            return pose_check_results

        if ligand_ready:
            try:
                strain = pc.calculate_strain_energy()[0]
                pose_check_results['strain'] = strain
            except Exception as e:
                pass

        if protein_ready and ligand_ready:
            try:
                clash = pc.calculate_clashes()[0]
                pose_check_results['clash'] = clash
            except Exception as e:
                pass

            try:
                df = pc.calculate_interactions()
                columns = np.array([column[2] for column in df.columns])
                flags = np.array([df[column][0] for column in df.columns])

                def count_inter(inter_type):
                    if len(columns) == 0:
                        return 0
                    count = sum((columns == inter_type) & flags)
                    return count

                # ['Hydrophobic', 'HBDonor', 'VdWContact', 'HBAcceptor']
                hb_donor = count_inter('HBDonor')
                hb_acceptor = count_inter('HBAcceptor')
                vdw = count_inter('VdWContact')
                hydrophobic = count_inter('Hydrophobic')

                pose_check_results['hb_donor'] = hb_donor
                pose_check_results['hb_acceptor'] = hb_acceptor
                pose_check_results['vdw'] = vdw
                pose_check_results['hydrophobic'] = hydrophobic
            except Exception as e:
                pass

        for k, v in pose_check_results.items():
            mol.SetProp(k, str(v))

        return pose_check_results

    def evaluate(self):
        mol = Chem.SDMolSupplier(self.ligand_fn, removeHs=False)[0]

        chem_results = self.vina_dock(mol)
        pose_check_results = self.pose_check(mol)
        chem_results.update(pose_check_results)

        return chem_results


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def process_mol(out_fn, protein_path, ligand_path):
    try:
        mol = Chem.SDMolSupplier(out_fn, removeHs=False)[0]
        metrics = Metrics(protein_path, ligand_path, out_fn).evaluate()
        return os.path.basename(out_fn), metrics, mol
    except Exception as e:
        return os.path.basename(out_fn), {'error': str(e)}, None

if __name__ == '__main__':
    protein_path = sys.argv[1]
    ligand_path = sys.argv[2]

    # call(protein_path, ligand_path)
    nm_p = os.path.basename(os.path.dirname(protein_path))
    # mols_dir = os.path.join('./logs/Pocket_splat/test_outputs_v8', nm_p)
    # mols_dir = os.path.join(mols_dir, 'mols')
    mols_dir = os.path.join('./data/case_study', nm_p)

    base = os.path.basename(protein_path).split('_')[0]
    sdf_files = sorted(glob.glob(os.path.join(mols_dir, '*.sdf')))
    # out_sdf_path = os.path.join(os.path.dirname(protein_path), f'{base}_RenderSBDD.sdf')
    out_sdf_path = os.path.join(os.path.dirname(protein_path), f'{base}_active.sdf')
    with Chem.SDWriter(out_sdf_path) as writer, concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_mol, out_fn, protein_path, ligand_path) for out_fn in sdf_files]

        for future in concurrent.futures.as_completed(futures):
            name, metrics, mol = future.result()

            if mol is not None:
                mol.SetProp("_Name", name)
                for key, value in metrics.items():
                    mol.SetProp(key, str(value))
                writer.write(mol)
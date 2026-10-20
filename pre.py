import sys, os
sys.path.insert(0, "/home/yangyuwei_cluster/RenderFlow")
os.chdir("/home/yangyuwei_cluster/RenderFlow")

import concurrent.futures

# from absl import logging
import glob

from evaluation.utils import scoring_func
from evaluation.docking_vina import VinaDockingTask
from posecheck import PoseCheck
import numpy as np
from rdkit import Chem
from eval_for_pocket_test import process_mol

from rdkit.Chem import AllChem, Descriptors
mols_dir = '/home/yangyuwei_cluster/RenderFlow/data/case_study/HPK1'
sdf_files = sorted(glob.glob(os.path.join(mols_dir, '*.sdf')))
out_sdf_path = '/home/yangyuwei_cluster/RenderFlow/data/HPK1_active_ligands.sdf'
with Chem.SDWriter(out_sdf_path) as writer:
    for sdf_file in sdf_files:
        mol = Chem.SDMolSupplier(sdf_file)[0]  # This will raise an error if the file is not a valid SDF
        if mol:
            mol = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if res != -1:
    
                AllChem.MMFFOptimizeMolecule(mol)
                writer.write(mol)
            else:
                continue

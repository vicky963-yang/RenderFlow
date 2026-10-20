import os
import time

os.environ['WANDB_API_KEY'] = 'your_own_key'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from pytorch_lightning.loggers import WandbLogger
import datetime, pytz
import argparse
import torch
torch.autograd.set_detect_anomaly(True)
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pytorch_lightning.callbacks import ModelCheckpoint

from data import load_dataloader
from models.sbdd_train_bfn_loop import RenderDockBFNModule
from utils.config import load_config
from utils.optimizer import AdamUniform
import utils.transforms as trans
from data.datasets import get_dataset
from data.datasets.pl_data import FOLLOW_BATCH
from callbacks.basic import RecoverCallback, GradientClip, NormalizerCallback, EMACallback
from callbacks.bfn_validation_callback  import (
    ClassifierValidationCallback, DockingTestCallback, ValidationCallback, CondMolGenValidationCallback,
    MolVisualizationCallback,
)
from absl import logging
import pytorch_lightning as pl
from lightning_fabric.utilities.rank_zero import rank_zero_only

def get_dataloader(cfg):
    if cfg.data.name == 'pl_tr':
        dataset, subsets = get_dataset(config=cfg.data, version='guided_v3')
        train_set, test_set = subsets['train'], subsets['test']
        cfg.model.protein_atom_feature_dim = dataset.protein_atom_feature_dim
        cfg.model.ligand_atom_feature_dim = dataset.ligand_atom_feature_dim
    else:
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
        transform_list = [
            protein_featurizer,
            ligand_featurizer,
            trans.FeaturizeLigandBond(),
            trans.NormalizeVina(),
            trans.AddMolProp(),
        ]
        if cfg.test_only:
            if 'inpainting' in cfg.evaluation.sample_num_atoms:
                transform_list.append(trans.AddScaffoldMask('./data/Mask_cd_test.pkl', cfg.evaluation.change_scaffold))
        transform = Compose(transform_list)
        cfg.model.protein_atom_feature_dim = protein_featurizer.feature_dim
        cfg.model.ligand_atom_feature_dim = ligand_featurizer.feature_dim
        dataset, subsets = get_dataset(config=cfg.data, transform=transform)
        train_set, test_set = subsets['train'], subsets['test']
    if 'val' in subsets and len(subsets['val']) > 0:
        val_set = subsets['val']
    else:
        val_set = test_set

    print(f"protein feature dim: {cfg.model.protein_atom_feature_dim}, " +
          f"ligand feature dim: {cfg.model.ligand_atom_feature_dim}")

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ["ligand_nbh_list"]

    logging.info(f"Training: {len(train_set)} Validation: {len(val_set)}")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )
    cfg.training.scheduler.max_iters = cfg.training.epochs * len(train_loader)

    return train_loader, val_loader, test_loader


class LinearInterpolateScheduler:
    def __init__(self, start_iter, end_iter, start_val, end_val, freq):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_val = start_val
        self.end_val = end_val
        self.freq = freq

    def __call__(self, iter):
        if iter < self.start_iter or iter % self.freq != 0 or iter == 0:
            return None

        p = (iter - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_val * (1 - p) + self.end_val * p

def get_logger(cfg):
    os.makedirs(cfg.accounting.wandb_logdir, exist_ok=True)
    # TODO save code
    if cfg.wandb_resume_id is not None:
        wandb_logger = WandbLogger(
            entity='vicky863',
            id=cfg.wandb_resume_id,
            project=cfg.project_name,
            offline=cfg.no_wandb,
            save_dir=cfg.accounting.wandb_logdir,
            resume='must',
        )
    else: # start a new run
        wandb_logger = WandbLogger(
            entity='vicky863',
            name=f"{cfg.exp_name}_{cfg.revision}"
            + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
            project=cfg.project_name,
            offline=cfg.no_wandb,
            save_dir=cfg.accounting.wandb_logdir,
        )  # add wandb parameters
    return wandb_logger

@rank_zero_only
def prepare_dir(path):
    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    args, extras = parser.parse_known_args()
    
    cfg_file = args.config

    cfg = load_config(cfg_file, cli_args=extras)
    wandb_logger = get_logger(cfg)
    train_loader, val_loader, test_loader = get_dataloader(cfg)
    wandb_logger.log_hyperparams(cfg)
    print(f"The config of this process is:\n{cfg}")

    lit = RenderDockBFNModule(
        cfg=cfg
    )
    last_ckpt_path = os.path.join(cfg.accounting.checkpoint_dir, "last.ckpt") # if full epoch trained
    # last_ckpt_path = os.path.join(cfg.accounting.checkpoint_dir, "last-v2.ckpt")[
    resume_flag = cfg.training.resume and os.path.exists(last_ckpt_path)
    checkpoint_dir = cfg.accounting.checkpoint_dir
    timestr = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(checkpoint_dir, timestr)
    prepare_dir(checkpoint_dir)
    callbacks = [
        RecoverCallback(
            latest_ckpt=last_ckpt_path,
            resume=resume_flag,
            recover_trigger_loss=1e7,
        ),
        GradientClip(max_grad_norm=cfg.training.max_grad_norm),  # time consuming
        NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
        ValidationCallback(
            dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
            atom_decoder=cfg.data.atom_decoder,
            atom_enc_mode=cfg.data.transform.ligand_atom_mode,
            atom_type_one_hot=False,
            single_bond=True,
            docking_config=cfg.evaluation.docking_config,
            val_freq=cfg.training.val_freq,
            max_epochs=cfg.training.epochs,
            model_type=cfg.model.net_name,
        ),
        DockingTestCallback(
            dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
            atom_decoder=cfg.data.atom_decoder,
            atom_enc_mode=cfg.data.transform.ligand_atom_mode,
            atom_type_one_hot=False,
            single_bond=True,
            docking_config=cfg.evaluation.docking_config,
        ),
        ModelCheckpoint(
            monitor="val/recon_loss",
            every_n_epochs=cfg.training.ckpt_freq,
            dirpath=checkpoint_dir,
            filename="epoch{epoch:02d}-val_loss{val/recon_loss:.2f}",
            save_top_k=5,
            mode="min",
            auto_insert_metric_name=False,
            save_last=True,
        ),
        EMACallback(decay=cfg.training.ema_decay, ema_device="cuda"),
    ]

    # if not cfg.evaluation.save_traj:
    #     callbacks.extend([
    #         # CondMolGenValidationCallback(
    #         #     dataset=None,  # TODO: implement CrossDockGen & NewBenchmark
    #         #     # atom_decoder={1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'},
    #         #     atom_decoder=cfg.data.atom_decoder,
    #         #     atom_enc_mode=cfg.data.transform.ligand_atom_mode,
    #         #     atom_type_one_hot=False,
    #         #     single_bond=True,
    #         #     docking_config=cfg.evaluation.docking_config,
    #         #     # single_bond=cfg.evaluation.single_bond,  # TODO: check compatibility
    #         # ),
    #         MolVisualizationCallback(
    #             # dataset=train_loader.loader.ds,
    #             # atom_enc_mode=cfg.data.transform.ligand_atom_mode,
    #             atom_decoder=cfg.data.atom_decoder,
    #             colors_dic=cfg.data.colors_dic,
    #             radius_dic=cfg.data.radius_dic,
    #         ),
    #         # ReconValidationCallback(
    #         #     val_freq=cfg.training.val_freq,
    #         # ),
    #     ])

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy='ddp',
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.training.epochs,
        check_val_every_n_epoch=cfg.training.ckpt_freq,
        devices=4,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        # val_check_interval=100,
        callbacks=callbacks,
        inference_mode=False,
    )

    if cfg.test_only:
        current_path = os.path.dirname(os.path.realpath(__file__))
        resume = os.path.join(current_path, cfg.test_ckpt_path)
        trainer.test(lit, dataloaders=test_loader, ckpt_path=resume)
    else:
        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit, dataloaders=test_loader)
    # train(cfg)

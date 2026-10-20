# RenderFlow
It is a multi-objective structure based generative model.
## Environment

## Data
We use the same data as TargetDiff[https://github.com/guanjq/targetdiff/tree/main?tab=readme-ov-file#data]. Data used for training / evaluating the model should be put in the data folder by default, and accessible in the data Google Drive folder.

To train the model from scratch, download the lmdb file and split file into data folder:
```
crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
crossdocked_pocket10_pose_split.pt
```
To evaluate the model on the test set, download and unzip the `test_set.zip` into data folder. It includes the original PDB files that will be used in Vina Docking.

## Training
According to the default parameters, the model can be trained as:
```
python train_bfn.py --config config/pocketsplatbfn.yaml
```

## Testing
or quick evaluation of the official checkpoint, refer to make evaluate in scripts.mk:
```
python train_bfn.py --test_only --no_wandb --ckpt_path ./checkpoints/${CKPT_NAME}
```

## Sampling
We provide the pretrained RenderFlow checkpoint here[https://drive.google.com/file/d/1ahg7BaUtFQQjZ1XMICW3ZRHGYxQ5JCIh/view?usp=drive_link].

### Sampling for pockets in the testset

```
sh test.sh
```

### Sampling from pdb file

As `extract_pocket.ipynb`, the pdb file can be preprocessed to pocket file.
Then the molecules can be sampled as:
```
sh test_sample.sh
```

The user need to change the file path in `*.sh` as real path.

## Evaluation

The samples of testset can be analyzed as:
```
sh eval.sh
```
and specific samples can be analyzed as:
```
sh eval_testsample.sh
```

***The paper describing this model has not yet been published.***

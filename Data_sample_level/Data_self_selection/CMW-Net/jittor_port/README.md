# CMW-Net Jittor Port (Phase 1-3)

This directory contains the first runnable Jittor migration sample for CMW-Net.

Current scope:
- `vnet_jittor.py`: Jittor implementation of the CMW-Net meta-weight network (`VNet`)
- `sanity_check.py`: forward/backward/optimization sanity run for `VNet`
- `training_utils_jittor.py`: phase-2 training helpers (`KeepWeightLoss`, `SelfAdaptiveTrainingCE`, EMA update, `train_ce_epoch`)
- `train_ce_demo_jittor.py`: phase-2 demo loop that mirrors the structure of `train_CE` with `(inputs, targets, index)` batches
- `train_meta_demo_jittor.py`: phase-3 demo loop with warmup + VNet-weighted meta-training on synthetic train/meta batches

## Environment

- Python 3.8+
- jittor

Install:

```bash
pip install jittor
```

## Quick Run

```bash
cd /home/runner/work/Auto-6ML/Auto-6ML/Data_sample_level/Data_self_selection/CMW-Net/jittor_port
python sanity_check.py --steps 5
```

If it prints step losses and `Jittor VNet sanity check finished.`, the phase-1 port is runnable.

Phase-2 demo run:

```bash
cd /home/runner/work/Auto-6ML/Auto-6ML/Data_sample_level/Data_self_selection/CMW-Net/jittor_port
python train_ce_demo_jittor.py --epochs 3
```

If it prints per-epoch `avg_loss/acc` and `Phase-2 Jittor train_CE demo finished.`, the phase-2 sample is runnable.

Phase-3 demo run:

```bash
cd /home/runner/work/Auto-6ML/Auto-6ML/Data_sample_level/Data_self_selection/CMW-Net/jittor_port
python train_meta_demo_jittor.py --epochs 4 --warmup-epochs 1
```

If it prints per-epoch `stage=warmup/meta` metrics and `Phase-3 Jittor meta-training demo finished.`, the phase-3 sample is runnable.

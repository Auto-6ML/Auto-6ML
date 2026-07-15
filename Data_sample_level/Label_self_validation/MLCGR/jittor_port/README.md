# MLCGR Jittor Port (Core Demo)

This directory contains a lightweight Jittor migration of key MLCGR components:

- `meta_net_jittor.py`: Jittor implementation of `MetaNet`
- `negative_learning_jittor.py`: Jittor implementation of negative-learning loss
- `sanity_check.py`: synthetic-data sanity run for forward/backward/optimization

## Environment

- Python 3.8+
- jittor

Install:

```bash
pip install jittor
```

## Quick Run

```bash
cd /home/runner/work/Auto-6ML/Auto-6ML/Data_sample_level/Label_self_validation/MLCGR/jittor_port
python sanity_check.py --steps 5
```

If it prints step losses and `MLCGR Jittor sanity check finished.`, this core port is runnable.

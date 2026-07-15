# NNCBO Jittor Port (Teacher-Weight Core Demo)

This directory contains a lightweight Jittor migration of the NNCBO teacher-weight module:

- `teachers_weight_matrix_jittor.py`: Jittor implementation of weighted teacher fusion
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
cd /home/runner/work/Auto-6ML/Auto-6ML/Data_sample_level/Label_self_validation/NNCBO/jittor_port
python sanity_check.py --steps 5
```

If it prints step losses and `NNCBO Jittor teacher-weight matrix sanity check finished.`, this core port is runnable.

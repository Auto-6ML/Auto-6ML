# CMW-Net Jittor Port (Phase 1)

This directory contains the first runnable Jittor migration sample for CMW-Net.

Current scope:
- `vnet_jittor.py`: Jittor implementation of the CMW-Net meta-weight network (`VNet`)
- `sanity_check.py`: forward/backward/optimization sanity run on synthetic data

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


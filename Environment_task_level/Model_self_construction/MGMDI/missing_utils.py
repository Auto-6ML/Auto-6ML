import numpy as np


def generate_missing_mask(x, missing_rate=0.2, missing_type="MCAR", seed=None):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n, d = x.shape
    missing_type = missing_type.upper()

    if missing_type == "MCAR":
        miss = rng.random((n, d)) < missing_rate

    elif missing_type == "MAR":
        driver = x[:, : max(1, d // 2)].mean(axis=1)
        driver = (driver - driver.min()) / (driver.max() - driver.min() + 1e-8)
        prob_row = missing_rate * (0.5 + driver)  # 平均附近仍接近 missing_rate
        prob_row = np.clip(prob_row, 0.0, 0.95)
        miss = rng.random((n, d)) < prob_row[:, None]

    elif missing_type == "MNAR":
        x_norm = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)
        prob = missing_rate * (0.5 + x_norm)
        prob = np.clip(prob, 0.0, 0.95)
        miss = rng.random((n, d)) < prob

    else:
        raise ValueError(f"未知缺失机制: {missing_type}，请使用 MCAR/MAR/MNAR")

    mask = (~miss).astype(np.float32)
    all_missing_rows = np.where(mask.sum(axis=1) == 0)[0]
    for i in all_missing_rows:
        mask[i, rng.integers(0, d)] = 1.0
    return mask


def apply_missing(x, mask):
     return np.asarray(x, dtype=np.float32) * np.asarray(mask, dtype=np.float32)

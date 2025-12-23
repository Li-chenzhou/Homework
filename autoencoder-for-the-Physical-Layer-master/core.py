from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def snr_db_to_noise_std(EbN0_dB: float, R: float) -> float:
    belta = 1.0 / (2.0 * R * (10.0 ** (EbN0_dB / 10.0)))
    return math.sqrt(belta)


def int_to_bits(i: int, k: int) -> np.ndarray:
    """整数 -> 二进制比特向量（长度k，低位在后）"""
    return np.array([(i >> j) & 1 for j in range(k)][::-1], dtype=np.uint8)


def build_bits_table(M: int, k: int) -> np.ndarray:
    """生成索引0..M-1对应的二进制比特表，shape=(M,k)。"""
    return np.stack([int_to_bits(i, k) for i in range(M)], axis=0)


class OneHotTileDataset(Dataset):
    def __init__(self, M: int, repeat: int = 1000):
        self.M = M
        self.repeat = repeat
        self.N = M * repeat

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cls = idx % self.M
        one_hot = torch.zeros(self.M, dtype=torch.float32)
        one_hot[cls] = 1.0
        return one_hot, cls


# -----------------------------
# 模型定义
# -----------------------------

class SISOAutoencoder(nn.Module):
    def __init__(self, M: int, n: int, hidden: int | None = None, power_norm: bool = True):
        super().__init__()
        hidden = hidden or M
        self.M = M
        self.n = n
        self.power_norm = power_norm

        self.enc = nn.Sequential(
            nn.Linear(M, n),
        )
        self.dec = nn.Sequential(
            nn.Linear(n, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, M),
        )

    def encode(self, x_oh: torch.Tensor) -> torch.Tensor:
        z = self.enc(x_oh)
        avg_power = z.pow(2).mean()
        return z / torch.sqrt(avg_power*2) * math.sqrt(self.n)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)  # logits

    def forward(self, x_oh: torch.Tensor, noise_std: float) -> torch.Tensor:
        z = self.encode(x_oh)
        if noise_std > 0:
            noise = torch.randn_like(z) * noise_std
            z = z + noise
        logits = self.decode(z)
        return logits


@dataclass
class TrainConfig:
    M: int
    n: int
    k: int
    R: float
    epochs: int = 200
    batch_size: int = 256
    train_snr_db: float = 3.0
    lr: float = 1e-3
    train_repeat: int = 800
    device: str = "cpu"
    seed: int = 42
    verbose: bool = True  # 是否打印每个epoch的loss进度


def train_autoencoder(cfg: TrainConfig) -> Tuple[SISOAutoencoder, Dict[str, List[float]]]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ds = OneHotTileDataset(cfg.M, repeat=cfg.train_repeat)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = SISOAutoencoder(cfg.M, cfg.n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "acc": []}
    noise_std = snr_db_to_noise_std(cfg.train_snr_db, cfg.R)

    model.train()
    for ep in range(cfg.epochs):
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        for x_oh, cls in dl:
            x_oh = x_oh.to(device)
            cls = cls.to(device)
            logits = model(x_oh, noise_std)
            loss = criterion(logits, cls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * x_oh.size(0)
            pred = torch.argmax(logits, dim=1)
            ep_correct += (pred == cls).sum().item()
            ep_total += cls.numel()
        ep_loss /= len(ds)
        ep_acc = ep_correct / max(1, ep_total)
        history["loss"].append(ep_loss)
        history["acc"].append(ep_acc)
        if cfg.verbose:
            print(f"[Train] epoch {ep+1}/{cfg.epochs} loss={ep_loss:.4f} acc={ep_acc:.4f}")
    return model, history


def build_square_qam_constellation(M: int, n: int) -> np.ndarray:
    assert n == 2, "QAM基线仅支持n=2 (I/Q)"
    m_side = int(round(math.sqrt(M)))

    # 坐标：等间距对称点，如{-3,-1,1,3}，按天然二进制顺序排列（不使用Gray映射）
    levels = np.arange(-(m_side - 1), (m_side + 1), 2, dtype=np.float64)

    pts = []
    for i in range(m_side):
        for j in range(m_side):
            I = levels[i]
            Q = levels[j]
            pts.append([I, Q])
    pts = np.array(pts, dtype=np.float64)

    # 归一化：平均能量= n（与AE一致）
    avg_energy = np.mean(np.sum(pts ** 2, axis=1))
    scale = math.sqrt(n / (avg_energy + 1e-12))
    pts = pts * scale
    return pts


def qam_nearest_detector(rx: np.ndarray, constel: np.ndarray) -> np.ndarray:
    d2 = ((rx[:, None, :] - constel[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


# -----------------------------
# 评估：BER与星座提取
# -----------------------------

@dataclass
class EvalConfig:
    M: int
    n: int
    k: int
    R: float
    snr_db_list: List[float]
    eval_repeat: int = 50000
    device: str = "cpu"


def get_ae_constellation(model: SISOAutoencoder, M: int, device: str = "cpu") -> np.ndarray:
    device_t = torch.device(device)
    model.eval()
    with torch.no_grad():
        eye = torch.eye(M, dtype=torch.float32, device=device_t)
        z = model.encode(eye)
    return z.cpu().numpy()


def evaluate_ber_autoencoder(
    model: SISOAutoencoder,
    cfg: EvalConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """评估AE在不同SNR下的BER曲线。

    返回：(snr_db_list, ber_list)
    说明：
    - 比特映射采用天然二进制（与QAM基线保持一致），通过“类别索引<->比特”换算BER。
    - 生成eval_repeat次均匀符号（等价于将单位阵重复eval_repeat次并打乱）。
    """
    device = torch.device(cfg.device)
    model.eval()
    snrs = np.array(cfg.snr_db_list, dtype=np.float64)
    ber = []

    # 准备符号与比特表
    bits_tbl = build_bits_table(cfg.M, cfg.k)  # (M,k)
    # 随机符号序列
    N = cfg.eval_repeat
    labels = np.random.randint(0, cfg.M, size=(N,), dtype=np.int64)
    bits_true = bits_tbl[labels]  # (N,k)

    with torch.no_grad():
        x_oh = torch.zeros(N, cfg.M, dtype=torch.float32)
        x_oh[torch.arange(N), torch.from_numpy(labels)] = 1.0
        x_oh = x_oh.to(device)

        for snr_db in snrs:
            noise_std = snr_db_to_noise_std(float(snr_db), cfg.R)
            logits = model(x_oh, noise_std)
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
            bits_pred = bits_tbl[pred_idx]
            bit_errors = np.not_equal(bits_true, bits_pred).sum()
            ber.append(bit_errors / (N * cfg.k))
    return snrs, np.array(ber, dtype=np.float64)


def evaluate_ber_qam(
    M: int,
    n: int,
    k: int,
    R: float,
    snr_db_list: List[float],
    eval_repeat: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """评估QAM基线的BER曲线（天然二进制映射、最近邻检测）。

    - M: 调制阶数（需为平方数以生成方形QAM）
    - n: 维度（必须为2）
    - k: 每符号比特数
    - R: 码率k/n
    - snr_db_list: SNR列表
    - eval_repeat: 评估样本数
    返回：(snr_db_list, ber_list)
    """
    constel = build_square_qam_constellation(M, n)  # (M,2)
    bits_tbl = build_bits_table(M, k)

    N = eval_repeat
    labels = np.random.randint(0, M, size=(N,), dtype=np.int64)
    bits_true = bits_tbl[labels]
    tx = constel[labels]  # (N,2)

    snrs = np.array(snr_db_list, dtype=np.float64)
    ber = []
    for snr_db in snrs:
        noise_std = snr_db_to_noise_std(float(snr_db), R)
        noise = np.random.randn(N, n) * noise_std
        rx = tx + noise
        pred_idx = qam_nearest_detector(rx, constel)
        bits_pred = bits_tbl[pred_idx]
        bit_errors = np.not_equal(bits_true, bits_pred).sum()
        ber.append(bit_errors / (N * k))
    return snrs, np.array(ber, dtype=np.float64)


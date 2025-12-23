"""
Plot utilities for SISO Autoencoder experiments.

提供：
- `plot_constellations(...)`：画AE实际星座与QAM理论星座对比（旧）
- `plot_theory_vs_noisy(...)`：画单一调制算法的“理论星座 vs 含噪实际点”
- `plot_ber_curves(...)`：画SNR-BER曲线（对数坐标）
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_constellations(
    ae_constel: np.ndarray,
    qam_constel: np.ndarray,
    title: str = "Constellation (AE vs QAM)",
    xlim: Optional[float] = None,
    ylim: Optional[float] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """绘制星座点对比。

    - ae_constel: AE提取的星座 (M,2)
    - qam_constel: QAM理论星座 (M,2)
    - title: 图标题
    - xlim/ylim: 轴范围（对称取值，如2则范围[-2,2]）
    - show: 是否展示
    - save_path: 若提供则保存到文件
    """
    assert ae_constel.shape[1] == 2 and qam_constel.shape[1] == 2, "仅支持2D星座绘制"

    plt.figure(figsize=(5,5))
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.scatter(qam_constel[:,0], qam_constel[:,1], c='C0', marker='x', label='QAM (theory)')
    plt.scatter(ae_constel[:,0],  ae_constel[:,1],  c='C3', marker='o', label='AE (learned)')
    if xlim is not None:
        plt.xlim(-abs(xlim), abs(xlim))
    if ylim is not None:
        plt.ylim(-abs(ylim), abs(ylim))
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_theory_vs_noisy(
    theory_constel: np.ndarray,
    noisy_points: np.ndarray,
    title: str = "Constellation: Theory vs Noisy",
    xlim: Optional[float] = None,
    ylim: Optional[float] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    theory_label: str = 'Theory',
    noisy_label: str = 'Noisy',
    noisy_alpha: float = 0.35,
) -> None:
    """绘制单一调制算法的“理论调制点 vs 实际含噪点”。

    - theory_constel: (M,2) 理论星座点（如AE编码点或QAM星座）
    - noisy_points: (M*K,2) 实际含噪点（每个星座点K个样本）
    - title/xlim/ylim: 图参数
    - theory_label/noisy_label: 图例
    - noisy_alpha: 含噪点透明度
    """
    assert theory_constel.shape[1] == 2, "仅支持2D星座绘制"
    assert noisy_points.shape[1] == 2, "仅支持2D星座绘制"

    plt.figure(figsize=(5,5))
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.scatter(noisy_points[:,0], noisy_points[:,1], c='C2', s=8, alpha=noisy_alpha, label=noisy_label)
    plt.scatter(theory_constel[:,0], theory_constel[:,1], c='k', marker='x', s=40, label=theory_label)
    if xlim is not None:
        plt.xlim(-abs(xlim), abs(xlim))
    if ylim is not None:
        plt.ylim(-abs(ylim), abs(ylim))
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_ber_curves(
    snr_db_list: Iterable[float],
    ber_ae: Iterable[float],
    ber_qam: Iterable[float],
    title: str = "SNR-BER (AE vs QAM)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """绘制SNR-BER曲线（半对数坐标）。

    - snr_db_list: Eb/N0(dB)列表
    - ber_ae: AE的BER
    - ber_qam: QAM的BER
    - title: 标题
    """
    snr_db = np.array(list(snr_db_list), dtype=float)
    ber_ae = np.array(list(ber_ae), dtype=float)
    ber_qam = np.array(list(ber_qam), dtype=float)

    plt.figure(figsize=(6,4))
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.yscale('log')
    plt.plot(snr_db, ber_qam, 'C0.-', label='QAM')
    plt.plot(snr_db, ber_ae,  'C3.-', label='Autoencoder')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.ylim(1e-5, 1)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


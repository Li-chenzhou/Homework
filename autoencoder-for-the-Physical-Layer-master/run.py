"""简化版运行脚本 - 仅保留必要参数"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch

from core import (
    TrainConfig, EvalConfig,
    train_autoencoder,
    get_ae_constellation,
    evaluate_ber_autoencoder,
    evaluate_ber_qam,
    build_square_qam_constellation,
    snr_db_to_noise_std,
    SISOAutoencoder,
)
from plot import plot_ber_curves, plot_theory_vs_noisy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SISO Autoencoder - Simplified")
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train 或 test')
    p.add_argument('--k', type=int, default=2, help='每符号比特数 (M=2^k)')
    p.add_argument('--n', type=int, default=2, help='发射维度')
    p.add_argument('--epochs', type=int, default=50, help='训练轮数')
    p.add_argument('--batch-size', type=int, default=256, help='批大小')
    p.add_argument('--lr', type=float, default=1e-3, help='学习率')
    p.add_argument('--snr', type=float, default=3.0, help='训练SNR (dB)')
    p.add_argument('--eval-snr-start', type=float, default=-2.0, help='评估SNR起始')
    p.add_argument('--eval-snr-stop', type=float, default=20.0, help='评估SNR终止')
    p.add_argument('--device', type=str, default='cpu', help='cpu 或 cuda')
    p.add_argument('--outdir', type=str, default='runs/siso', help='输出目录')
    p.add_argument('--weights', type=str, default=None, help='模型权重路径')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    M = 2 ** args.k
    R = args.k / args.n
    os.makedirs(args.outdir, exist_ok=True)

    if args.weights is None:
        args.weights = os.path.join(args.outdir, 'model.pt')

    device = torch.device(args.device)

    if args.mode == 'train':
        tcfg = TrainConfig(
            M=M, n=args.n, k=args.k, R=R,
            epochs=args.epochs, batch_size=args.batch_size, train_snr_db=args.snr,
            lr=args.lr, device=args.device
        )
        model, history = train_autoencoder(tcfg)
        torch.save(model.state_dict(), args.weights)
        np.savez(os.path.join(args.outdir, 'train_history.npz'),
                 loss=np.array(history['loss']), acc=np.array(history['acc']))
        print(f"✓ 权重已保存: {args.weights}")
        return

    # test 模式
    model = SISOAutoencoder(M, args.n).to(device)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"权重文件不存在: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # 星座与功率
    ae_constel = get_ae_constellation(model, M, device=args.device)
    qam_constel = build_square_qam_constellation(M, args.n)

    # BER 评估
    snr_list = list(np.arange(args.eval_snr_start, args.eval_snr_stop + 1e-9, 0.5))
    ecfg = EvalConfig(M=M, n=args.n, k=args.k, R=R, snr_db_list=snr_list, device=args.device)
    snr_ae, ber_ae = evaluate_ber_autoencoder(model, ecfg)
    snr_qam, ber_qam = evaluate_ber_qam(M, args.n, args.k, R, snr_list)

    # 星座图 1: AE
    with torch.no_grad():
        eye = torch.eye(M, dtype=torch.float32, device=device)
        x_oh = eye.repeat_interleave(100, dim=0)  # 100个采样/点
        z = model.encode(x_oh)
        noise_std = snr_db_to_noise_std(args.snr, R)
        noisy = z + torch.randn_like(z) * noise_std
    plot_theory_vs_noisy(
        ae_constel, noisy.cpu().numpy(),
        title=f"AE Constellation (Eb/N0={args.snr}dB)",
        show=True, save_path=os.path.join(args.outdir, 'constellation_ae.png')
    )

    # 星座图 2: QAM
    noise_std = snr_db_to_noise_std(args.snr, R)
    qam_noisy = np.repeat(qam_constel, 100, axis=0) + np.random.randn(M*100, args.n) * noise_std
    plot_theory_vs_noisy(
        qam_constel, qam_noisy,
        title=f"QAM Constellation (Eb/N0={args.snr}dB)",
        show=True, save_path=os.path.join(args.outdir, 'constellation_qam.png')
    )

    # BER 曲线
    plot_ber_curves(
        snr_ae, ber_ae, ber_qam,
        title=f"BER Curves (M={M}, n={args.n})",
        show=True, save_path=os.path.join(args.outdir, 'snr_ber.png')
    )

    # 保存结果
    np.savez(os.path.join(args.outdir, 'results.npz'),
             ae_constel=ae_constel, qam_constel=qam_constel,
             snr=np.array(snr_list), ber_ae=ber_ae, ber_qam=ber_qam)
    print(f"✓ 图表已生成 ({args.outdir})")


if __name__ == '__main__':
    main()

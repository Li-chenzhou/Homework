#####################################################
# 测试训练好的 SISO Autoencoder 在不同 Eb/N0 下的性能
#####################################################

# ===== 必要库 =====
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# ===== 系统参数 =====
k = 8
n = 2
M = 2 ** k
R = k / n

# ===== 自定义 BER（⚠️ 必须与训练时完全一致）=====
def BER(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)), axis=-1)

# ===== 生成 one-hot 符号 =====
eye_matrix = np.eye(M)

# ===== 生成测试数据 =====
x_try = np.tile(eye_matrix, (10000, 1))   # 减少重复次数以避免内存溢出
rd.shuffle(x_try)
print(x_try.shape)

# ===== 误码率列表 =====
ER = []

# ===== ⭐ 安全加载模型（关键修复点）=====
autoencoder = load_model(
    r'D:\00_Course\00_ComSys\08_SmartComm\Homework\autoencoder-for-the-Physical-Layer-master\autoencoder22.h5',
    custom_objects={
        'K': K,
        'BER': BER
    }
)

# ===== 遍历 Eb/N0 =====
for Eb_No_dB in np.arange(-2.0, 10.0, 0.5):

    # ---- 噪声方差 ----
    belta = 1 / (2 * R * (10 ** (Eb_No_dB / 10)))
    belta_sqrt = np.sqrt(belta)

    # ---- AWGN ----
    noise_try = belta_sqrt * np.random.randn(x_try.shape[0], n)

    # ---- 解码 ----
    decoded = autoencoder.predict(
        [x_try, noise_try],
        batch_size=1024,
        verbose=0
    )
    decoded_round = np.round(decoded)

    # ---- Block Error Rate（符号错误率）----
    error_rate = np.mean(
        np.not_equal(x_try, decoded_round).max(axis=1)
    )

    ER.append(error_rate)
    print(f"Eb/N0 = {Eb_No_dB:.1f} dB | BLER = {error_rate:.5e}")

# ===== 绘制 BLER 曲线 =====
plt.figure()
plt.yscale('log')
plt.plot(np.arange(-2.0, 10.0, 0.5), ER, 'r.-')
plt.grid(True)
plt.ylim(1e-5, 1)
plt.xlim(-2, 10)
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Block Error Rate")
plt.title("SISO Autoencoder BLER Performance")
plt.show()

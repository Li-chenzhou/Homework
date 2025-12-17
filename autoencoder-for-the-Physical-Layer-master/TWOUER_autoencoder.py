# 导入数值计算库numpy
import numpy as np
# 导入Keras的层：输入层、LSTM（未使用）、全连接层、高斯噪声层、Lambda层、丢弃层、嵌入层（注意：原代码embeddings为拼写习惯，实际应为Embedding）、展平层、加法层
from tensorflow.keras.layers import Input, LSTM, Dense, GaussianNoise, Lambda, Dropout, Embedding, Flatten, Add
# 导入Keras的模型类
from tensorflow.keras.models import Model
# 导入Keras的正则化模块
from tensorflow.keras import regularizers
# 导入Keras的批归一化层
from tensorflow.keras.layers import BatchNormalization  # 直接从layers导入
# 导入Keras的优化器：Adam和SGD
from tensorflow.keras.optimizers import Adam, SGD
# 导入Keras的后端函数（用于张量操作）
from tensorflow.keras import backend as K
# 导入Keras的回调函数基类（用于自定义训练过程中的操作）
from tensorflow.keras.callbacks import Callback
# 导入绘图库matplotlib.pyplot
import matplotlib.pyplot as plt
# 从numpy设置随机种子（保证实验可复现）
from numpy.random import seed
seed(1)
# 从tensorflow设置随机种子（保证实验可复现）
from tensorflow import random as tf_random  
tf_random.set_seed(3)  # TF2.x的随机种子



# ====================== 初始化超参数 ======================
NUM_EPOCHS = 100  # 预设训练轮数（实际训练用了45轮）
BATCH_SIZE = 32  # 批次大小
M = 4  # 调制阶数（4-QAM，每个符号携带2比特）
k = np.log2(M)  # 每个符号的比特数（log2(4)=2）
k = int(k)  # 转为整数
n_channel = 2  # 信道维度（2维实数值，对应复平面的IQ分量）
emb_k = 4  # 嵌入层的输出维度
R = k / n_channel  # 码率（2/2=1）
train_data_size = 10000  # 训练数据量
bertest_data_size = 50000  # 误码率测试数据量
EbNodB_train = 7  # 训练时的信噪比（Eb/No，单位dB）
EbNo_train = 10 ** (EbNodB_train / 10.0)  # 将dB转换为线性信噪比
# 计算训练时的噪声标准差（基于通信系统的功率约束和信噪比）
noise_std = np.sqrt(1 / (2 * R * EbNo_train))
alpha = K.variable(0.5)  # 动态损失权重：用户1的损失权重（初始值0.5）
beta = K.variable(0.5)   # 动态损失权重：用户2的损失权重（初始值0.5）

# ====================== 定义混合AWGN信道函数 ======================
# 功能：模拟两用户干扰信道，添加用户间干扰和高斯噪声
def mixed_AWGN(x):
    signal = x[0]  # 当前用户的发射信号
    interference = x[1]  # 另一个用户的干扰信号
    # 生成与信号形状相同的高斯噪声（均值0，标准差noise_std）
    noise = K.random_normal(K.shape(signal), mean=0, stddev=noise_std)
    signal = Add()([signal, interference])  # 信号叠加干扰
    signal = Add()([signal, noise])  # 叠加高斯噪声
    return signal  # 返回经过干扰和噪声的接收信号

# ====================== 自定义回调函数：动态调整损失权重 ======================
# 功能：在每个训练轮次结束后，根据两个用户的损失值动态更新alpha和beta
class Mycallback(Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha  # 用户1的损失权重变量
        self.beta = beta    # 用户2的损失权重变量
        self.epoch_num = 0  # 记录当前训练轮次
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_num += 1  # 轮次计数+1
        # 获取用户1和用户2的损失值（从训练日志中）
        loss1 = logs.get('u1_receiver_loss')
        loss2 = logs.get('u2_receiver_loss')
        # 打印训练信息（便于监控）
        print("epoch %d" % self.epoch_num)
        print("total_loss%f" % logs.get('loss'))
        print("u1_loss %f" % (loss1))
        print("u2_loss %f" % (loss2))
        # 计算新的权重：损失值占比（损失越大，权重越高，保证公平训练）
        a = loss1 / (loss1 + loss2)
        b = 1 - a
        # 更新Keras变量的取值
        K.set_value(self.alpha, a)
        K.set_value(self.beta, b)
        # 打印更新后的权重（验证是否生效）
        print("alpha %f" % K.get_value(alpha))
        print("beta %f" % K.get_value(beta))
        print("selfalpha %f" % K.get_value(self.alpha))
        print("selfbeta %f" % K.get_value(self.beta))

# ====================== 生成训练和测试数据 ======================
# ---- 生成用户1的标签数据 ----
seed(1)  # 设置随机种子（保证数据可复现）
# 生成训练标签：随机整数（0~M-1），数量为train_data_size
train_label_s1 = np.random.randint(M, size=train_data_size)
train_label_out_s1 = train_label_s1.reshape((-1, 1))  # 调整形状为(样本数, 1)（适配稀疏分类损失）
# 生成测试标签：数量为bertest_data_size
test_label_s1 = np.random.randint(M, size=bertest_data_size)
test_label_out_s1 = test_label_s1.reshape((-1, 1))

# ---- 生成用户2的标签数据 ----
seed(2)  # 独立随机种子（避免与用户1数据相同）
train_label_s2 = np.random.randint(M, size=train_data_size)
train_label_out_s2 = train_label_s2.reshape((-1, 1))
test_label_s2 = np.random.randint(M, size=bertest_data_size)
test_label_out_s2 = test_label_s2.reshape((-1, 1))

# ====================== 构建两用户的自编码器模型 ======================
# ---- 构建用户1的发射机（编码器） ----
u1_input_signal = Input(shape=(1,))  # 输入层：接收1维的符号标签（0~3）
# 嵌入层：将整数标签映射为emb_k维的稠密向量（替代one-hot编码，降低维度）
u1_encoded = Embedding(input_dim=M, output_dim=emb_k, input_length=1)(u1_input_signal)
u1_encoded1 = Flatten()(u1_encoded)  # 展平：将(样本数,1,emb_k)转为(样本数,emb_k)
u1_encoded2 = Dense(M, activation='relu')(u1_encoded1)  # 全连接层：增加非线性
u1_encoded3 = Dense(n_channel, activation='linear')(u1_encoded2)  # 映射到信道维度（2维）
# 能量约束：L2归一化后乘以sqrt(n_channel)，保证信号能量恒定
u1_encoded4 = Lambda(lambda x: np.sqrt(n_channel)*K.l2_normalize(x, axis=1))(u1_encoded3)

# ---- 构建用户2的发射机（编码器） ----
u2_input_signal = Input(shape=(1,))  # 输入层
u2_encoded = Embedding(input_dim=M, output_dim=emb_k, input_length=1)(u2_input_signal)
u2_encoded1 = Flatten()(u2_encoded)
u2_encoded2 = Dense(M, activation='relu')(u2_encoded1)
u2_encoded3 = Dense(n_channel, activation='linear')(u2_encoded2)
u2_encoded4 = Lambda(lambda x: np.sqrt(n_channel)*K.l2_normalize(x, axis=1))(u2_encoded3)

# ---- 混合AWGN信道层 ----
# 用户1的接收信号：自身信号 + 用户2的干扰 + 噪声
u1_channel_out = Lambda(lambda x: mixed_AWGN(x))([u1_encoded4, u2_encoded4])
# 用户2的接收信号：自身信号 + 用户1的干扰 + 噪声
u2_channel_out = Lambda(lambda x: mixed_AWGN(x))([u2_encoded4, u1_encoded4])

# ---- 构建用户1的接收机（解码器） ----
# 全连接层：从信道维度映射到M维，relu激活
u1_decoded = Dense(M, activation='relu', name='u1_pre_receiver')(u1_channel_out)
# 输出层：softmax激活，输出符号的概率分布（命名为u1_receiver，便于监控损失）
u1_decoded1 = Dense(M, activation='softmax', name='u1_receiver')(u1_decoded)

# ---- 构建用户2的接收机（解码器） ----
u2_decoded = Dense(M, activation='relu', name='u2_pre_receiver')(u2_channel_out)
u2_decoded1 = Dense(M, activation='softmax', name='u2_receiver')(u2_decoded)

# ---- 定义整体两用户自编码器模型 ----
twouser_autoencoder = Model(
    inputs=[u1_input_signal, u2_input_signal],  # 输入：两个用户的标签
    outputs=[u1_decoded1, u2_decoded1]  # 输出：两个用户的解码概率
)
# 定义Adam优化器（学习率0.01）
adam = Adam(lr=0.01)
# 编译模型：
# - 优化器：adam
# - 损失函数：稀疏分类交叉熵（适用于整数标签，无需one-hot编码）
# - 损失权重：动态的alpha和beta
twouser_autoencoder.compile(
    optimizer=adam,
    loss='sparse_categorical_crossentropy',
    loss_weights=[alpha, beta]
)
# 打印模型结构摘要（便于查看各层参数和形状）
print(twouser_autoencoder.summary())
# 训练模型：
# - 输入：两个用户的训练标签
# - 输出：两个用户的训练标签（自编码器重构目标）
# - 训练轮数：45（覆盖预设的100轮）
# - 批次大小：32
# - 回调函数：自定义的Mycallback（动态调整损失权重）
twouser_autoencoder.fit(
    [train_label_s1, train_label_s2],
    [train_label_out_s1, train_label_out_s2],
    epochs=45,
    batch_size=32,
    callbacks=[Mycallback(alpha, beta)]
)

# # （注释）绘制模型结构图（需安装pydot和graphviz）
from tensorflow.keras.utils import plot_model  # 去掉vis_utils子模块，直接从utils导入
plot_model(twouser_autoencoder, to_file='model.png')

# ====================== 提取单个用户的编码器和解码器模型 ======================
# ---- 提取用户1的编码器和解码器 ----
# 编码器：输入标签，输出编码后的信号
u1_encoder = Model(u1_input_signal, u1_encoded4)
# 解码器输入层：接收信道维度的信号
u1_encoded_input = Input(shape=(n_channel,))
# 从整体模型中获取接收机层的权重
u1_deco = twouser_autoencoder.get_layer("u1_pre_receiver")(u1_encoded_input)
u1_deco = twouser_autoencoder.get_layer("u1_receiver")(u1_deco)
# 解码器模型：输入信道信号，输出解码概率
u1_decoder = Model(u1_encoded_input, u1_deco)

# ---- 提取用户2的编码器和解码器 ----
u2_encoder = Model(u2_input_signal, u2_encoded4)
u2_encoded_input = Input(shape=(n_channel,))
u2_deco = twouser_autoencoder.get_layer("u2_pre_receiver")(u2_encoded_input)
u2_deco = twouser_autoencoder.get_layer("u2_receiver")(u2_deco)
u2_decoder = Model(u2_encoded_input, u2_deco)

# ====================== 绘制星座图（编码后的符号分布） ======================
# ---- 绘制用户1的星座点 ----
u1_scatter_plot = []
for i in range(M):
    # 预测每个符号（0~3）的编码输出
    u1_scatter_plot.append(u1_encoder.predict(np.expand_dims(i, axis=0)))
u1_scatter_plot = np.array(u1_scatter_plot)  # 转为数组
u1_scatter_plot = u1_scatter_plot.reshape(M, 2)  # 调整形状为(M, 2)（便于绘图）
# 绘制散点图（红色）
plt.scatter(u1_scatter_plot[:, 0], u1_scatter_plot[:, 1], color='red', label='user1(2,2)')

# ---- 绘制用户2的星座点 ----
u2_scatter_plot = []
for i in range(M):
    u2_scatter_plot.append(u2_encoder.predict(np.expand_dims(i, axis=0)))
u2_scatter_plot = np.array(u2_scatter_plot)
u2_scatter_plot = u2_scatter_plot.reshape(M, 2)
# 绘制散点图（蓝色）
plt.scatter(u2_scatter_plot[:, 0], u2_scatter_plot[:, 1], color='blue', label='user2(2,2)')

# 图表美化
plt.legend(loc='upper left', ncol=1)  # 图例
plt.axis((-2.5, 2.5, -2.5, 2.5))  # 坐标轴范围
plt.grid()  # 网格线
fig = plt.gcf()  # 获取当前图表
fig.set_size_inches(16, 12)  # 设置尺寸
# fig.savefig('graph/TwoUsercons(2,2)0326_1.png',dpi=100)  # 保存图片（注释）
plt.show()  # 显示图表

# ====================== 计算不同信噪比下的误码率（BER） ======================
# 定义信噪比范围：0~14dB，共28个点
EbNodB_range = list(np.linspace(0, 14, 28))
ber = [None] * len(EbNodB_range)  # 存储平均误码率
u1_ber = [None] * len(EbNodB_range)  # 存储用户1的误码率
u2_ber = [None] * len(EbNodB_range)  # 存储用户2的误码率

# 遍历每个信噪比
for n in range(0, len(EbNodB_range)):
    EbNo = 10 ** (EbNodB_range[n] / 10.0)  # 转换为线性信噪比
    noise_std = np.sqrt(1 / (2 * R * EbNo))  # 计算当前信噪比的噪声标准差
    nn = bertest_data_size  # 测试数据量
    # 生成高斯噪声（两个用户的噪声独立）
    noise1 = noise_std * np.random.randn(nn, n_channel)
    noise2 = noise_std * np.random.randn(nn, n_channel)
    
    # 编码：生成两个用户的发射信号
    u1_encoded_signal = u1_encoder.predict(test_label_s1)
    u2_encoded_signal = u2_encoder.predict(test_label_s2)
    
    # 信道：叠加干扰和噪声
    u1_final_signal = u1_encoded_signal + u2_encoded_signal + noise1
    u2_final_signal = u2_encoded_signal + u1_encoded_signal + noise2
    
    # 解码：预测接收信号的符号
    u1_pred_final_signal = u1_decoder.predict(u1_final_signal)
    u2_pred_final_signal = u2_decoder.predict(u2_final_signal)
    
    # 取概率最大的符号作为预测结果
    u1_pred_output = np.argmax(u1_pred_final_signal, axis=1)
    u2_pred_output = np.argmax(u2_pred_final_signal, axis=1)
    
    # 计算误码数：预测结果与真实标签不相等的数量
    u1_no_errors = (u1_pred_output != test_label_s1).astype(int).sum()
    u2_no_errors = (u2_pred_output != test_label_s2).astype(int).sum()
    
    # 计算误码率（误码数/总样本数）
    u1_ber[n] = u1_no_errors / nn
    u2_ber[n] = u2_no_errors / nn
    ber[n] = (u1_ber[n] + u2_ber[n]) / 2  # 平均误码率
    
    # 打印结果（便于监控）
    print('U1_SNR:', EbNodB_range[n], 'U1_BER:', u1_ber[n])
    print('U2_SNR:', EbNodB_range[n], 'U2_BER:', u2_ber[n])
    print('SNR:', EbNodB_range[n], 'BER:', ber[n])

# ====================== 绘制误码率曲线 ======================
# 绘制用户1的误码率曲线
plt.plot(EbNodB_range, u1_ber, label='TwoUserSNR(2,2)U1,emb_k=4')
# 绘制用户2的误码率曲线
plt.plot(EbNodB_range, u2_ber, label='TwoUserSNR(2,2)U2,emb_k=4')
# 绘制平均误码率曲线
plt.plot(EbNodB_range, ber, label='TwoUserSNR(2,2),emb_k=4')

# 图表美化
plt.yscale('log')  # y轴对数刻度（适合展示误码率的大范围变化）
plt.xlabel('SNR (Eb/No dB)')  # x轴标签
plt.ylabel('Block Error Rate')  # y轴标签
plt.grid()  # 网格线
plt.legend(loc='upper right', ncol=1)  # 图例

fig = plt.gcf()
fig.set_size_inches(16, 12)  # 设置尺寸
fig.savefig('graph/TwoUserSNR(2,2)0326_1.png',dpi=100)  # 保存图片（注释）
plt.show()  # 显示图表
# 导入Keras所需的层（输入层、全连接层、高斯噪声层、Lambda层）
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Lambda
# 导入Keras的模型类
from tensorflow.keras.models import Model  
# 导入Keras的后端函数
from tensorflow.keras import backend as K
# 导入数值计算库numpy
import numpy as np 
# 导入随机数库random（用于打乱数据）
import random as rd
# 导入绘图库matplotlib.pyplot（用于可视化）
import matplotlib.pyplot as plt 

# 初始化通信系统参数
k = 8  # 每个符号携带的信息比特数
n = 2  # 编码后的信号维度（调制符号的维度，此处为2D）
M = 2**k  # 调制阶数（可能的符号数量），2^2=4（类似QPSK调制）
R = k/n  # 码率（信息比特数/传输符号维度），此处为1

# 生成单位矩阵（作为one-hot编码的符号集合，每行代表一个独特的符号）
eye_matrix = np.eye(M)
# 生成训练数据：将单位矩阵重复600次，形状为(600*M, M)，确保每个符号有足够多的训练样本
x_train = np.tile(eye_matrix, (600, 1))  
# 生成测试数据：单位矩阵重复100次，形状为(100*M, M)
x_test = np.tile(eye_matrix, (100, 1)) 
# 生成验证用数据：单位矩阵重复1000次，形状为(1000*M, M)
x_try = np.tile(eye_matrix, (1000, 1)) 
# 打乱训练数据顺序（增加随机性，避免符号顺序影响训练）
rd.shuffle(x_train)
# 打乱测试数据顺序
rd.shuffle(x_test)
# 打乱验证数据顺序
rd.shuffle(x_try)
# 打印训练数据形状（验证数据生成是否正确）
print(x_train.shape)  
# 打印测试数据形状
print(x_test.shape) 

# 自定义误码率（BER）计算函数（Keras指标）
def BER(y_true, y_pred):
    # 计算真实标签与预测值四舍五入后的差异，取均值作为误码率（按特征轴计算）
    return K.mean(K.not_equal(y_true, K.round(y_pred)), axis=-1)  

# 设置训练时的信噪比参数（Eb/No，单位dB）
Eb_No_dB = 7
# 将dB转换为线性信噪比（噪声功率的倒数）
noise = 1/(10**(Eb_No_dB/10))
# 计算噪声标准差（基于噪声功率）
noise_sigma = np.sqrt(noise)
# 计算噪声方差相关参数（结合码率R，用于生成信道噪声）
belta = 1/(2*R*(10**(Eb_No_dB/10)))
# 噪声标准差（用于高斯噪声层）
belta_sqrt = np.sqrt(belta)

# 定义自编码器的输入层，形状为(M,)（对应one-hot编码的符号）
input_sys = Input(shape=(M,))
  
# 构建深度自编码器的编码部分
# 第一个全连接层：输入为M维，输出为M维，使用relu激活函数
encoded = Dense(M, activation='relu')(input_sys)  
# 第二个全连接层：将维度从M压缩到n（编码后的信号维度）
encoded = Dense(n)(encoded) 
# （注释）可选的L2正则化层（用于约束参数，防止过拟合）
# encoded = ActivityRegularization(l2=0.02)(encoded)
# Lambda层：对编码后的信号进行能量约束（L2归一化后乘以sqrt(n)，确保信号能量恒定为n）
encoded = Lambda(lambda x: np.sqrt(n) * K.l2_normalize(x, axis=1))(encoded) # 能量约束
# （注释）另一种功率约束方式（归一化到平均功率为1）
# encoded = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(encoded) # 平均功率约束
# 高斯噪声层：向编码后的信号添加高斯噪声，噪声标准差为belta_sqrt（模拟信道噪声）
encoded_noise = GaussianNoise(belta_sqrt)(encoded)# 噪声层

# 构建自编码器的解码部分
# 第一个全连接层：从n维映射到M维，使用relu激活函数
decoded = Dense(M, activation='relu')(encoded_noise)
# 第二个全连接层：从M维映射回M维，使用softmax激活函数（输出符号的概率分布）
decoded = Dense(M, activation='softmax')(decoded)

# 定义完整的自编码器模型：输入为原始信号，输出为解码后的信号
autoencoder = Model(inputs=input_sys, outputs=decoded)  
# 定义编码器模型：输入为原始信号，输出为编码后的信号（用于后续星座图绘制）
encoder = Model(inputs=input_sys, outputs=encoded)

# 编译自编码器：使用adam优化器，损失函数为交叉熵（适合多分类），监控指标为二进制准确率和自定义BER
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['binary_accuracy', BER])  
  
# 训练自编码器：输入训练数据x_train，目标输出也是x_train（自编码器重构输入），训练200个epoch，使用测试集验证
hist = autoencoder.fit(x_train, x_train, epochs=15, validation_data=(x_test, x_test))# 未指定batch_size，使用默认值

# （注释）计算误码个数的代码：
# 用编码器编码验证数据
encoded_sys = encoder.predict(x_try) 
# 用自编码器解码验证数据
decoded_sys = autoencoder.predict(x_try)
# 对解码结果四舍五入（转为0/1的one-hot格式）
decoded_sys_round = np.round(decoded_sys)
# 计算误码率：每行若有一个位置错误则视为符号错误，求平均
error_rate = np.mean(np.not_equal(x_try,decoded_sys_round).max(axis=1))

# 绘制星座图（编码后的符号在2D平面的分布）
# 用编码器对所有基准符号（单位矩阵的行）进行编码
encoded_planisphere = encoder.predict(eye_matrix) 
# 图表标题
plt.title('Constellation')
# 设置x轴范围
plt.xlim(-2, 2)
# 设置y轴范围
plt.ylim(-2, 2)
# 绘制编码后的符号点（红色）
plt.plot(encoded_planisphere[:,0], encoded_planisphere[:,1], 'r.')
# 显示网格
plt.grid(True)

# 绘制训练损失曲线
# 创建新的图表
plt.figure()
# 绘制训练过程中的损失值
plt.plot(hist.history['loss'])
# 图表标题
plt.title('model loss')
# y轴标签
plt.ylabel('loss')
# x轴标签（训练轮次）
plt.xlabel('epoch')
plt.show()
# 导入Keras所需的层：Input（输入层）、Dense（全连接层）、Lambda（匿名函数层）、Add（加法层）
from tensorflow.keras.layers import Input, Dense, Lambda, Add
# 导入Keras的模型类
from tensorflow.keras.models import Model  
# 导入Keras后端（用于底层张量操作）
from tensorflow.keras import backend as K
# 导入数值计算库numpy
import numpy as np 
# 导入随机数库random（用于数据打乱）
import random as rd
# 导入绘图库matplotlib.pyplot
import matplotlib.pyplot as plt 

# 随机种子部分
from numpy.random import seed
from tensorflow import random as tf_random  
seed(5)  # numpy随机种子
tf_random.set_seed(3)  # TF2.x的随机种子


# 初始化参数
k = 2  # 编码前的比特数（每个符号携带的比特数）
n = 2  # 编码后的信号维度（调制符号的维度）
M = 2**k  # 调制阶数（可能的符号数量），此处2^2=4，类似QPSK
R = k/n  # 码率（信息比特数/传输符号维度），此处2/2=1

# 生成单位矩阵（作为one-hot编码的符号集合，每行代表一个独特的符号）
eye_matrix = np.eye(M)
# 扩展训练数据：将单位矩阵重复2000次，生成训练样本（形状为(2000*M, M)）
x_train = np.tile(eye_matrix, (2000, 1))  
# 扩展测试数据：重复100次，生成测试样本
x_test = np.tile(eye_matrix, (100, 1)) 
# 扩展尝试数据（用于后续误码率评估）：重复10000次
x_try = np.tile(eye_matrix, (10000, 1)) 
# 打乱训练、测试、尝试数据的顺序（增加随机性，提升模型泛化能力）
rd.shuffle(x_train)
rd.shuffle(x_test)
rd.shuffle(x_try)
# 打印训练和测试数据的形状（验证数据生成是否正确）
print(x_train.shape)  
print(x_test.shape) 

# 定义误码率（BER）计算函数（作为模型评价指标）
def BER(y_true, y_pred):
    # 对比真实标签与预测值（四舍五入后），计算不相等的比例（平均误码率）
    return K.mean(K.not_equal(y_true, K.round(y_pred)), axis=-1)  

# 设置信噪比（SNR）相关参数
Eb_No_dB = 7  # 以dB为单位的信噪比（Eb/N0）
noise = 1/(10**(Eb_No_dB/10))  # 将dB转换为线性尺度的噪声功率（近似）
noise_sigma = np.sqrt(noise)  # 噪声标准差（辅助计算，未直接使用）
# 计算噪声方差系数：根据码率R和SNR，推导得到符合能量约束的噪声方差
belta = 1/(2*R*(10**(Eb_No_dB/10)))
belta_sqrt = np.sqrt(belta)  # 噪声标准差（实际用于生成噪声）

# 生成训练、测试、尝试数据对应的噪声（服从均值0、标准差belta_sqrt的高斯分布）
# 噪声形状与对应数据的样本数匹配，维度为n（与编码后信号维度一致）
noise_train = belta_sqrt * np.random.randn(np.shape(x_train)[0],n)
noise_test = belta_sqrt * np.random.randn(np.shape(x_test)[0],n)
noise_try = belta_sqrt * np.random.randn(np.shape(x_try)[0],n)

# 构建自编码器模型
# 定义系统输入层：接收维度为M的one-hot符号（M为调制阶数）
input_sys = Input(shape=(M,))
# 定义噪声输入层：接收维度为n的噪声（与编码后信号维度一致）
input_noise = Input(shape=(n,))
  
# 编码器部分（发射端）
# 全连接层：将输入从M维映射到M维，使用relu激活函数（增加非线性）
encoded = Dense(M, activation='relu')(input_sys)  
# 全连接层：将M维映射到n维（得到编码后的信号雏形）
encoded = Dense(n)(encoded) 
# 能量约束：对编码信号进行L2归一化后乘以sqrt(n)，保证信号能量为n（符合通信系统功率约束）
encoded = Lambda(lambda x: np.sqrt(n) * K.l2_normalize(x, axis=1))(encoded) 
# 模拟信道：将编码后的信号与噪声相加（Add层实现）
encoded_noise = Add()([encoded, input_noise]) 

# 解码器部分（接收端）
# 全连接层：将含噪声的n维信号映射到M维，使用relu激活函数
decoded = Dense(M, activation='relu')(encoded_noise)
# 全连接层：将M维映射回M维，使用softmax激活函数（输出符号的概率分布）
decoded = Dense(M, activation='softmax')(decoded)

# 定义完整自编码器模型：输入为信号和噪声，输出为解码后的符号概率
autoencoder = Model(inputs=[input_sys,input_noise], outputs=decoded)  
# 定义编码器模型：仅保留发射端部分（用于后续生成星座图）
encoder = Model(inputs=input_sys, outputs=encoded)

# 编译自编码器：优化器使用adam，损失函数为分类交叉熵（多分类问题），评价指标包括二进制准确率和BER
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['binary_accuracy',BER])  

# 训练自编码器：输入为训练数据和对应的噪声，标签为训练数据本身（自编码器目标是重构输入）
# 训练100轮，批次大小32，使用测试数据和噪声作为验证集
hist = autoencoder.fit([x_train,noise_train], x_train, epochs=100, batch_size=32, validation_data=([x_test, noise_test], x_test))

# （注释部分）计算误码率：用尝试数据测试模型性能
encoded_sys = encoder.predict(x_try)  # 编码尝试数据
decoded_sys = autoencoder.predict([x_try,noise_try])  # 解码含噪声的信号
decoded_sys_round = np.round(decoded_sys)  # 对预测结果四舍五入（转为0/1）
error_rate = np.mean(np.not_equal(x_try,decoded_sys_round).max(axis=1))  # 计算平均误码率

# 绘制星座图：用编码器对所有可能的符号（单位矩阵的行）进行编码，得到星座点
encoded_planisphere = encoder.predict(eye_matrix) 
plt.figure()  # 创建新图
plt.title('Constellation')  # 标题：星座图
plt.xlim(-2, 2)  # x轴范围
plt.ylim(-2, 2)  # y轴范围
plt.plot(encoded_planisphere[:,0], encoded_planisphere[:,1], 'r.')  # 绘制星座点（红色点）
plt.grid(True)  # 显示网格

# 绘制训练损失曲线
plt.figure()  # 创建新图
plt.plot(hist.history['loss'])  # 绘制训练损失随轮次的变化
plt.title('model loss')  # 标题：模型损失
plt.ylabel('loss')  # y轴标签：损失值
plt.xlabel('epoch')  # x轴标签：训练轮次
plt.show()
# 保存模型：取消注释可将训练好的模型保存为HDF5文件
autoencoder.save('C:\\Users\\33958\\Downloads\\autoencoder-for-the-Physical-Layer-master\\autoencoder-for-the-Physical-Layer-master\\autoencoder22.h5')
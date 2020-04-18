import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle   #对数据进行打乱
from sklearn.preprocessing import scale #一会对数据做归一化

# 数据准备

df=pd.read_csv("dataset/boston.csv")
# print(df.describe())
ds=df.values

x_data=ds[:,:12]
y_data=ds[:,12]
# print(x_data.shape)
train_num=300   #训练集数目
valid_num=100   #验证集数目
test_num=len(x_data)-train_num-valid_num    #测试集数目
# 训练集划分
x_train=x_data[:train_num]
y_train=y_data[:train_num]
# 验证集划分
x_valid=x_data[train_num:train_num+valid_num]
y_valid=y_data[train_num:train_num+valid_num]
# 测试集划分
x_test=x_data[train_num+valid_num:train_num+valid_num+test_num]
y_test=y_data[train_num+valid_num:train_num+valid_num+test_num]

x_train=tf.cast(scale(x_train),dtype=tf.float32)
x_valid=tf.cast(scale(x_valid),dtype=tf.float32)
x_test=tf.cast(scale(x_test),dtype=tf.float32)

# 构建模型
def model(x,w,b):
    return tf.matmul(x,w)+b

# 创建待优化变量
W=tf.Variable(tf.random.normal([12,1],mean=0.0,stddev=1.0,dtype=tf.float32))
B=tf.Variable(tf.zeros(1),dtype=tf.float32)

# 模型训练

    ## 设置超参数
training_epochs=70  #迭代次数
learning_rate=0.0005 #学习率 
barch_size=10   #批量训练一次样本数
    ##定义损失函数
def loss(x,y,w,b):
    err=model(x,w,b)-y
    squared_err=tf.square(err)  #   求平方，得出方差
    return tf.reduce_mean(squared_err)  #求均值，得出均方差

    ##定义梯度计算函数
def grad(x,y,w,b):
    with tf.GradientTape() as tape:
        loss_=loss(x,y,w,b)
    return tape.gradient(loss_,[w,b])#返回梯度向量

# 选择优化器(梯度下降优化器)
optimizer=tf.keras.optimizers.SGD(learning_rate)

# 迭代训练
loss_list_train=[]  #用于保存训练集loss值得列表
loss_list_valid=[]  #用于保存验证集loss值得列表
total_step=int(train_num/barch_size)

for epoch in range(training_epochs):
    for step in range(total_step):
        xs=x_train[step*barch_size:(step+1)*barch_size,:]
        ys=y_train[step*barch_size:(step+1)*barch_size]
        grads=grad(xs,ys,W,B)
        optimizer.apply_gradients(zip(grads,[W,B]))
    loss_train=loss(x_train,y_train,W,B).numpy()
    loss_valid=loss(x_valid,y_valid,W,B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    print("epoch={:3d},train_loss={:4f},valid_loss={:4f}".format(epoch+1,loss_train,loss_valid))
# 可视化损失值
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train,'blue',label="Train Loss")
plt.plot(loss_list_valid,'red',label="Valid Loss")
plt.show()
# 查看测试集得损失
print("Test_Loss:{:.4f}".format(loss(x_test,y_test,W,B).numpy()))

# 从测试集中随机选一条

test_house_id=np.random.randint(0,test_num)
y=y_test[test_house_id]
y_pred=model(x_test,W,B)[test_house_id]
y_predit=tf.reshape(y_pred,()).numpy()
print("House id",test_house_id,"Actual value",y,"Predicted value",y_predit)


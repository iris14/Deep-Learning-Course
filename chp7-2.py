import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
mnist=tf.keras.datasets.mnist
plt.rcParams["font.sans-serif"]="SimHei"
(train_x,train_y),(test_x,test_y)=mnist.load_data()
ran_len=len(test_x)
plt.figure(16)
for i in range(16):
    num=np.random.randint(1,ran_len)
    plt.subplot(4,4,i+1)
    plt.axis("off")
    plt.imshow(test_x[num])
    str1="标签值"+str(test_y[num])
    plt.tight_layout(rect=[0,10,0,10])
    plt.title(str1,fontsize=14)
plt.suptitle("MNIST测试集样本",fontsize=20,color="red")
plt.tight_layout(rect=[0,10,0,10])
plt.show()
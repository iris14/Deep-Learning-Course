import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

boston_housing=tf.keras.datasets.boston_housing
(train_x,train_y),(_,_)=boston_housing.load_data(test_split=0)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

title=["crim",'zn','indus','chas','nox','rm','age','dis','rad','tax','piratio','b-1000','lstat','medv']

plt.figure(figsize=(12,12))
for i in range(13):
    plt.subplot(4,4,(i+1))
    plt.scatter(train_x[:,i],train_y)
    plt.xlabel(title[i])
    plt.ylabel("Price($1000's)")
    plt.title(str(i+1)+"."+title[i]+"-Price")


print("1--crim\n2--zn\n3--indus\n4--chas\n5--nox\n6--rm")
print("7--age\n8--dis\n9--rad\n10--tax\n11--piratio\n12--b-1000\n13--lstat")
print("请选择属性：")
n=int(input())
plt.tight_layout()
plt.suptitle("各个属性与房价关系",x=0.5,y=2,fontsize=20)
plt.figure()
plt.scatter(train_x[:,n-1],train_y)
plt.xlabel(title[n-1])
plt.ylabel("Price ($1000's)")
plt.suptitle(title[n-1])
plt.show()





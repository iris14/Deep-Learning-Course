import tensorflow as tf
import numpy as np

x1=tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
x2=tf.constant([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2],dtype=tf.float32)
y=tf.constant([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
x0=tf.ones(len(x1))
X=tf.stack((x0,x1,x2),axis=1)
Y=tf.reshape(y,(-1,1))
xT=tf.transpose(X)
xT_x_inv=tf.linalg.inv(tf.matmul(xT,X))
xT_x_inv_xT=tf.matmul(xT_x_inv,xT)
w=tf.matmul(xT_x_inv_xT,Y)
w=tf.reshape(w,(1,-1))

print("请输入房屋面积和房间数，预测房屋销售价格：")
print("面积：20-500之间的实数\n")
print("房间数：1-10之间的整数\n")
conti=True
while conti:
    try:
        area=float(input("商品屋子面积:"))
        num=int(input("房间数："))
        if (area>=20 and area<=500)and (num>0 and num<11):
            y_pre=w[0][1]*area+w[0][2]*num+w[0][0]
            y_pre=y_pre.numpy()
            print("预测价格为：",round(y_pre,2),"万元")
            conti=False
        else:
            print("输入数据范围不合理，请重新输入")    
    except IOError as e:
        print(e)
        print("数字类型输入错误，请重新输入")
    except TypeError as e:
        print(e)
        print("数字类型输入错误，请重新输入")
    except Exception as e:
        print(e)
        print("数字类型输入错误，请重新输入")
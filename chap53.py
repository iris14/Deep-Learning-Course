import numpy as np
x0=np.ones((10,))
x1=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
x2=np.array([2,3,4,2,3,4,2,4,1,3])
y=np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
Y=y.reshape(10,1)
x3=np.stack((x0,x1,x2),axis=1)
tx3=np.transpose(x3)
x4=np.matmul(tx3,x3)
invx4=np.linalg.inv(x4)
w=np.matmul(invx4,tx3)
W=np.matmul(w,Y)
print("W的shape结果为",end=":")
print(W.shape)
print("\n X的结果为")
print(x3)
print("\n Y的结果为")
print(Y)
print("\n W的结果为")
print(W)
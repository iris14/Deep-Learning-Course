import numpy as np

x=np.array([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03])
y=np.array([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84])
mean_x=x.sum()/x.size
mean_y=y.sum()/y.size
w=sum((x-mean_x)*(y-mean_y))/sum((x-mean_x)*(x-mean_x))
b=mean_y-w*mean_x
print("w结果为：%f,b的结果为：%f" %(w,b))
# print()


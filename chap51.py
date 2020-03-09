import numpy as np
n=int(input("please input a number:"))
np.random.seed(612)
a=np.random.rand(1,1000)
num=0
for i in range(1,1001):
    if(i%n==0):
        num+=1
        print("%d. %d %f"  %(num,i,a[0][i-1]))
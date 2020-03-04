import math as mh

a=int(input("请输入a的值:"))
b=int(input("请输入b的值:"))
c=int(input("请输入c的值:"))

deta=mh.pow(b,2)-4*a*c
if deta>0:
    x1=(-b+mh.sqrt(deta))/(2*a)
    x2=(-b-mh.sqrt(deta))/(2*a)
    print("此方程有两个解：%d,%d" %(x1,x2))
elif deta==0:
    x1=(-b+mh.sqrt(deta))/(2*a)
    print("此方程有一个解：%d" %(x1))
else:
    print("此方程无解")

import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
plt.rcParams['font.sans-serif']="SimHei"
image=Image.open("lena.tiff")
image_red,image_green,image_blue=image.split()
grey_red=image_red.convert("L")
grey_green=image_green.convert("L")
grey_blue=image_blue.convert("L")
re_red=grey_red.resize((50,50))    #缩放

tran_g=grey_green.transpose(Image.FLIP_LEFT_RIGHT) #旋转+镜像
tran_g=tran_g.transpose(Image.ROTATE_270)

crop_blue=grey_blue.crop((0,0,150,150))#裁剪
image_merge=Image.merge("RGB",[image_red,image_green,image_blue])   #图像合并

plt.figure(4)
plt.subplot(221)
plt.axis("off")     #不显示坐标
plt.title("R-缩放",fontsize=14)
plt.imshow(re_red,cmap="gray")
plt.subplot(222)
plt.title("G-镜像+旋转",fontsize=14)
plt.imshow(tran_g,cmap="gray")
plt.subplot(223)
plt.axis("off")
plt.title("B-裁1剪",fontsize=14)
plt.imshow(crop_blue,cmap="gray")
plt.subplot(224)
plt.axis("off")
plt.title(image_merge.mode,fontsize=14)
plt.imshow(image_merge)
plt.tight_layout()
plt.suptitle("图像基本操作",fontsize=20,color="blue")
plt.show()

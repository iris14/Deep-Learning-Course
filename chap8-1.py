import tensorflow as tf
x=tf.constant([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03],dtype=tf.float32)
y=tf.constant([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84],dtype=tf.float32)
mean_x=tf.reduce_mean(x)
mean_y=tf.reduce_mean(y)
mul=(x-mean_x)*(y-mean_y)
up=tf.reduce_sum(mul)
down=tf.reduce_sum(tf.square(x-mean_x))
w=up/down
b=mean_y-w*mean_x
print(w)
print(b)
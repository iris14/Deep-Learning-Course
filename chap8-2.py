import tensorflow as tf
x=tf.constant([64.3,99.6,145.45,63.75,135.46,92.85,86.97,144.76,59.3,116.03],dtype=tf.float32)
y=tf.constant([62.55,82.42,132.62,73.31,131.05,86.57,85.49,127.44,55.25,104.84],dtype=tf.float32)
mul=x*y
n=10
sum_xy=n*tf.reduce_sum(x*y)
sum_x=tf.reduce_sum(x)
sum_y=tf.reduce_sum(y)
square_x=tf.reduce_sum(tf.square(x))*n
x_square=tf.square(tf.reduce_sum(x))
up=sum_xy-sum_x*sum_y
down=square_x-x_square
w=up/down
b=(sum_y-w*sum_x)/n
print(w)
print(b)
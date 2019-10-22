#常量constant   在运行过程中不会改变的单元，在TensorFlow中无需进行初始化操作

import tensorflow as tf

a = tf.constant(1.0,name = "a")
b = tf.constant(2.5,name= "b")
c = tf.add(a,b,name = "c")

sess = tf.Session()
c_value = sess.run(c)
print(c_value)
sess.close()


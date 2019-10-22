#常量constant   在运行过程中不会改变的单元，在TensorFlow中无需进行初始化操作

import tensorflow as tf

a = tf.constant(1.0,name = "a")
b = tf.constant(2.5,name= "b")
c = tf.add(a,b,name = "c")

sess = tf.Session()
c_value = sess.run(c)
print(c_value)
sess.close()



#变量  Variable
#在运行过程中会该改变的单元，在TensorFlow中须进行初始化操作

'''
个别变量初始化：
            init_op = name_variable.initializer()
所有变量初始化：
            init_op = tf.global_variable_initializer()
            '''
node1 = tf.Variable(3.0,tf.float32,name = "node1")
node2 = tf.Variable(4.0,tf.float32,name = "node2")
result = tf.add(node1,node2,name = "add")

sess = tf.Session()

#变量初始化
init = tf.global_variables_initializer()
#init只是一个操作，也需要先运行才能实现（即所有操作只有运行后才能实现）
sess.run(init)

print(sess.run(result))

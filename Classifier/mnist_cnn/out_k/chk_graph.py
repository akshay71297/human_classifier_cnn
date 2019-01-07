import tensorflow as tf 

g = tf.GraphDef()

g.ParseFromString(open("opt_mnist_convnet.pb", "rb").read())

for n in g.node:
	print(n.name)
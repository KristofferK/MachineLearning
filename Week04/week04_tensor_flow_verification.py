import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
print(tf.Session().run(hello))
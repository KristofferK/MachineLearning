import tensorflow as tf
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)

print ("Tensorflow version "+str(tf.__version__))

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

init = tf.global_variables_initializer()
saver = tf.train.Saver()

sess = tf.InteractiveSession()
init.run()
result = f.eval()
save_path = saver.save(sess, "checkpoints/my_model.ckpt")
print(result) # 42
print(save_path)
sess.close()


##

w = tf.constant(3)
x2 = w + 2
y2 = x2 + 5
z2 = x2 * 3
with tf.Session() as sess:
    print(y2.eval()) # 10
    print(z2.eval()) # 15
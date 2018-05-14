import tensorflow as tf

A = tf.Variable(tf.ones(shape=(10, 5)), dtype=tf.float32)
B = A - 2.0
C = tf.subtract(2.0, A)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(A), '\n')
    print(sess.run(B), '\n')
    print(sess.run(C))
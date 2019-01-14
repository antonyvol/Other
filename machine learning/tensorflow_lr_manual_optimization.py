### same as tensorflow_linear_regression, but SGD done manually via tf.gradients 

import numpy as np
import tensorflow as tf

n_samples = 1000
batch_size = 100
num_steps = 20000
learning_rate = 0.00001 

# pregen data with noise
X_data = np.random.uniform(1, 10, (n_samples, 1))
y_data = 2 * X_data + 1	+ np.random.normal(0, 2, (n_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope('linear-regression'):
	k = tf.Variable(tf.random_normal((1, 1)), name='slope')
	b = tf.Variable(tf.zeros((1, )), name='bias')

y_pred = tf.matmul(X, k) + b # estimated y values
loss = tf.reduce_sum((y - y_pred) ** 2) # error on the batch

# get the gradients of loss with respect to k and b
grad_k, grad_b = tf.gradients(xs=[k, b], ys=loss)

# update the variables
new_k = k.assign(k - learning_rate * grad_k)
new_b = b.assign(b - learning_rate * grad_b)

display_step = 100
init = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)
	for i in range(num_steps):
		indices = np.random.choice(n_samples, batch_size)
		X_batch = X_data[indices]
		y_batch = y_data[indices]
		loss_val, k_val, b_val = session.run([loss, new_k, new_b], feed_dict={X: X_batch, y: y_batch})
		if (i + 1) % display_step == 0:
			print('Epoch %d: %.8f, k=%.4f, b=%.4f' % (i + 1, loss_val, k_val, b_val))


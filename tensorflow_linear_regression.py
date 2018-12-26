import numpy as np
import tensorflow as tf

n_samples = 1000
batch_size = 100
num_steps = 20000
learning_rate = 0.00001 # set empiricaly, maybe a better solution?

# pregen data with noise
X_data = np.random.uniform(1, 10, (n_samples, 1))
y_data = 2 * X_data + 1	+ np.random.normal(0, 2, (n_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope('linear-regression'):
	k = tf.Variable(tf.random_normal((1,1)), name='slope')
	b = tf.Variable(tf.zeros((1,)), name='bias')

y_pred = tf.matmul(X, k) + b # estimated y values
loss = tf.reduce_sum((y - y_pred)**2) # error on the batch
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

display_step = 100
init = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)
	for i in range(num_steps):
		indices = np.random.choice(n_samples, batch_size)
		X_batch = X_data[indices]
		y_batch = y_data[indices]
		_, loss_val, k_val, b_val = session.run([optimizer, loss, k, b], feed_dict={X: X_batch, y: y_batch})
		if (i + 1) % display_step == 0:
			print('Epoch %d: %.8f, k=%.4f, b=%.4f' % (i + 1, loss_val, k_val, b_val))


# to generate the graph diagram run the following command on the terminal
#         tensorboard --logdir /tmp/mnist/demo/1

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
n_classes = 10
batch_size = 100

# create a summary writer

def dense_layer(in_tensor, input_shape, output_shape, nodes, name = 'dense'):
	with tf.name_scope(name):
		weights = tf.Variable(tf.random_normal([input_shape, output_shape]), name = "W")
		biases = tf.Variable(tf.random_normal([nodes]), name = 'b')

		layer_op = tf.add(tf.matmul(in_tensor, weights), biases)
		# layer_op = tf.nn.relu(layer_op)
		tf.summary.histogram('weights', weights)
		tf.summary.histogram('biases', biases)

		return layer_op


x = tf.placeholder('float', [None, 784], name = 'X')
y = tf.placeholder('float', name = 'labels')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Define the layers
dense1 = dense_layer(x, 784, 500, 500, 'dense1')
dense2 = dense_layer(tf.nn.relu(dense1), 500, 500, 500, 'dense2')
dense3 = dense_layer(tf.nn.relu(dense2), 500, 500, 500,'dense3')
prediction = dense_layer(tf.nn.relu(dense3), 500, 10, 10, 'final_dense')

with tf.name_scope('xent'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.name_scope('accuracy'):
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

tf.summary.scalar('cross_entropy', cost)
tf.summary.scalar('accuracy', accuracy)
tf.summary.image('images', x_image, 3)

merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter("/tmp/mnist/demo/1")

epochs = 5
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for batch in range(int(mnist.train.num_examples/batch_size)):
			this_batch = mnist.train.next_batch(batch_size)		
			sess.run(optimizer, feed_dict = {x:this_batch[0], y: this_batch[1]})
			if batch%5 ==0:
				s = sess.run(merged_summary, feed_dict = {x: this_batch[0], y: this_batch[1]})
				writer.add_summary(s, batch)
		print("Epoch", epoch , 'completed out of', epochs)
	writer.add_graph(sess.graph)		
	print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}) )


# to generate the graph diagram run the following command on the terminal
#         tensorboard --logdir /tmp/mnist/demo/1

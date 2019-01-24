# to generate the graph diagram run the following command on the terminal
#         tensorboard --logdir /tmp/mnist/demo/1

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_hidden_l1 = 500
n_hidden_l2 = 500
n_hidden_l3 = 500

n_classes = 10

batch_size = 100

# create a summary writer
writer = tf.summary.FileWriter("/tmp/mnist/demo/1")



def dense_layer(in_tensor, input_shape, output_shape, nodes, name = 'dense'):
	with tf.name_scope(name):
		weights = tf.Variable(tf.random_normal([input_shape, output_shape]), name = "W")
		biases = tf.Variable(tf.random_normal([nodes]), name = 'b')

		layer_op = tf.add(tf.matmul(in_tensor, weights), biases)
		layer_op = tf.nn.relu(layer_op)

		return layer_op


def nn_model(data):
	with tf.name_scope('dense_1'):
		hidden_1_layer = {'weights': tf.Variable(  tf.random_normal([784, n_hidden_l1]), name = 'W'  ),
							'biases': tf.Variable(  tf.random_normal([n_hidden_l1]), name = 'b'  )}
	with tf.name_scope('dense_2'):							
		hidden_2_layer = {'weights': tf.Variable(  tf.random_normal([n_hidden_l1, n_hidden_l2]), name = 'W'  ),
							'biases': tf.Variable(tf.random_normal([n_hidden_l2]), name = 'b')}
	with tf.name_scope('dense_3'):								
		hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_l2, n_hidden_l3]), name = 'W'),
							'biases': tf.Variable(tf.random_normal([n_hidden_l3]), name = 'b')}

	with tf.name_scope('output_layer'):
		output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_l3, n_classes]), name = 'W'),
						'biases': tf.Variable(tf.random_normal([n_classes]), name = 'b')}

	layer_1_op = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'], name = 'layer1_op')
	layer_1_op = tf.nn.relu(layer_1_op)

	layer_2_op = tf.add(tf.matmul(layer_1_op, hidden_2_layer['weights']), hidden_2_layer['biases'], name = 'layer2_op')
	layer_2_op = tf.nn.relu(layer_2_op)

	layer_3_op = tf.add(tf.matmul(layer_2_op, hidden_3_layer['weights']), hidden_3_layer['biases'], name = 'layer3_op')
	layer_3_op = tf.nn.relu(layer_3_op)

	output = tf.add(tf.matmul(layer_3_op, output_layer['weights']), output_layer['biases'], name = 'labels')

	return output

def train_nn(x):
	prediction = nn_model(x)
	with tf.name_scope('xent'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
	
	with tf.name_scope('train'):		
		optimizer = tf.train.AdamOptimizer().minimize(cost)

	epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(epochs):
			epoch_loss = 0

			for batch in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
				epoch_loss += c
			print("Epoch", epoch , 'completed out of', epochs, 'loss: ' , epoch_loss)

		with tf.name_scope('accuracy'):
			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))		
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		
		writer.add_graph(sess.graph)
		print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}) )

# comment out following line to get previous approach of training
# train_nn(x)


# comment out this code block to implement other approach of training

x = tf.placeholder('float', [None, 784], name = 'X')
y = tf.placeholder('float', name = 'labels')

# Define the layers
dense1 = dense_layer(x, 784, 500, 500, 'dense1')
dense2 = dense_layer(dense1, 500, 500, 500, 'dense2')
dense3 = dense_layer(dense2, 500, 500, 500,'dense3')
prediction = dense_layer(dense3, 500, 10, 10, 'final_dense')

with tf.name_scope('xent'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.name_scope('accuracy'):
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

epochs = 10
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for batch in range(int(mnist.train.num_examples/batch_size)):
			this_batch = mnist.train.next_batch(batch_size)		
			sess.run(optimizer, feed_dict = {x:this_batch[0], y: this_batch[1]})
		print("Epoch", epoch , 'completed out of', epochs)
	writer.add_graph(sess.graph)		
	print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}) )


# to generate the graph diagram run the following command on the terminal
#         tensorboard --logdir /tmp/mnist/demo/1

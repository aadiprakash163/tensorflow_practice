import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_hidden_l1 = 500
n_hidden_l2 = 500
n_hidden_l3 = 500

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

n_classes = 10

batch_size = 100

def nn_model(data):
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_hidden_l1])),
						'biases': tf.Variable(tf.random_normal([n_hidden_l1]))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_l1, n_hidden_l2])),
						'biases': tf.Variable(tf.random_normal([n_hidden_l2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_l2, n_hidden_l3])),
						'biases': tf.Variable(tf.random_normal([n_hidden_l3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_l3, n_classes])),
					'biases': tf.Variable(tf.random_normal([n_classes]))}

	layer_1_op = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	layer_1_op = tf.nn.relu(layer_1_op)

	layer_2_op = tf.add(tf.matmul(layer_1_op, hidden_2_layer['weights']), hidden_2_layer['biases'])
	layer_2_op = tf.nn.relu(layer_2_op)

	layer_3_op = tf.add(tf.matmul(layer_2_op, hidden_3_layer['weights']), hidden_3_layer['biases'])
	layer_3_op = tf.nn.relu(layer_3_op)

	output = tf.add(tf.matmul(layer_3_op, output_layer['weights']), output_layer['biases'])

	return output

def train_nn(x):
	prediction = nn_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
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

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}) )



train_nn(x)

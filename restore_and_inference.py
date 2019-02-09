
# Reference following websites in case of any doubt
# https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("./summaries/data/", one_hot = True)
image = mnist.test.images[0]


image = np.array(image)
image = image.reshape(28, 28)
plt.imshow(image, cmap = 'gray')
plt.show()

image = image.reshape(1, -1)

with tf.Session() as sess:
	loader = tf.train.import_meta_graph('./trained_model/mnist_classifier-1000.meta')
	loader.restore(sess, tf.train.latest_checkpoint('./trained_model/'))
	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name('X:0')
	predictions = graph.get_tensor_by_name('final_dense/predictions:0')

	print('Classifier classified this as: ', sess.run(tf.argmax(predictions, 1), feed_dict = {x : image}))

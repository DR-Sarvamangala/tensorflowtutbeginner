import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hl1 = 1000
hl2 = 1000
hl3 = 1000

outl = 10
batch = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def nnmodel(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, hl1])),
                      'biases':tf.Variable(tf.random_normal([hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([hl1, hl2])),
                      'biases':tf.Variable(tf.random_normal([hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([hl2, hl3])),
                      'biases':tf.Variable(tf.random_normal([hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([hl3, outl])),
                    'biases':tf.Variable(tf.random_normal([outl])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train(x):
    prediction = nnmodel(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    epochs = 15
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        print('Total Epochs:', epochs)

        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch)):
                epoch_x, epoch_y = mnist.train.next_batch(batch)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train(x)
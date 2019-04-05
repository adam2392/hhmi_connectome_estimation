import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def createNetwork(activationInput, y, N, T):
    conv1 = tf.layers.conv2d(inputs = activationInput,
                             filters = 32,
                             kernel_size = [2,1],
                             padding = "same",
                             activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(inputs = conv1,
                             filters = 32,
                             kernel_size = [3,1],
                             padding = "same",
                             activation=tf.nn.relu)

    #conv3 = tf.layers.conv2d(inputs = conv2,
    #                         filters = 64,
    #                         kernel_size = [5,1],
    #                         padding = "same",
    #                         activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(inputs = conv2,
                             filters = 5,
                             kernel_size = [5,1],
                             padding = "same",
                             activation=tf.nn.relu)

    conv4_flat = tf.reshape(conv4, (-1, 5 * (N * T)))
    dense1 = tf.layers.dense(inputs=conv4_flat, units= N*N, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=N * N)

    output = tf.reshape(dense2, (-1, N, N,1))

    loss = tf.reduce_mean(tf.norm(output - y, axis=(1,2)))

    return (conv1,
            conv2,
            #conv3,
            conv4,
            dense1,
            dense2,
            output,
            loss,
            y)



def getTrainBatch(activationMatrices, connectivityMatrices):
    batchSize = 20

    trainSize = activationMatrices.shape[0]

    batchInices =  np.random.randint(trainSize, size=batchSize)

    batchActivations = np.reshape(activationMatrices[batchInices,:,:], (-1,) + activationMatrices.shape[1:3] + (1,))
    batchConnectivity = np.reshape(connectivityMatrices[batchInices, :, :], (-1,) + connectivityMatrices.shape[1:3] + (1,))

    return(batchActivations, batchConnectivity)

def main():
    iterations = 10000

    # Should we restore the network from disk
    RESTORE = True

    # Reading the training data files.
    path = '/home/zaslab/PycharmProjects/WorkshopCreateDataset/hhmi_connectome_estimation/TrainData'
    activationMatrices = np.load(os.path.join(path, 'activationMatrices.npy'))
    connectivityMatrices = np.load(os.path.join(path, 'connectivityMatrices.npy'))

    N = connectivityMatrices.shape[1]
    T = activationMatrices.shape[1]

    with tf.Session() as sess:
        currentActivation_ph = tf.placeholder(tf.float64, (None, T, N, 1), name="xPh")
        currentConnectivity_ph = tf.placeholder(tf.float64, (None, N, N, 1), name="yPh")

        (conv1, conv2, conv4, dense1, dense2, output, loss, y) = \
            createNetwork(currentActivation_ph, currentConnectivity_ph, N,T)

        optimizer = tf.train.AdamOptimizer().minimize(loss)

        saver = tf.train.Saver()
        if (RESTORE):
            saver.restore(sess, "./netData/model.ckpt")
        else:
            # Initializing the network with random weights.
            sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            [currentActivations, currentConnectivity] = getTrainBatch(activationMatrices, connectivityMatrices)

            feedDict = {currentActivation_ph : currentActivations, currentConnectivity_ph : currentConnectivity}
            sess.run(optimizer, feed_dict=feedDict )
            currentLoss = \
                loss.eval(feed_dict=feedDict)

            print('Iteration: ' + str(i) + ' Current Loss: ' + str(currentLoss))

            if (i % 20 == 0):
                saver.save(sess, "./netData/model.ckpt")

                # Print the matrices
                print('Actual Matric:')
                yEval = y.eval(feed_dict=feedDict)
                print(np.squeeze(yEval[0,:,:,0]))

                print('Output:')
                outputEval = output.eval(feed_dict=feedDict)
                print(np.squeeze(outputEval[0,:,:,0]))




if __name__ == "__main__":
    main()
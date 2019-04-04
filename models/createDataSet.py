import numpy as np
import matplotlib.pyplot as plt



def ReLU(x):
    return x * (x > 0)

def simulateActivity(connectivityMatrix, simTime):
    dt = 0.1
    N = connectivityMatrix.shape[0]
    T = int(np.ceil(simTime / dt))

    activityMatrix = np.zeros((T, N))

    # Setting initial conditions
    activityMatrix[0, :] = np.random.rand(1, N)


    for i in range(1, T):
        dActivity = ReLU(np.matmul(connectivityMatrix, activityMatrix[i-1, :])) - activityMatrix[i-1, :]
        activityMatrix[i,:] = ReLU(activityMatrix[i-1,:] + dt * dActivity)

        # Adding noise
        noiseStd = np.diag(np.sqrt(activityMatrix[i,:]))
        noiseMean = np.zeros(activityMatrix.shape[1])

        activityMatrix[i,:] = \
            activityMatrix[i,:] + np.random.multivariate_normal(np.zeros((activityMatrix.shape[1], )),
                                                  np.diag(np.squeeze(np.sqrt(activityMatrix[i,:]))),
                                                  (1,))

        activityMatrix[i, :] = ReLU(activityMatrix[i, :])

    return(activityMatrix)

def createConnectivityMatrix(N, pDist):
    return(np.random.choice((0,1,-1), size=(N, N), p=pDist))

def main(neuronsNum, simTime, m, pDist):

    connectivityMatrices = None
    activationMatrices = None

    for i in range(m):
        # First we create the connectivity matrix.
        # Working with a single connectivity matrix
        np.random.seed(17)
        connectivityMatrix = createConnectivityMatrix(neuronsNum, pDist)
        activity = simulateActivity(connectivityMatrix, simTime )

        connectivityMatrix = np.reshape(connectivityMatrix, connectivityMatrix.shape + (1,))
        activity = np.reshape(activity, activity.shape + (1,))

        if (connectivityMatrices is None):
            connectivityMatrices = connectivityMatrix

        else:
            connectivityMatrices = np.concatenate((connectivityMatrices, connectivityMatrix), axis=2)

        if (activationMatrices is None):
            activationMatrices = activity
        else:
            activationMatrices = np.concatenate((activationMatrices, activity), axis=2)

        # Log
        print(str(i))

    np.save('./hhmi_connectome_estimation/TrainData/connectivityMatrices.npy', connectivityMatrices)
    np.save('./hhmi_connectome_estimation/TrainData/activationMatrices.npy', activationMatrices)

if __name__ == '__main__':
    main(100,100, 10,(0.2, 0.4, 0.4))
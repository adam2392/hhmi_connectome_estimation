import numpy as np
import matplotlib.pyplot as plt


def simulateActivity(connectivityMatrix, simTime):
    dt = 0.1
    N = connectivityMatrix.shape[0]
    T = int(np.ceil(simTime / dt))

    activityMatrix = np.zeros((T, N))

    # Setting initial conditions
    activityMatrix[0, :] = np.random.rand(1, N)


    for i in range(1, T):
        dActivity = np.tanh(np.matmul(connectivityMatrix, activityMatrix[i-1, :])) - activityMatrix[i-1, :]
        activityMatrix[i,:] = activityMatrix[i-1,:] + dt * dActivity


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

    np.save('./connectivityMatrices.npy', connectivityMatrices)
    np.save('./activationMatrices.npy', activationMatrices)

if __name__ == '__main__':
    main(3, 100, 10,(0.6, 0.2, 0.2))
import numpy
from loguru import logger
from functions import *

def training(err, maxIterationNumber, xMatrix, yMatrix, alp, p, m, SequenceName):
    logger.info(f"Start training...")
    currentError = 1
    iterationNumber = 1
    weightsMatrix1 = numpy.random.rand(m+p, m)
    weightsMatrix2 = numpy.random.rand(m, 1)
    while  maxIterationNumber > iterationNumber and currentError > err:
        currentError = 0
        for i in range(len(xMatrix)):
            zeros = numpy.zeros((1, len(xMatrix[i])))
            for j in range(len(xMatrix[i])):
                zeros[0][j] = xMatrix[i][j]
            hMatrix = activationFnction(zeros @ weightsMatrix1)
            delt = activationFnction(hMatrix @ weightsMatrix2) - yMatrix[i]
            weightsMatrix1 -= delt * alp * zeros.T @ weightsMatrix2.T * drFnction(zeros @ weightsMatrix1)
            weightsMatrix2 -= delt * alp * hMatrix.T * drFnction(hMatrix @ weightsMatrix2)
            currentError = (delt**2)[0]/2 + currentError
        if (iterationNumber / 10000) % 1 == 0: logger.info(f"{round(100 * iterationNumber / maxIterationNumber, 1)}% of training, current error: {currentError}")
        iterationNumber = 1 + iterationNumber
    logger.success(f"Training finished on {iterationNumber} iteration, final error: {currentError}\n")
    logger.info(f"Save weights to file...")
    saveWeights(f"weights\\weights1_{SequenceName}", weightsMatrix1)
    saveWeights(f"weights\\weights2_{SequenceName}", weightsMatrix2)
    logger.success(f"Weights saved")

def activationFnction(xMatrix):
    for i in range(len(xMatrix[0])):
        xMatrix[0][i] = 1/(1 + numpy.exp(-xMatrix[0][i]))
    return xMatrix


def drFnction(xMatrix):
    for i in range(len(xMatrix[0])):
        xMatrix[0][i] = xMatrix[0][i]*(1-xMatrix[0][i])
    return xMatrix
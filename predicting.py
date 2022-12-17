import numpy
from loguru import logger
from functions import *

def predictSequence(numberOfPrediction, m, xMatrix, yMatrix, SequenceName):
    logger.info(f"Start predicting...")
    weightsMatrix1 = loadWeights(f"weights\weights1_{SequenceName}.npy")
    weightsMatrix2 = loadWeights(f"weights\weights2_{SequenceName}.npy")
    MATRIX = xMatrix[-1, :-m]
    temp = yMatrix[-1].reshape(1)
    finalSequence = []
    for _ in range(numberOfPrediction):
        MATRIX = MATRIX[1:]
        concantenatedMatrix = numpy.concatenate((MATRIX, temp))
        MATRIX = numpy.concatenate((MATRIX, temp))
        concantenatedMatrix = numpy.append(concantenatedMatrix, numpy.array([0] * m))
        hMatrix = concantenatedMatrix @ weightsMatrix1
        temp = hMatrix @ weightsMatrix2
        finalSequence.append((hMatrix @ weightsMatrix2)[0])
    return finalSequence
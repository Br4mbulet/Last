import numpy
import json
from loguru import logger

def loadSeqs():
    with open('sequences.json', 'r', encoding='utf-8') as f:
        seqs = json.load(f)
        logger.success("Sequences loaded...")                                      
    return seqs


def loadWeights(PASS):
    return numpy.load(PASS)


def saveWeights(PASS, weights):
    numpy.save(PASS, weights)


def creatMatrix(sequence: list, m: int, p: int):
    iter = 0
    xMatrix = []
    yMatrix = []
    while (iter + p) < len(sequence):
        array = []
        for j in range(p):
            array.append(sequence[j + iter])
        yMatrix.append(sequence[iter + p])
        xMatrix.append(array)
        iter = iter + 1
    xMatrix = numpy.array(xMatrix)
    zerosMatrix = numpy.zeros((len(xMatrix), m))
    logger.success("Initialized...")
    return numpy.append(xMatrix, zerosMatrix, axis=1), numpy.array(yMatrix)
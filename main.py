from functions import *
from training import *
from predicting import *
from loguru import logger
from sys import stderr
import json


logger.remove()
logger.add(stderr, format="<level>{level: <7}</level> | <white>{message}</white>")

if __name__ == "__main__":
    parameters = {'p': 5, 'm': 2, 'err': 0.0000001, 'n': 1000000, 'alp': 0.000005} 
    numberOfPrediction = 4
    logger.info("Starting loading sequences from file...")
    seqs = loadSeqs()
    currentSequenceName = ""
    while True:
        selectSequence = 0
        sequenceChoice = int(input("\nSelect sequence to work: \n"
                           "1. Fibonachi numbers \n"
                           "2. Factorial \n"
                           "3. Power function\n"
                           "4. Periodical\n"
                           "5. Exit\n"))
        if sequenceChoice == 1:
            currentSequenceName = "fibonachi_numbers"
        elif sequenceChoice == 2:
            currentSequenceName = "factorial"
        elif sequenceChoice == 3:
            currentSequenceName = "power_function"
        elif sequenceChoice == 3:
            currentSequenceName = "periodical"            
        elif sequenceChoice == 5:
            exit()
        else:
            logger.error('Wrong info')

        logger.info("Matrix initialization...")
        xMatrix, yMatrix = creatMatrix(seqs[currentSequenceName], parameters['m'], parameters['p'])

        while selectSequence == 0:
            option = int(input("\nSelect option: \n"
                            "1. Train \n"
                            "2. Predict \n"))
            if option == 1:
                training(parameters['err'], parameters['n'], xMatrix, yMatrix, parameters['alp'], parameters['p'], parameters['m'], currentSequenceName)
            elif option == 2:
                finalSequence = predictSequence(numberOfPrediction, parameters['m'], xMatrix, yMatrix, currentSequenceName)
                logger.success(f"Final sequence: {finalSequence}")
            else:
                logger.error('Wrong option')
            if input("Change sequence y/n: ") == "y": selectSequence = 1
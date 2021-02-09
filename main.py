#! /usr/bin/env python
#-*- coding: utf-8 -*-

########################
# Python 3.8
# Author : Maxence Blanc - https://github.com/maxenceblanc
########################

# IMPORTS
import os
import numpy as np
np.set_printoptions(precision=3, linewidth=43)
import pprint
import random as rd

import matplotlib.pyplot as plt

# CUSTOM IMPORTS
import perceptron


''' TO DO LIST

    
Partie 1 : attentes de l'enseignant dans le rendu : 
    
1 courbe qui montre le suivi de l'apprentissage = 1 courbe sur laquelle on va afficher en fonction du temps l'erreur totale

erreur totale = somme des valeurs absolues à chaque étape de l'erreur pour l'entrée 0 ET l'érreur pour l'entrée 1


prévisions : le réseau arrive a une erreur totale de 0 (en valeur absolue) en 1 à 25 itérations


1 courbe qui correspond au TEST (test signifie que le réseau n'apprends plus) du réseau face aux exemples bruites

'''


''' NOTES
'''

####################################################
###################| CLASSES |######################
####################################################

class Dataset():
    """
    """

    def __init__(self):

        self.data = None


    def __repr__(self):
        chain = "### Dataset ###\n"

        return chain

    def shuffle(self, seed=None):
        rd.Random(seed).shuffle(self.data)

    def pickRandomData(self):
        return rd.choice(self.data)


    def datasetFromFiles(self, path):

        filenames = getFilenames(path)
        data = [fileToArray(path, filename) for filename in filenames]

        self.data = data

    

####################################################
##################| FUNCTIONS |#####################
####################################################

def fileToArray(path, filename):
    """ takes a filename, returns the data in an array + the expected label.
    """

    path = os.path.join(DATASET_PATH, filename)

    with open(path) as data:
        content = list(data.read())

    clean = [char for char in content if char != '\n']
    ans = int(clean[-1])

    converted = [1 if char == '*' else 0 for char in clean]


    return (np.array(converted[:-1]), ans)


def getFilenames(path):
    """
    """

    return [filename for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename))]


def noiseData(ref, percent):

    size   = len(ref)
    copy = np.array(ref)

    for i in range(len(copy)):

        if rd.randint(0, 100) <= percent:
            copy[i] = 1 if copy[i] == 0 else 0

    return copy


def testNoisedData(data, percent, amount):

    error_amount = 0
    for i in range(amount):

        vector = noiseData(data[0], percent)
        evaluation = p.evaluate(vector)
        prediction = p.predict(evaluation, {})


        if prediction != data[1]:
            error_amount += 1

    return error_amount


def generalizationNoise(dataset):

    errorsZero = [] 

    for noise in range(0, 101):
        error = testNoisedData(dataset.data[0], noise, 50)
        errorsZero.append(error)

    errorsUn = [] 

    for noise in range(0, 101):
        error = testNoisedData(dataset.data[1], noise, 50)
        errorsUn.append(error)

    print(errorsZero)
    print(errorsUn)

    
    plt.plot(range(101), errorsZero, "r")
    plt.plot(range(101), errorsUn, "b")
    plt.ylabel('error amount')
    plt.xlabel('noise %')
    plt.show()


def printArrayFormated(array, line_length):
    chaine = ""

    for i, elt in enumerate(array):
        chaine += f"{elt}, "
        if (i+1) % line_length == 0:
            chaine += "\n"

    print(chaine)

####################################################
###################| CONSTANTS |####################
####################################################

DATASET_FOLDER = "datasets"
DATASET_NAME = "zero_un"

DATASET_PATH = os.path.join(DATASET_FOLDER, DATASET_NAME)

####################################################
####################| PROGRAM |#####################
####################################################

if __name__ == '__main__':

    ### Getting the dataset ###

    dataset = Dataset()
    dataset.datasetFromFiles(DATASET_PATH)
    dataset.shuffle(0)
    # pprint.pprint(dataset.data)



    ### Init the perceptron ###

    p = perceptron.Perceptron(48, 0.5, 0.01)
    # pprint.pprint(p)

    pprint.pprint(p)

    # Apprentissage
    count = p.learnErr(0.0000001, dataset.data)
    print(f"appris en {count} tours.")

    pprint.pprint(p)

    pprint.pprint(p.evaluate(dataset.data[1][0]))
    # pprint.pprint(dataset.data[1][0])

    
    # printArrayFormated(noiseData(dataset.data[1][0], 100), 6)
    # pprint.pprint(p.evaluate(noiseData(dataset.data[1][0], 100)))
    # pprint.pprint(testNoisedData(dataset.data[1], 100, 1))



    generalizationNoise(dataset)
    
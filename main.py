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

    def shuffle(self):
        rd.shuffle(self.data)

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
    """ Takes some binary data and noises it, based on a percentage.
    Each binary element has a percent of chance to be swapped.
    """

    size   = len(ref)
    copy = np.array(ref)

    for i in range(len(copy)):

        if rd.randint(0, 100) <= percent:
            copy[i] = 1 if copy[i] == 0 else 0

    return copy


def testNoisedData(perceptron, data, percent: int, amount: int):
    """ Tests a given perceptron on a noised instance a given amount of times
    and returns the amount of times the perceptron was wrong.

    INPUTS:
            A perceptron
            A testing instance with it's true label
            Amount of noise
            Amount of tests

    """

    error_amount = 0
    for i in range(amount):

        vector = noiseData(data[0], percent)
        evaluation = perceptron.evaluate(vector)
        prediction = perceptron.predict(evaluation, {})


        if prediction != data[1]:
            error_amount += 1

    return error_amount


def generalizationNoise(perceptron, dataset):
    """ Tests the perceptron on data noised from 0 to 100%.
    """

    captions = []

    for i, instance in enumerate(dataset.data):
        errors = [] 

        for noise in range(0, 101):
            error = testNoisedData(perceptron, instance, noise, 50)
            errors.append(error)

        plt.plot(range(101), errors)

        captions.append(f"{int(i)}")


    # plt.plot(range(101), errorsZero, "r")
    # plt.plot(range(101), errorsUn, "b")

    caption_tuple = (caption for caption in captions)
    # print(captions)
    # tuple()

    plt.legend(captions,
           loc='upper right')

    plt.ylabel('error amount')
    plt.xlabel('noise %')
    plt.show()


def printArrayFormated(array, line_length):
    chaine = "[\n"

    for i, elt in enumerate(array):
        chaine += f"{elt}, "
        if (i+1) % line_length == 0:
            chaine += "\n"
    
    chaine += "]"

    print(chaine)

####################################################
###################| CONSTANTS |####################
####################################################

DATASET_FOLDER = "datasets"
DATASET_NAME = "zero_nine"

DATASET_PATH = os.path.join(DATASET_FOLDER, DATASET_NAME)


PERCEPTRON_FOLDER = "perceptrons"

RANDOM_SEED = 0

####################################################
####################| PROGRAM |#####################
####################################################

if __name__ == '__main__':

    # Setting the seed for random generators
    rd.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### Preparing the dataset ###

    dataset = Dataset()
    dataset.datasetFromFiles(DATASET_PATH)
    dataset.shuffle()


    # # Q1.

    # ### Init the perceptron ###
    # p1 = perceptron.Perceptron(48, 0.5, 0.01)

    # # Apprentissage
    # count = p1.learnErr(0, dataset.data, True)
    # print(f"appris en {count} tours.")

    # generalizationNoise(p1, dataset)
    

    # # Q2.

    # ### Init the perceptron ###
    # p2 = perceptron.Perceptron(48, 0.5, 0.01)

    # # Apprentissage
    # count = p2.learnErr(0.001, dataset.data, False)
    # print(f"appris en {count} tours.")

    # generalizationNoise(p2, dataset)


    # Q3.
    # 2 approaches:
    # - 1) training each perceptron individually
    # - 2) training the model as a whole, updating the two faulty perceptrons when an error is made

    ### Sorting the dataset for human readability and ploting

    dataset.data.sort(key=lambda instance: instance[1])

    ### Init the perceptrons ###
    
    perceptrons = [perceptron.Perceptron(48, 0.5, 0.01) for i in range(10)]
    
    ### Approach n°1 ###

    for i, perceptron in enumerate(perceptrons):

        print(f"Generating training set for perceptron {i} ...")

        # Tweaking dataset for approach n°1:
        dataset_train = Dataset()
        dataset_train.data = []
        
        for instance in dataset.data:

            dataset_train.data.append((instance[0], 1 if instance[1] == i else 0))


        ### Print to check the generation of training dataset

        # for instance in dataset_train.data:
        #     print()
        #     printArrayFormated(instance[0], 6)
        #     print("->", instance[1])

        
        ### Apprentissage
        count = perceptron.learnErr(0.001, dataset_train.data)
        print(f"Perceptron n°{i}, apprentissage en {count} tours.")

        perceptron.export(PERCEPTRON_FOLDER, f"perceptron{i}.py")

        # generalizationNoise(perceptron, dataset_train)

    for i in range(len(dataset.data)):

        print(f"Test on {i} :")

        perceptrons_ans = [perceptron.evaluate(dataset.data[i][0]) for perceptron in perceptrons]
        print(perceptrons_ans)
        print(np.argmax(perceptrons_ans))

        print()


        

    



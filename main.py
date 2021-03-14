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
import Dataset
import Perceptron
import Network
# import utility


''' TO DO LIST

'''


''' NOTES
'''


####################################################
##################| FUNCTIONS |#####################
####################################################


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


    # Plot options
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


def plotErrors(errors):

    plt.plot(range(len(errors)), errors)

    plt.legend(["errors"],
           loc='upper right')

    plt.ylabel('error amount')
    plt.xlabel('iteration')
    plt.show()


####################################################
###################| CONSTANTS |####################
####################################################

DATASET_FOLDER = "datasets"
# DATASET_NAME = "zero_one"
DATASET_NAME = "zero_nine" # For part 3

DATASET_PATH = os.path.join(DATASET_FOLDER, DATASET_NAME)


PERCEPTRON_FOLDER = "perceptrons"

RANDOM_SEED = None

####################################################
####################| PROGRAM |#####################
####################################################

if __name__ == '__main__':

    # Setting the seed for random generators
    rd.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### Preparing the dataset ###

    dataset = Dataset.Dataset()
    dataset.datasetFromFiles(DATASET_PATH)
    dataset.shuffle()


    # # Q1.

    # ### Init the perceptron ###
    # p1 = Perceptron.Perceptron(48, 0.5, 0.01)

    # # Apprentissage
    # count, errors = p1.learnErr(0, dataset.data, True)
    # print(f"appris en {count} tours.")
    # plotErrors(errors)

    # generalizationNoise(p1, dataset)
    

    # # Q2.

    # ### Init the perceptron ###
    # p2 = Perceptron.Perceptron(48, 0.5, 0.01)

    # # Apprentissage
    # count, errors = p2.learnErr(0.001, dataset.data, False)
    # print(f"appris en {count} tours.")
    # plotErrors(errors)

    # generalizationNoise(p2, dataset)


    # Q3.
    # 2 approaches:
    # - 1) training each perceptron individually
    # - 2) training the model as a whole, updating the two faulty perceptrons when an error is made

    ### Sorting the dataset for human readability and ploting

    dataset.data.sort(key=lambda instance: instance[1])

    ### Init the perceptrons ###

    network_1 = Network.Network([Perceptron.Perceptron(48, 0.5, 0.01) for i in range(10)])
    network_2 = Network.Network([Perceptron.Perceptron(48, 0.5, 0.01) for i in range(10)])
    
    # ### Approach n°1 ###

    # for i, perceptron in enumerate(network_1.perceptrons):

    #     print(f"Generating training set for perceptron {i} ...")

    #     # Tweaking dataset for approach n°1:
    #     dataset_train = Dataset.Dataset()
    #     dataset_train.data = []
        
    #     for instance in dataset.data:

    #         dataset_train.data.append((instance[0], 1 if instance[1] == i else 0))


    #     ### Print to check the generation of training dataset

    #     # for instance in dataset_train.data:
    #     #     print()
    #     #     printArrayFormated(instance[0], 6)
    #     #     print("->", instance[1])

        
    #     ### Apprentissage
    #     count, errors = perceptron.learnErr(0.001, dataset_train.data)
    #     print(f"Perceptron n°{i}, apprentissage en {count} tours.")
    #     plotErrors(errors)

    #     perceptron.export(PERCEPTRON_FOLDER, f"perceptron{i}.py")

    #     generalizationNoise(perceptron, dataset_train)


    # generalizationNoise(network_1, dataset)


    ### Approach n°2 ###

    count, errors = network_2.learnErr(0.001, dataset.data)
    plotErrors(errors)


    print(f"Learned in {count} turns.")

    generalizationNoise(network_2, dataset)



    



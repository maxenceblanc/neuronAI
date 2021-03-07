#! /usr/bin/env python
#-*- coding: utf-8 -*-

########################
# Python 3.8
# Author : Maxence Blanc - https://github.com/maxenceblanc
########################

# IMPORTS
import os

import numpy as np
import random as rd
import functools

# CUSTOM IMPORTS


''' TO DO LIST
'''


''' NOTES
'''

####################################################
###################| CLASSES |######################
####################################################

class Perceptron():
    """

    INPUTS:
            The amount of parameters the perceptron takes.
            The threshold for decision.
            The learning rate.

    """

    def __init__(self, size: int, theta: float, epsilon: float, weight=None):

        self.size = size
        self.theta = theta
        self.epsilon = epsilon
        
        if weight is None:
            self.weight = np.random.rand(size)
            self.normalize_weight()
            # self.weight = np.ones(size)
        else:
            self.weight = np.array(weight)

        self.decision_function = functools.partial(thresholdFunction, threshold=0.5)


    def __repr__(self):

        chain = "### Perceptron ###\n"
        chain += f"size: {self.size}\n"
        chain += f"weight = \n{self.weight}"
        return chain

    def normalize_weight(self):
        self.weight /= self.size

    def evaluate(self, vector):
        res = np.multiply(self.weight, vector)
        res = np.sum(res) - self.theta
        return res

    def predict(self, x, params):
        return self.decision_function(x, **params)

    def error(self, evaluation, trueLabel):
        return trueLabel - evaluation

    def update(self, error, vector):
        
        self.weight = self.weight + self.epsilon * error * vector

    def pickRandomData(self, dataset):
        return rd.choice(dataset.data)

    def learnErr(self, error_max: float, training_set: list, use_decision:bool=False):
        """ Learning function of the perceptron, based on error in the evaluation.
        
        INPUTS:
                The sum of absolute errors (for each instance) that we want to be under
                The instances with their label
                OPTION: if we want to use the decision (values 1 and 0) instead
                    of the precise evaluation.
        """

        totalErr = error_max + 1 # Just some init trickery
        count = 0
        
        while totalErr > error_max:

            count += 1
            totalErr = 0

            for vector, ans in training_set:
                evaluation = self.evaluate(vector)

                if use_decision:
                    evaluation = self.predict(evaluation, {})
                
                error = self.error(evaluation, ans)
                self.update(error, vector)

                totalErr += abs(error)

            # print(totalErr)
        
        return count

    def export(self, path, filename):
        """ Exports the perceptron data as python variables in a custom file.
        """

        filepath = os.path.join(path, filename)

        with open(filepath, "w") as p_file:
            p_file.write(f"weights = {list(self.weight)}\nsize = {self.size}\ntheta = {self.theta}\nepsilon = {self.epsilon}")

####################################################
##################| FUNCTIONS |#####################
####################################################

def thresholdFunction(x, threshold=0, outputs=[0,1]):
    """
    """
    ans = outputs[1] if x > threshold else outputs[0]
    return ans

def identityFunction(x):
    return x

####################################################
###################| CONSTANTS |####################
####################################################


####################################################
####################| PROGRAM |#####################
####################################################

if __name__ == '__main__':

    p = Perceptron(4)

    print(p)

    p.normalize_weight()

    print(p)

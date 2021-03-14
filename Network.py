#! /usr/bin/env python
#-*- coding: utf-8 -*-

########################
# Python 3.7
# Author : Maxence Blanc - https://github.com/maxenceblanc
########################

# IMPORTS
import numpy as np

# CUSTOM IMPORTS


''' TO DO LIST
'''


''' NOTES
'''

####################################################
###################| CLASSES |######################
####################################################

class Network():

    def __init__(self, perceptrons):

        self.perceptrons = perceptrons


    def evaluations(self, vector):
        return [perceptron.evaluate(vector) for perceptron in self.perceptrons]


    def evaluate(self, vector):

        evaluations = self.evaluations(vector)
        ans         = np.argmax(evaluations)

        return ans

    def predict(self, x, params):
        return x

    def learnErr(self, error_max: float, training_set: list):
        """ Learning function of the network, based on error in the evaluation.
        """

        totalErr = error_max + 1 # Just some init trickery
        count = 0

        errors = []
        
        while totalErr > error_max:

            count += 1
            totalErr = 0

            for i in range(len(training_set)):

                vector = training_set[i][0]
                label  = training_set[i][1]

                perceptrons_ans = self.evaluations(vector)
                answer = self.evaluate(vector)

                # Update perceptrons if answer isn't correct
                if answer != label:

                    # Get evaluations
                    evaluation_answer = perceptrons_ans[answer]
                    evaluation_label = perceptrons_ans[label]

                    # Deduce errors
                    error_answer = 0 - evaluation_answer
                    error_label =  1 - evaluation_label

                    # Update perceptrons
                    self.perceptrons[answer].update(error_answer, vector)
                    self.perceptrons[label].update(error_label, vector)

                    totalErr += abs(error_answer) + abs(error_label)

            errors.append(totalErr)

        return count, errors


####################################################
##################| FUNCTIONS |#####################
####################################################



####################################################
###################| CONSTANTS |####################
####################################################


####################################################
####################| PROGRAM |#####################
####################################################

if __name__ == '__main__':

    pass
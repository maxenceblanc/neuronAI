#! /usr/bin/env python
#-*- coding: utf-8 -*-

########################
# Python 3.7
# Author : Maxence Blanc - https://github.com/maxenceblanc
########################

# IMPORTS

# CUSTOM IMPORTS
import utility
import random as rd


''' TO DO LIST
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

        filenames = utility.getFilenames(path)
        data = [utility.fileToArray(path, filename) for filename in filenames]

        self.data = data

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
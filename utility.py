#! /usr/bin/env python
#-*- coding: utf-8 -*-

########################
# Python 3.7
# Author : Maxence Blanc - https://github.com/maxenceblanc
########################

# IMPORTS
import os
import numpy as np

# CUSTOM IMPORTS


''' TO DO LIST
'''


''' NOTES
'''

####################################################
###################| CLASSES |######################
####################################################


####################################################
##################| FUNCTIONS |#####################
####################################################


def fileToArray(path, filename):
    """ takes a filename, returns the data in an array + the expected label.
    """

    path = os.path.join(path, filename)

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


####################################################
###################| CONSTANTS |####################
####################################################


####################################################
####################| PROGRAM |#####################
####################################################

if __name__ == '__main__':

    pass
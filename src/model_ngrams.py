# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:35:40 2018

@author: bnuryyev
"""

"""
Note that n-gram modeling is equivalent to Markov Chains here, where the
n-gram represents the state.
"""

import numpy as np
import pandas as pd

import sys
import os

# Markov chain class
class MarkovChain(object):
    def __init__(self, text, n_grams, min_len):
        self.grams = []
        self.grams_possibilities = {}
        self.grams_count = {}
        self.n_grams = n_grams
        self.text = text.split(" ")
        self.min_len = min_len
        self.nGramsWords()
        self.nGramsCount()
        self.nGramsPossibilities()
        
        
    def nGramsWords(self):
        """ Get the n-grams as list of lists """
        for i in range(len(self.text) - self.n_grams + 1):
            self.grams.append(self.text[i : i + self.n_grams])    


    def nGramsCount(self):
        """ Compute the frequency of each gram """
        for i in range(len(self.text) - self.n_grams + 1):
            # Get the gram first
            gram = self.grams[i]
            
            # Count the frequency using dictionary of lists as keys
            if not repr(gram) in self.grams_count:
                self.grams_count[repr(gram)] = 1
            else:
                self.grams_count[repr(gram)] += 1
                            
    
    def nGramsPossibilities(self):
        """ Get the transition probabilities for each state. 
            Appends the possible words into array inside dictionary.
            The more of the same word X - the more the probability of
            the word X to get generated
        """
        for i in range(len(self.text) - self.n_grams):
            # Get the gram
            gram = self.grams[i]
            
            # Get the next possible word
            next_word = self.text[i + self.n_grams]
            
            # Initialize array if the gram is not there yet
            if not repr(gram) in self.grams_possibilities:
                self.grams_possibilities[repr(gram)] = []
            
            # Otherwise append the word to already existing gram in the dict
            self.grams_possibilities[repr(gram)].append(next_word)            
    
                
    def generateText(self):
        """ Generates the text randomly choosing 
            from the probability dictionary 
        """                
        # Here you have two choices: start from the beginning of a text
        # or somewhere randomly in the text.
        # Currently starts somewhere randomly.
        current_i = np.random.choice(len(self.grams))
        current = self.grams[current_i]
        
        # Append the current gram
        output = " ".join(current)
        
        # Initialize variable to store the next word
        next_word = ""
        
        for i in range(self.min_len):            
            possibilities = self.grams_possibilities[repr(current)]
            
            # If there are no possibilities, skip to the next iteration
            if len(possibilities) == 0:
                continue
            else:
                next_word = np.random.choice(possibilities)
            
            # Append the word to the so far generated text
            output = output + " " + str(next_word)
            
            # Get the next gram
            current_i = current_i + 1
            current = self.grams[current_i]        
        
        return output
    

def main():
    # Load the csv file
    if len(sys.argv) != 2:
        sys.exit("You should only pass 1 arg, which is a csv file")
    if not sys.argv[1].endswith('.csv'):
        sys.exit("Hmm are you sure you passed a csv file?")
    if not os.path.exists(sys.argv[1]):
        sys.exit("Hmm the csv file does not exist, does it?")

    # Load data and convert it to numpy array
    data = pd.read_csv(sys.argv[1], header=None, na_values=['.'], encoding='latin-1')
    data = np.array(data)
    
    # Flatten it so we got array of strings
    data = data.flatten()
    
    # Concatenate the articles
    data = ' '.join(data)
    
    # Set the grams and designated text size
    n_grams = 5
    text_size = 200
    
    # Instantiate markov chain and generate text
    markov = MarkovChain(data, n_grams, text_size)
    output = markov.generateText()
    print(output)

# Ignite the machine
main()

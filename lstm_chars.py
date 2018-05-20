# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:42:21 2018

@author: bnuryyev
"""

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# Load the news articles
raw_text = pd.read_csv('news_cleaned.csv', header=None, na_values=['.'], encoding='latin-1')
raw_text = np.array(raw_text)
raw_text = raw_text.flatten()

# Create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Get the text stats (num of characters, num of vocabulary)
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# Prepare the dataset of input to output pairs encoded as integers
seq_length = 50
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

# Get the number of patterns
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# Reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# Normalize
X = X / float(n_vocab)

# One hot encode the output variable
y = np_utils.to_categorical(dataY)

# Define the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]

# Write the generated text to file
text_file = open("generated_text_rnn_chars.txt", "w+")

# Generate characters
for i in range(1):
	# Reshape the pattern and normalize
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)

	# Predict the next character
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)

	# Convert character to text
	result = int_to_char[index]

	# Write result to file
	text_file.write(result + " ")

	# Append to pattern
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

# Close the text file and we are done!
text_file.close()
print("\nDone.")

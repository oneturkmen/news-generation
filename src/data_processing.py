"""
Author: Batyr Nuryyev
Date: April 28, 2018

"""

import re
import sys
import numpy as np
import pandas as pd
import os


def cleanData(data_raw):
    # Our "unwanted" characters including stopwords
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    
    # Container for cleaned data
    data_clean = []
    
    for article in data_raw:
        # Remove unwanted weird characters
        article[0] = article[0].lower()
        article[0] = re.sub('\x9d', '', article[0])
        article[0] = re.sub(REPLACE_BY_SPACE_RE, ' ', article[0])
        article[0] = re.sub(BAD_SYMBOLS_RE, '', article[0])        
        
        # Remove if the length of article is less than 100 characters
        if (len(article[0]) >= 150):
            data_clean.append(article[0])

    # returns numpy array
    return np.expand_dims(np.array(data_clean), axis=1)


def main():
    # Read args: <input.csv> <output.csv>
    if len(sys.argv) != 3:
        sys.exit("Lacking or too many arguments! Expected only 2: <input> <output>.")

    # Check if the input file exists
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    if not os.path.exists(inputFile):
        sys.exit("{0} does not exist!!!".format(inputFile))

    if os.path.exists(outputFile):
        print("{0} will be overwritten!".format(outputFile))


    # Load data 
    data = pd.read_csv(inputFile, header=None, na_values=['.'], encoding='latin-1')
    data_np = np.array(data)
    
    # Clean data
    data_cleaned = cleanData(data_np)
    
    # Save cleaned tweets
    pd.DataFrame(data_cleaned).to_csv(outputFile, index=False, header=False)
    
    

if __name__ == '__main__':
    main()

"""
Author: Batyr Nuryyev
Date: April 28, 2018

"""

import re
import numpy as np
import pandas as pd


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
    # Load data 
    data = pd.read_csv('articles1.csv', header=None, na_values=['.'], encoding='latin-1')
    data_np = np.array(data)
    
    # Clean data
    data_cleaned = cleanData(data_np)
    
    # Save cleaned tweets
    pd.DataFrame(data_cleaned).to_csv('news_cleaned.csv', index=False, header=False)
    
    

if __name__ == '__main__':
    main()
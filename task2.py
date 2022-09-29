# --------------------------------------------------------------------------------------------------------------
# Task 2: Inverted Index
# --------------------------------------------------------------------------------------------------------------

from nltk.stem import WordNetLemmatizer
import contractions
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

df = pd.read_csv('candidate-passages-top1000.tsv', delimiter='\t', header=None, names=['qid','pid','query','passage'])

passages = df[['pid', 'passage']].copy()
passages = passages.drop_duplicates()
passages = passages.reset_index(drop=True)

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

def normalise(text):
    '''
    Function that normalises text and returns tokens.
    Input: text --> text string we want to tokenise
    Output: tokens --> list of tokens taken from the text string
    '''

    text = text.lower() # convert all to lower case
    text = contractions.fix(text) # expand contractions
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) # remove punctuation
    tokens = text.split() # tokenisation 
    filtered_tokens = [w for w in tokens if not w in stop_words] # remove stop words
    filtered_tokens = list(map(lemmatizer.lemmatize, filtered_tokens)) # lemmatization of nouns

    return filtered_tokens

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

inv_index = {}

def inverted_index_func(row):
    pid = row['pid']
    passage = row['passage']
    check = normalise(passage)
    unique_words = list(set(check))

    for item in unique_words:
        if item not in inv_index:
            inv_index[item] = {}
        if item in inv_index:
            inv_index[item][pid] = check.count(item) # Frequency of the word (not normalised)

_ = passages.apply(lambda row: inverted_index_func(row), axis=1)

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

with open('inverted_index.pkl', 'wb') as f:
    pickle.dump(inv_index, f)

# --------------------------------------------------------------------------------------------------------------
# End of Task 2
# --------------------------------------------------------------------------------------------------------------
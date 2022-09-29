# ------------------------------------------------------------------------------------------------------------------
# Task 4: Query Likelihood Language Models
# ------------------------------------------------------------------------------------------------------------------

from nltk.stem import WordNetLemmatizer
import contractions
import string
import numpy as np
import pickle
import pandas as pd
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

cp = pd.read_csv('candidate-passages-top1000.tsv', delimiter='\t', header=None, names=['qid','pid','query','passage'])
tq = pd.read_csv('test-queries.tsv', delimiter='\t', header=None, names=['qid','query'])

passages = cp[['pid', 'passage']].copy()
passages = passages.drop_duplicates()
passages = passages.reset_index(drop=True)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

with open('inverted_index.pkl', 'rb') as f:
    inv_index = pickle.load(f)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# Size of our vocabulary
V = len(inv_index)

# Calculate length of passages
len_passages = {}

def len_passages_func(row):
    pid = row['pid']
    passage = row['passage']
    check = normalise(passage)
    len_passages[pid] = len(check)

_ = passages.apply(lambda row: len_passages_func(row), axis=1)

# For simplicity, and keeping things separate, we compute the frequencies of the query terms separately
qf_dict = {i:{} for i in tq['qid']}

def qf_func(row):
    qid = row['qid']
    query = row['query']
    check = normalise(query)
    unique_words = list(set(check))

    for item in unique_words:
        if item in inv_index.keys():
            qf_dict[qid][item] = check.count(item) # non-normalised frequency

_ = tq.apply(lambda row: qf_func(row), axis=1)

# ------------------------------------------------------------------------------------------------------------------
# Laplace Smoothing
# ------------------------------------------------------------------------------------------------------------------

# Laplace Smoothing
laplace_scores = np.array([[0,0,0]])

for k in range(len(tq)):
    scores = []
    qid = tq['qid'][k]
    query = qf_dict[qid]
    query_words = list(query.keys())

    for pid in cp.loc[cp['qid'] == tq['qid'][k]]['pid']:

        dl = len_passages[pid]

        lap_est = np.zeros(len(query_words))

        for i in range(len(query_words)):
            if pid in inv_index[query_words[i]]:
                m = inv_index[query_words[i]][pid]
            else:
                m = 0
            
            lap_est[i] = (m + 1) / (dl + V)

        score = np.prod(lap_est)
        scores.append([qid,pid,np.log(score)])
    
    scores = np.array(scores, dtype="O")
    scores = scores[np.argsort(-scores[:,-1])] # sort in descending order
    
    laplace_scores = np.append(laplace_scores, scores[:100,:], axis=0)

laplace_scores = laplace_scores[1:,:] # remove the [0,0,0] row we used to initialise

# Store results in .csv file
pd.DataFrame(laplace_scores).to_csv("laplace.csv", header=None, index=None)

# ------------------------------------------------------------------------------------------------------------------
# Lidstone Correction
# ------------------------------------------------------------------------------------------------------------------

# Lidstone correction
eps = 0.1
lidstone_scores = np.array([[0,0,0]])

for k in range(len(tq)):
    scores = []
    qid = tq['qid'][k]
    query = qf_dict[qid]
    query_words = list(query.keys())

    for pid in cp.loc[cp['qid'] == tq['qid'][k]]['pid']:

        dl = len_passages[pid]

        lid_est = np.zeros(len(query_words))

        for i in range(len(query_words)):
            if pid in inv_index[query_words[i]]:
                m = inv_index[query_words[i]][pid]
            else:
                m = 0
            
            lid_est[i] = (m + eps) / (dl + eps*V)

        score = np.prod(lid_est)
        scores.append([qid,pid,np.log(score)])
    
    scores = np.array(scores, dtype="O")
    scores = scores[np.argsort(-scores[:,-1])] # sort in descending order
    
    lidstone_scores = np.append(lidstone_scores, scores[:100,:], axis=0)

lidstone_scores = lidstone_scores[1:,:] # remove the [0,0,0] row we used to initialise

# Store results in .csv file
pd.DataFrame(lidstone_scores).to_csv("lidstone.csv", header=None, index=None)

# ------------------------------------------------------------------------------------------------------------------
# Dirichlet Smoothing
# ------------------------------------------------------------------------------------------------------------------

# For simplicity, let's get the frequency of each term in the inverted index in the whole collection of documents
freq_coll = {}

for key in inv_index.keys():
    freq_coll[key] = sum(inv_index[key].values())

# Dirichlet smoothing
mu = 50
dirichlet_scores = np.array([[0,0,0]])
cl = sum(freq_coll.values())

for k in range(len(tq)):
    scores = []
    qid = tq['qid'][k]
    query = qf_dict[qid]
    query_words = list(query.keys())

    for pid in cp.loc[cp['qid'] == tq['qid'][k]]['pid']:

        dl = len_passages[pid]

        dir_est = np.zeros(len(query_words))

        for i in range(len(query_words)):
            if pid in inv_index[query_words[i]]:
                fd = inv_index[query_words[i]][pid]
            else:
                fd = 0
            
            fc = freq_coll[query_words[i]]

            dir_est[i] = (dl/(dl+mu) * fd/dl) + (mu/(dl+mu) * fc/cl)

        score = np.prod(dir_est)
        scores.append([qid,pid,np.log(score)])
    
    scores = np.array(scores, dtype="O")
    scores = scores[np.argsort(-scores[:,-1])] # sort in descending order
    
    dirichlet_scores = np.append(dirichlet_scores, scores[:100,:], axis=0)

dirichlet_scores = dirichlet_scores[1:,:] # remove the [0,0,0] row we used to initialise

# Store results in .csv file
pd.DataFrame(dirichlet_scores).to_csv("dirichlet.csv", header=None, index=None)

# ------------------------------------------------------------------------------------------------------------------
# End of Task 4
# ------------------------------------------------------------------------------------------------------------------

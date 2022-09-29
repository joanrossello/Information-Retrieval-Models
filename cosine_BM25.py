# ------------------------------------------------------------------------------------------------------------------
# Task 3: Retrieval Models
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

with open('inverted_index.pkl', 'rb') as f:
    inv_index = pickle.load(f)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

cp = pd.read_csv('candidate-passages-top1000.tsv', delimiter='\t', header=None, names=['qid','pid','query','passage'])
tq = pd.read_csv('test-queries.tsv', delimiter='\t', header=None, names=['qid','query'])

passages = cp[['pid', 'passage']].copy()
passages = passages.drop_duplicates()
passages = passages.reset_index(drop=True)

# ------------------------------------------------------------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------------------------------------------------------------

N = len(passages)

# TF·IDF passages
p_tfidf = {i:{} for i in passages['pid']}

# Also calculate the length of passages which will be useful in BM25
len_passages = {}

def p_tfidf_func(row):
    pid = row['pid']
    passage = row['passage']
    check = normalise(passage)
    len_passages[pid] = len(check)
    unique_words = list(set(check))

    for item in unique_words:
        tf = inv_index[item][pid] / len(check) # Normalise the frequency from inverted index
        idf = np.log(N/len(inv_index[item]))
        p_tfidf[pid][item] = tf*idf
    p_tfidf[pid]['norm_of_passage'] = np.linalg.norm(np.array(list(map(p_tfidf[pid].get, p_tfidf[pid].keys())))) # add norm of the passage

_ = passages.apply(lambda row: p_tfidf_func(row), axis=1)

# TF·IDF queries
q_tfidf = {i:{} for i in tq['qid']}

def q_tfidf_func(row):
    qid = row['qid']
    query = row['query']
    check = normalise(query)
    unique_words = list(set(check))

    for item in unique_words:
        if item in inv_index.keys():
            tf = check.count(item) / len(check)
            idf = np.log(N/len(inv_index[item]))
            q_tfidf[qid][item] = tf*idf
    q_tfidf[qid]['norm_of_query'] = np.linalg.norm(np.array(list(map(q_tfidf[qid].get, q_tfidf[qid].keys())))) # add norm of the query

_ = tq.apply(lambda row: q_tfidf_func(row), axis=1)

# Define a function to compute the cosine similarity
def cosine_sim(query,passage):
    '''
    This function computes the cosine similarityu between queries and passages.
    Inputs
    query: dictionary with tf·idf vector representation of the query
    passage: dictionary with tf·idf vector representation of the passage
    Note that we only include the matchin words in the vector representation.

    Output
    score: cosine similarity score
    '''

    qk = query.keys()
    pk = passage.keys()

    common = set(qk) & set(pk)

    q = np.array(list(map(query.get, common)))
    p = np.array(list(map(passage.get, common)))

    dot_product = np.dot(q, p)
    norm_q = query['norm_of_query']
    norm_p = passage['norm_of_passage']
    
    score =  dot_product / (norm_q * norm_p)

    return score

# Calculate cosine similarity scores
final_scores = np.array([[0,0,0]])

for k in range(len(tq)):
    scores = []
    qid = tq['qid'][k]

    for pid in cp.loc[cp['qid'] == tq['qid'][k]]['pid']:

        query = q_tfidf[qid]
        passage = p_tfidf[pid]

        score = cosine_sim(query,passage)

        scores.append([qid,pid,score])
    
    scores = np.array(scores, dtype="O")
    scores = scores[np.argsort(-scores[:,-1])] # sort in descending order
    
    final_scores = np.append(final_scores, scores[:100,:], axis=0)

final_scores = final_scores[1:,:] # remove the [0,0,0] row we used to initialise

# Store results in .csv file
pd.DataFrame(final_scores).to_csv("tfidf.csv", header=None, index=None)


# ------------------------------------------------------------------------------------------------------------------
# BM25
# ------------------------------------------------------------------------------------------------------------------

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

# Define a function that calculates the BM25 score
def BM25(n, N, k1, k2, b, dl, avdl, f, qf, r=0, R=0):
    '''
    BM25 score calculating function

    Inputs
    n: (vector of integers) number of total docs. containing each term in the query (each vector element corresponds to a term)
    N: (integer) total number of documents we have
    k1: (scalar) constant parameter set empirically
    k2: (scalar) constant parameter set empirically
    b: (scalar) constant parameter set empirically
    dl: (integer) document length --> number of tokens
    avdl: (scalar) average document length --> average number of tokens in the set of documents
    f: (vector of integers) frequency in the document of each term in the query (each vector element corresponds to a term)
    qf: (vector of integers) frequency in the document of each term in the query (each vector element corresponds to a term)
    r: (vector of integers) number of relevant docs. containing each term in the query (each vector element corresponds to a term)
    R: (integer) total number of relevant documents

    Note that if we do not have information about relevance feedback, r and R are set to 0

    Output
    score: (scalar) BM25 score of a document with respect to a query

    '''

    K = k1 * ((1 - b) + b * (dl/avdl))

    score = np.sum( np.log( ((r+0.5)/(R-r+0.5)) / ((n-r+0.5)/(N-n-R+r*0.5)) ) * (((k1+1)*f)/(K+f)) * (((k2+1)*qf)/(k2+qf)) )

    return score

# Calculate BM25 scores
N = len(passages)
k1 = 1.2
k2 = 100
b = 0.75
avdl = sum(len_passages.values()) / len(len_passages)

BM25_scores = np.array([[0,0,0]])

for k in range(len(tq)):
    scores = []
    qid = tq['qid'][k]
    query = qf_dict[qid]
    query_words = query.keys()

    for pid in cp.loc[cp['qid'] == tq['qid'][k]]['pid']:
        
        passage = p_tfidf[pid]
        passage_words = passage.keys()

        common = list(set(query_words) & set(passage_words))

        dl = len_passages[pid]

        n, f, qf = np.zeros(len(common)), np.zeros(len(common)), np.zeros(len(common))
        for i in range(len(common)):
            n[i] = len(inv_index[common[i]])
            f[i] = inv_index[common[i]][pid]
            qf[i] = qf_dict[qid][common[i]]

        score = BM25(n,N,k1,k2,b,dl,avdl,f,qf)

        scores.append([qid,pid,score])
    
    scores = np.array(scores, dtype="O")
    scores = scores[np.argsort(-scores[:,-1])] # sort in descending order
    
    BM25_scores = np.append(BM25_scores, scores[:100,:], axis=0)

BM25_scores = BM25_scores[1:,:] # remove the [0,0,0] row we used to initialise

# Store results in .csv file
pd.DataFrame(BM25_scores).to_csv("bm25.csv", header=None, index=None)

# ------------------------------------------------------------------------------------------------------------------
# End of Task 3
# ------------------------------------------------------------------------------------------------------------------

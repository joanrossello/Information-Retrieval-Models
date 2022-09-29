# Information-Retrieval-Models

Develop information retrieval models that solve the problem of passage retrieval, i.e. given a query, return a ranked list of short texts (passages).

* Text statistics: extract 1-grams from raw text and perform text pre-processing steps (expand contractions, remove punctuation and stop words, tokenization, lemmatization, etc.). Count word occurrences and analysie Zipf's distribution.
* Generate an inverted index of the unique terms in the corpus and their occurrences in each passage and query.
* Use TF-IDF vector representation of passages, cosine similarity and BM25 score to extract top 100 passages for each query in ranking order. 
* Implement query likelihood language models with Laplace smoothing, Lidstone correction, and Dirichlet smoothing.
* Evaluate retrieval quality by computing average precision and NDCG metrics.
* Create feature representation for each query and passage pair, such as GloVe word embeddings, cosine similarity, sequence length, etc., and implement a logistic regression model to assess relevance of a passage to a given query.
* Use the LambdaMART learning to rank algorithm to learn a model that can re-rank passages.
* Build a neural network based model with Pytorch that can re-rank passages.

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize 
import nltk
import gensim.downloader as api
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

def get_sample_word_vectors(n_words=100, random_state=0):
    # Load pre-trained word-vectors from gensim-data
    fname = "W2V_Dataset/word_vectors.kv"    
    try:
        f = open(fname, 'r')
        f.close()
        word_vectors = KeyedVectors.load(fname, mmap='r')
    except FileNotFoundError:
        word_vectors = api.load("glove-wiki-gigaword-100")        
        word_vectors.save(fname)
    
    # Reads ‘alice.txt’ file 
    sample = open("W2V_Dataset/alice.txt", "r") 
    s = sample.read() 
    
    # Replaces escape character with space 
    f = s.replace("\n", " ") 
    f = s.replace("-", " ") 
    
    # Setup nltk
    nltk.download('popular', quiet=True)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokens = set()

    # iterate through each sentence in the file 
    for i in sent_tokenize(f): 
        temp = [] 
        
        # tokenize the sentence into words 
        for j in word_tokenize(i): 
            temp.append(j.lower()) 
    
        tokens.update(temp)
    
    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords]

    # Get random words
    rand = np.random.RandomState(random_state)
    random_tokens = rand.choice(tokens, int(n_words/10), replace=False)
    random_tokens_vector = []
    for token in random_tokens:
        if token in word_vectors:
            random_tokens_vector.append([token, word_vectors[token]])
            similar_words = word_vectors.most_similar(positive=[token], topn=10)
            for similar_word in similar_words:
                the_word = similar_word[0]
                random_tokens_vector.append([the_word, word_vectors[the_word]])
    
    #random_tokens_vector = {token: word_vectors[token] for token in random_tokens if token in word_vectors}
    #random_tokens_vector = [[k, v] for k, v in random_tokens_vector.items()]

    return random_tokens_vector
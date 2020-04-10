import gzip 
import logging
import re 
import string 
import nltk 
import pandas as pd 
import numpy as np 
import math 
from tqdm import tqdm 
from pprint import pprint

import spacy 
from spacy.matcher import Matcher 
from spacy.tokens import Span 
from spacy import displacy 

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# use logistic --- reduce algorithm calculation, getas much point as possible

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import csv

txt_file = r"test_rf_dataset.txt"
csv_file = r"mycsv.csv"
lines = list(open(txt_file, 'r'))
with open(csv_file, 'w', encoding='utf-8', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['id', 'content','error'])
    for row in lines:
        csvwriter.writerow([row.split(";")[0][1:-1],row.split(";")[1][1:-1]])


# call out html using accession number, and give it a name: html_a
for i in range(6):
    if accession[i]=="0000002488-16-000111":
        html_a=htmlrf[i]        # change name to: html_b, for another accession number
                                # then, be able to compare two "paragraph"
        print(htmlrf[i])
      
      
# remove all the punctuations form html_a
import re
p_text=html_a                  # plain text
p_text=re.sub(r'[^\w\s]','',p_text)

## tokenize words
# word tokenizer breaks paragraph into words
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(p_text)
print(tokenized_word)


# frequency distribution of these words
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)

# frequency distribution plot
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()


# remove stop words such as in, and, the,......
# use "english" stop word list, filter out the stop word list
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words=set(stopwords.words("english"))
### print(stop_words)
filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
### print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)
    

# Lexicon Normalization--remove another type of noise in text
# stemming process-- reduce words to their root word
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokeniz
ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
### print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

## detokenize the cleaned, stop word, stemmed text
from nltk.tokenize.treebank import TreebankWordDetokenizer
clean_text=TreebankWordDetokenizer().detokenize(stemmed_words)
## 'clean_text' is cleaned, stop word, stemmed plain text


# turn word from 'stemmeds_word' to vector
# use small or larger spacy model
# small: faster, but don't ship with word vectors, only include context_sentitive tnesors
#        result won't be as good
# large: slower than small, but better result
#        individual tokens will have vectors assigned
# import spacy
# import en_core_web_sm
# nlp=en_core_web_sm.load()
# tokens = nlp("dog cat banana afskfsd")
# for token in tokens:
#    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
import spacy
import en_core_web_lg
# nlp=en_core_web_lg.load()
# tokens=clean_text.nlp
nlp = spacy.load("en_core_web_lg")
tokens = nlp(clean_text)
for token in tokens:
    print(token.text, token.vector)
# print(token.has_vector) gives TRUE if is a word


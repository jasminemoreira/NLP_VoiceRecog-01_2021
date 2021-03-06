# -*- coding: utf-8 -*-
"""
Criado em Sat Mar 27 06:32:03 2021

@author: Jasmine Moreira
"""

import csv
import PyPDF2
import string
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from nltk import word_tokenize, BigramCollocationFinder 
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from plotnine import ggplot, geom_col, aes, coord_flip
import matplotlib.pyplot as plt

# Carregar documento e stop words
stop_words_path = "C:\\Users\\jasmi\\OneDrive\\Área de Trabalho\\PLN RV\\PLN R e Python\\stopwords.csv"
document_path = "C:\\Users\\jasmi\\OneDrive\\Área de Trabalho\\PLN RV\\PLN R e Python\\senhor_aneis_1.1.pdf"

sw = []
with open(stop_words_path, encoding="utf8") as file:
    reader = csv.reader(file)
    next(reader, None) #skip header
    for row in reader:
        sw.append(row[0].strip())

table = str.maketrans('', '', string.punctuation)        
pages = []
pages_raw = []
reader = PyPDF2.PdfFileReader(document_path)

#Ler páginas do documento
for page in range(0,reader.numPages-1):
    txt = reader.getPage(page).extractText()
    pages_raw.append(txt)
    txt = txt.translate(table)
    pages.append(txt)
  
#Tokenizar
tokens_raw = []
for page in pages:
    tokens_raw.extend(word_tokenize(page))   

tokens = [token.lower() for token in tokens_raw if token.lower() not in sw and token!=""]

#Análise de frequência
tokens_count = Counter(tokens).most_common(10)

#Plotar frequências
df = pd.DataFrame({
    'words': [token[0] for token in tokens_count ],
    'freq': [token[1] for token in tokens_count ]
})

(ggplot(df, aes(x='reorder(words, freq)', y='freq', fill='words'))
 + geom_col()
 + coord_flip()
)

#Análise de Bigramas
finder = BigramCollocationFinder.from_words(tokens)
bigram = []
bfreq = []
for k,v in finder.ngram_fd.items():
    bigram.append(k[0]+" "+k[1])
    bfreq.append(v)

#Análise de Trigramas
from nltk.collocations import TrigramCollocationFinder
trigram = []
tfreq = []  
finder = TrigramCollocationFinder.from_words(tokens)
for k,v in finder.ngram_fd.items():
    trigram.append(k[0]+" "+k[1]+" "+k[2])
    tfreq.append(v)

#Análise de Quadrigramas
from nltk.collocations import QuadgramCollocationFinder    
finder = QuadgramCollocationFinder.from_words(tokens)
quadgram = []
qfreq = []  
for k,v in finder.ngram_fd.items():
    quadgram.append(k[0]+" "+k[1]+" "+k[2]+" "+k[3])
    qfreq.append(v)

#Análise de Tópicos
doc_set = []
for page in pages_raw:
    doc_set.append(page)

tokenizer = RegexpTokenizer(r'\w+')
texts = []
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in sw]  
    texts.append(stopped_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
    
def plot_bar_x(labels,values,number):
    # this is for plotting purpose
    index = np.arange(len(labels))
    plt.bar(index, values)
    plt.xlabel('Word', fontsize=10)
    plt.ylabel('Beta', fontsize=10)
    plt.xticks(index, labels, fontsize=10, rotation=70)
    plt.title('Topic Analysis '+str(number))
    plt.show()

# generate LDA model
ntopics = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=ntopics, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=ntopics, num_words=4))
topics = ldamodel.show_topics( num_words=8,formatted=False)

cnt = 0
for n,t in topics:
    cnt += 1
    words = [i[0] for i in t]
    betas = [i[1] for i in t]
    plot_bar_x(words,betas,cnt)


# -*- coding: utf-8 -*-
"""
Criado em Sat Mar 27 06:32:03 2021

@author: Jasmine Moreira
"""

import csv
import PyPDF2
import string
import pandas as pd
from nltk import word_tokenize, BigramCollocationFinder
from collections import Counter
from plotnine import ggplot, geom_col, aes, coord_flip
from gensim import corpora, models

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
reader = PyPDF2.PdfFileReader(document_path)
for page in range(0,reader.numPages-1):
    txt = reader.getPage(page).extractText()
    txt = txt.translate(table)
    pages.append(txt)
    
tokens_raw = []
for page in pages:
    tokens_raw.extend(word_tokenize(page))   

tokens = [token.lower() for token in tokens_raw if token.lower() not in sw and token!=""]

tokens_count = Counter(tokens).most_common(10)

df = pd.DataFrame({
    'words': [token[0] for token in tokens_count ],
    'freq': [token[1] for token in tokens_count ]
})

(ggplot(df, aes(x='reorder(words, freq)', y='freq', fill='words'))
 + geom_col()
 + coord_flip()
)

finder = BigramCollocationFinder.from_words(tokens)
bigram = []
bfreq = []
for k,v in finder.ngram_fd.items():
    bigram.append(k[0]+" "+k[1])
    bfreq.append(v)

from nltk.collocations import TrigramCollocationFinder
trigram = []
tfreq = []  
finder = TrigramCollocationFinder.from_words(tokens)
for k,v in finder.ngram_fd.items():
    trigram.append(k[0]+" "+k[1]+" "+k[2])
    tfreq.append(v)

from nltk.collocations import QuadgramCollocationFinder    
finder = QuadgramCollocationFinder.from_words(tokens)
quadgram = []
qfreq = []  
for k,v in finder.ngram_fd.items():
    quadgram.append(k[0]+" "+k[1]+" "+k[2]+" "+k[3])
    qfreq.append(v)


dictionary = corpora.Dictionary(tokens)

# convert tokenized documents into a document-term matrix
corpus = dictionary.doc2bow(tokens)

def plot_bar_x(labels,values):
    # this is for plotting purpose
    index = np.arange(len(labels))
    plt.bar(index, values)
    plt.xlabel('Word', fontsize=10)
    plt.ylabel('Beta', fontsize=10)
    plt.xticks(index, labels, fontsize=10, rotation=70)
    plt.title('Topic Analysis')
    plt.show()

# generate LDA model
ntopics = 2
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=ntopics, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=ntopics, num_words=4))
topics = ldamodel.show_topics( num_words=5,formatted=False)

for n,t in topics:
    words = [i[0] for i in t]
    betas = [i[1] for i in t]
    plot_bar_x(words,betas)



# -*- coding: utf-8 -*-
"""
Criado em Sat Mar 27 06:32:03 2021

@author: Jasmine Moreira
"""

import csv
import PyPDF2
import string
import pandas as pd
from nltk import word_tokenize, BigramCollocationFinder, collocations
from collections import Counter
from plotnine import ggplot, geom_col, aes, coord_flip, scales

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

bigram_measures = collocations.BigramAssocMeasures()
trigram_measures = collocations.TrigramAssocMeasures()

finder = BigramCollocationFinder.from_words(tokens)
best10 = finder.nbest(bigram_measures.pmi, 10)
bigram = []
bfreq = []
for k,v in finder.ngram_fd.items():
    
    bigram.append(k[0]+" "+k[1])
    bfreq.append(v)
    
df = pd.DataFrame({
    'words': bigram,
    'freq': bfreq
})

(ggplot(df, aes(x='reorder(words, freq)', y='freq', fill='words'))
 + geom_col()
 + coord_flip()
)



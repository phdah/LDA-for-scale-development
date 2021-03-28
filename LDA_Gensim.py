import pandas as pd
import numpy as np
#import math
#import re
import string

import spacy

# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Gensim pipeline
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

# Stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# libraries for visualization
#import pyLDAvis
#import pyLDAvis.gensim
#from IPython.display import display

import matplotlib.pyplot as plt
#import seaborn as sns
import itertools

# %matplotlib inline


wd = 'C:/Users/name/Desktop/Korning/04 Python Scripts/LDA phyton 2.0/'

text_RawFile = 'All_komments.xlsx'

topic_min = 2
topic_max = 10
chunk = 100

df_org = pd.read_excel(wd + text_RawFile)

# For testing convergence with increasning number of observations
#df_org = df_org.sample(700, random_state=1).reset_index(drop=True)

df = df_org.copy()

print(df.head(2))
print('Number of documents: ' + str(len(df)))


def clean_text(text):
    # List of all special signs
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    # Make string
    delete_dict[' '] = ' '
    # Fill table with dictionary
    table = str.maketrans(delete_dict)
    # Apply the dictionaty to the text cell
    text1 = text.translate(table)
    # print('cleaned:'+text1)
    # Tokenize the words
    textArr = text1.split()
    # Join back tokenz in one string, removing the digits and words shorter than 3
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and (not w.isdigit() and len(w) > 1))])

    # Return, with lower casing
    return text2.lower()


# Drop na rows in text
df.dropna(axis=0, how='any', inplace=True)

# Clean text for column 'Text', with out function
df['Text'] = df['Text'].apply(clean_text)

# Add column of number of words
df['Num_words_text'] = df['Text'].apply(lambda x: len(str(x).split()))


# print(df.head)

# function to remove stopwords
def remove_stopwords(text):
    # Split into tokenz
    textArr = text.split(' ')
    # Text, without stopwords
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text


# remove stopwords from the text
df['Text'] = df['Text'].apply(remove_stopwords)


# print(df.head)

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ']):
    # Remove stemming
    output = []
    for sent in texts:
        # Apply lemma dictionary on each documents
        doc = nlp(sent)
        # Append all allowed tokenz to each document, as tokenz
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return output


# Make list
text_list = df['Text'].tolist()
# print(text_list[0])

# Lemmatization on list, output tokenz for each list item
tokenized_text = lemmatization(text_list)

# print(tokenized_text[0])

# Mapping between token and ID
dictionary = corpora.Dictionary(tokenized_text)

# Create bag of words matrix
doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_text]

# Interoperate the matrix in words, with count of words
id_words = [[(dictionary[id], count) for id, count in line] for line in doc_term_matrix]


# Compute Coherence Score
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
    gensim.models.coherencemodel.CoherenceModel(model=None, topics=None, texts=None, corpus=None, dictionary=None,
                                        window_size=None, keyed_vectors=None, coherence='c_v', topn=20, processes=-1)

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                                                random_state=100)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


print('\nCalculating Coherence score...')
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix,
                                                        texts=tokenized_text, start=topic_min, limit=topic_max, step=1)
topic_best = model_list[np.argmax(coherence_values)].num_topics
print('Finished:::')
print('\nOptimal number of topics: ', topic_best)

'''
# Show graph
limit = 50;
start = 2;
step = 1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()  # Print the coherence scores
'''

# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
''' 
# https://miningthedetails.com/blog/python/lda/GensimLDA/
passes:         Number of passes through the entire corpus
chunksize:      Number of documents to load into memory at a time and process E step of EM.
update_every:   Number of chunks to process prior to moving onto the M step of EM.

# Gensim model
https://radimrehurek.com/gensim/models/ldamodel.html
'''
print('\nRunning model...')
best_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=topic_best, random_state=100,
                 per_word_topics=True, chunksize=chunk, passes=500, iterations=10000, alpha='auto',
                 eta='auto')  # , minimum_phi_value=0.2
print('Model finished:::')

best_model.show_topics()
# A measure of how good the model is. lower the better.
print('\nPerplexity best model: ', best_model.log_perplexity(doc_term_matrix, total_docs=len(tokenized_text)))

# Coherence best model
coherence_values_best = CoherenceModel(model=best_model, texts=tokenized_text, dictionary=dictionary, coherence='c_v')
coherence_score_best = coherence_values_best.get_coherence()
print('Coherence of best model: ' + str(round(coherence_score_best, 4)))

# Extract table of topics keywords
df_rho = pd.DataFrame(best_model.print_topics())
df_rho.columns = ('Topic', 'Rho')

# Clean the df
s = df_rho['Rho'].apply(lambda x: x.replace('+', ''))
s = s.apply(lambda x: x.replace('"', ''))
s = s.apply(lambda x: x.replace('*', ' '))
s = s.str.split()
s = s.apply(lambda x: pd.Series(list(x)))

df_rho = pd.concat([df_rho['Topic'].reset_index(drop=True), s], axis=1)


def format_topics_sentences(ldamodel=None, corpus=None, text_org=None, text=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                # print(wp)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 2), topic_keywords, row]), ignore_index=True)
                # print(row)
            else:
                break
    # sent_topics_df.columns = ['Dominant_Topic', 'Perc_main_topic', 'Topic_keywords', 'All_topic_prob']

    # Add original text to the end of the output
    sent_topics_df = pd.concat([sent_topics_df, text_org, text], axis=1)
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_main_topic', 'Topic_keywords', 'All_topic_prob', 'Text_org',
                              'Text']
    return (sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=best_model, corpus=doc_term_matrix, text_org=df_org['Text'],
                                                  text=df['Text'])

# Sort documents on topic assignment
df_topic_sents_keywords = df_topic_sents_keywords.sort_values(by='Dominant_Topic')

# Calculate topic frequency
(unique, counts) = np.unique(df_topic_sents_keywords['Dominant_Topic'], return_counts=True)
frequency = np.asarray((unique, counts)).T
print('\nTopic frequency: ')
print(frequency.astype(int))

# Automatic labeling of multinomial topic models
df_bigram = df_topic_sents_keywords[['Dominant_Topic', 'Topic_keywords', 'Text']].sort_values(by='Dominant_Topic')

Topic_keywords = [[topic.split(', ')] for topic in df_bigram['Topic_keywords'].unique()]

# Create subset of topic keywords combinations
subset = []
for topic in range(len(Topic_keywords)):
    subset.append([subset_loop for subset_loop in itertools.combinations(Topic_keywords[topic][0], 2)])

# Calculate pointwise mutual information (PMI)
epsilon = 0.000001
PMI = [[] for i in range(len(Topic_keywords))]
for topic in range(len(Topic_keywords)):
    for bigram in subset[topic]:
        #print(bigram)
        z2_freq = []
        z1_1_freq = []
        z1_2_freq = []
        for doc in df_bigram['Text'][df_bigram['Dominant_Topic'] == topic]:
            z1_1_freq.append(bigram[0] in doc)
            z1_2_freq.append(bigram[1] in doc)
            z2_freq.append(bigram[0] in doc and bigram[1] in doc)

        PMI[topic].append([bigram, np.log((sum(z2_freq) / len(df_bigram[df_bigram['Dominant_Topic'] == topic]) + epsilon) / (
                    epsilon+((sum(z1_1_freq) / len(df_bigram[df_bigram['Dominant_Topic'] == topic]))) * (
                        sum(z1_2_freq) / len(df_bigram['Text'][df_bigram['Dominant_Topic'] == topic]))))])

Topic_lable = []
for topic in range(len(Topic_keywords)):
    PMI[topic].sort(key=lambda x: x[1], reverse=True)
    Topic_lable.append([topic, PMI[topic][0][0]])

Topic_frequency = pd.DataFrame(frequency[:,1].astype(int), columns=['Frequency'])

Topic_lable_df = pd.DataFrame(Topic_lable, columns=('Topic', 'PMI Label'))
df_rho = pd.concat([Topic_lable_df.reset_index(drop=True), Topic_frequency, df_rho.iloc[:,1:]], axis=1)

# Write to excel
writer = pd.ExcelWriter(wd + 'LDA_Thesis_Output.xlsx', engine='xlsxwriter')

df_topic_sents_keywords.to_excel(writer, sheet_name='Topic_assign')
df_rho.to_excel(writer, sheet_name='Word_prob')

writer.save()
writer.close()
print('Script finished:.')

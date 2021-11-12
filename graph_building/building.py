#!/usr/bin/env python
# coding: utf-8

# # Building Graph for LINE

# ---

# In[10]:


from math import log


# In[2]:


import numpy as np
title_content_dic = np.load("../preprocessing/model_vocab/mecab/sigle-title/title_content_token.dic.npy",allow_pickle=True)
title_content_dic = title_content_dic.tolist()


# In[3]:


svd_matrix = np.load("./data/LSA_matrix/only_ja_title/svd_matrix.npy",allow_pickle=True)
svd_matrix = svd_matrix.tolist()


# In[ ]:


print("buiding Words Graph...")


# In[4]:


# build vocab
word_freq = {}
word_set = set()
for doc_words in title_content_dic.values():
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)


# In[5]:


word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i


# In[6]:


# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in title_content_dic.values():
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)


# In[12]:


word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])


# In[ ]:





# In[18]:


word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1


# In[26]:


num_window = len(windows)

pmi_graph = open("word.graph", 'w')

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    pmi_graph.writelines("w-"+vocab[i]+" "+"w-"+vocab[j]+" "+str(pmi))
    pmi_graph.write('\n')
pmi_graph.close()


# In[ ]:


# doc


# In[ ]:


print("buiding docs Graph...")


# In[6]:


doc_word_freq = {}

for doc_id in range(len(title_content_dic.values())):
    doc_words = list(title_content_dic.values())[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1


# In[7]:


word_doc_list = {}

for i in range(len(title_content_dic.values())):
    doc_words = list(title_content_dic.values())[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)


# In[13]:


title_content_key_list=list(title_content_dic.keys())


# In[16]:


tfidf_graph = open("doc.graph", 'w')



for i in range(len(title_content_dic.values())):
    doc_words = list(title_content_dic.values())[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        
        idf = log(1.0 * len(title_content_dic.values()) /
                  word_doc_freq[vocab[j]])
        
        tfidf_graph.writelines("w-"+vocab[j]+" "+"d-"+title_content_key_list[i]+" "+str(freq * idf))
        tfidf_graph.write('\n')
        
        doc_word_set.add(word)
        
tfidf_graph.close()


# In[ ]:


# topic


# In[ ]:


print("buiding topics Graph...")


# In[62]:


lsa_graph = open("topic.graph", 'w')

for doc_i, doc_vec in enumerate(svd_matrix):
    for topic_i, weight in enumerate(doc_vec):
        if weight > 0.05:
            lsa_graph.writelines("d-"+title_content_key_list[doc_i]+" "+"t-"+str(topic_i)+" "+str(weight))
            lsa_graph.write('\n')
            #print("doc"+str(doc_i)+" "+"topic"+str(topic_i)+" "+str(weight))
        else:
            continue
            
lsa_graph.close()


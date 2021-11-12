#!/usr/bin/env python
# coding: utf-8

# # MeCab

# In[32]:


import MeCab
import numpy as np


# In[33]:


#np.save("title_content.dic", title_content_dic)

title_content_dic = np.load("./model_vocab/mecab/sigle-title/title_content.dic.npy",allow_pickle=True)
title_content_dic = title_content_dic.tolist()


# In[34]:


dataset = "mecab"
import re
stop_words = []
for word in open("./stopwords/stopwords/stop-words_japanese_1_ja.txt",'r'):
    stop_words.append(re.sub(r"\n","",word))


# In[27]:


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\d", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\n", "", string)
    return string.strip().lower()


# In[42]:


mecab_tagger = MeCab.Tagger("-Owakati")
word_freq = {}  # to remove rare words
for doc_content in title_content_dic.values():
    doc_content = mecab_tagger.parse(doc_content).split(" ")
    for word in doc_content:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
print("building words frequncy...")


# In[29]:


np.save("./model_vocab/"+dataset+"/word_freq.dic", word_freq)


# In[39]:


print("len(title_content_dic.keys())",len(title_content_dic.keys()))


# In[43]:


print("number of content words ferquncy: ",len(word_freq.keys()))


# In[100]:


word_freq_sorted = sorted(word_freq.items(), key = lambda kv:(kv[1], kv[0]))


# In[41]:


for title, doc_content in title_content_dic.items():
    doc_words = []
    doc_content = clean_str(doc_content)
    doc_content = mecab_tagger.parse(doc_content).split(" ")
    for word in doc_content:
        if word not in stop_words and word_freq[word] >= 200:
            doc_words.append(word)
    title_content_dic[title] = " ".join(doc_words)
    
np.save("./model_vocab/mecab/sigle-title/title_content_token.dic", title_content_dic)


# In[44]:


title_content_dic["会社"]


# In[ ]:





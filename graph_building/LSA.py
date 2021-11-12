#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
title_content_dic = np.load("../preprocessing/model_vocab/mecab/sigle-title/title_content_token.dic.npy",allow_pickle=True)
title_content_dic = title_content_dic.tolist()


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


# In[4]:


# raw documents to tf-idf matrix: 
vectorizer = TfidfVectorizer(use_idf=True,smooth_idf=True)


# In[5]:


# SVD to reduce dimensionality: 
svd_model = TruncatedSVD(n_components=2000,algorithm='randomized',n_iter=10)


# In[6]:


# pipeline of tf-idf + SVD, fit to and applied to documents:
svd_transformer = Pipeline([('tfidf', vectorizer),('svd', svd_model)])


# In[8]:


svd_matrix = svd_transformer.fit_transform(title_content_dic.values())


# In[9]:


svd_matrix.shape


# In[42]:


np.save("./data/LSA_matrix/only_ja_title/svd_matrix", svd_matrix)


# In[43]:


list(title_content_dic.items())[1]


# In[79]:


def count_min_max_aver(lines):
    min_len = 130383
    aver_len = 0
    max_len = 0
    for temp in lines:
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    aver_len = 1.0 * aver_len / len(lines)
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))


# In[12]:


from sklearn.metrics.pairwise import cosine_similarity


# In[72]:


comp_sim = cosine_similarity(svd_matrix[1].reshape(1,-1),svd_matrix)


# In[73]:


similar_indices = comp_sim.argsort().flatten()[-5:] 


# In[74]:


for i in similar_indices:
    print("----------------------------")
    print(list(title_content_dic.items())[i])


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Wikipeia Per-processing

# ---

# for text_gcn form

# In[57]:


import re
import MeCab

# In[67]:


index_content_dic = {}
title_list = []
count = -1
for line in open('wiki.txt'):
    if re.match('<.*>', line):
        title_list.append(line)
        count+=1
        index_content_dic[count] = []
        continue
    index_content_dic_value = index_content_dic[count]
    index_content_dic_value.append(line)
    index_content_dic[count] = index_content_dic_value


# In[68]:


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


# In[69]:


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\d", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"（", "", string)
    string = re.sub(r"）", "", string)
    string = re.sub(r"。", "", string)
    string = re.sub(r"、", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\n", "", string)
    return string.strip().lower()


# In[70]:


print("index_content_dic.values()",count_min_max_aver(index_content_dic.values()))


# In[71]:





# In[30]:


def check_contain_ja(check_str):
    for ch in check_str:
        if '\u3040' <= ch <= '\u309f' or  '\u30a0' <= ch <= '\u30ff'  or  '\u4e00' <= ch <= '\u9fbf' :
            return False
    return True


# In[72]:


mt = MeCab.Tagger('')
mt.parse('')

title_content_dic = {}
for i, content in index_content_dic.items():
    if not content:
        continue
    title = re.sub(r"\n", "", str(content[0]))
    
    text = []
    node = mt.parseToNode(title.strip())
    while node:
        fields = node.feature.split(",")
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            text.append(node.surface)
        node = node.next
    
    
    if len(text) != 1 or check_contain_ja(text):
        continue
    content = content[1:]
    if len(content) > 10:
        content = content[:10]
    content_str = ''.join(re.sub(r"\n", "", str(s)) for s in content)
    if len(content_str) < 5:
        continue
    title_content_dic[title] = clean_str(content_str)


# In[73]:


print("len(title_content_dic.keys())",len(title_content_dic.keys()))


# In[74]:


print("title len:",count_min_max_aver(title_content_dic.keys()))


# In[76]:


print("content len: ",count_min_max_aver(title_content_dic.values()))


# In[77]:


print(title_content_dic["アップル"])


# In[46]:


#len(title_content_dic["apple"])


# In[78]:


import numpy as np

np.save("./model_vocab/mecab/sigle-title/title_content.dic", title_content_dic)


# In[197]:


#title_content_dic = np.load("title_content.dic.npy",allow_pickle=True)


# In[199]:


#title_content_dic = title_content_dic.tolist()


# In[200]:


#len(title_content_dic.keys())


# In[ ]:





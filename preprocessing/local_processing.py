#!/usr/bin/env python
# coding: utf-8

# # Word processing for Sememe DNC

# ---

# In[1]:


import glob
import os


# In[2]:


def _get_text_file(text_dir):
    file_list = glob.glob(f'{text_dir}*/all.txt')
    files = ",".join(file_list)
    return files


# In[3]:


files = _get_text_file("./data/")


# In[2]:


import re
def word_processing(raw_text):
    filters =  "[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]|[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉・,\./『』【】「」→←○《》≪≫\n\u3000]"
    raw_words = raw_text
    cleanr = re.compile('<.*?>|[\\r]|[\\n]')
    raw_words = re.sub(cleanr, '', raw_words)
    return re.sub(filters, '', raw_words)



# In[4]:


import sys
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()





# In[5]:


f=open('wiki.txt','w')
for file in progressbar(_get_text_file("./data/").split(","), "Has been completed: ", 30):
    for line in open(file):
        f.writelines(line)
f.close()


# In[ ]:





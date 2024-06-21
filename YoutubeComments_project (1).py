#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import  seaborn as sns 
import matplotlib.pyplot as plt


# In[3]:


pip install --upgrade pandas


# In[4]:


pip install --upgrade pandas jupyter


# In[5]:


comments=pd.read_csv('C:\Data_analyst_udemy_project/UScomments.csv',  on_bad_lines='skip' ) 


# In[6]:


comments.head()


# In[7]:


comments.isnull().sum()


# In[8]:


comments.dropna(inplace=True)  


# In[9]:


comments.isnull().sum()


# In[ ]:





# In[10]:


get_ipython().system('pip install textblob')


# In[11]:


from textblob import TextBlob


# In[12]:


comments.head(6)


# In[13]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity


# In[14]:


comments.shape


# In[15]:


sample_df=comments[0:1000]


# In[16]:


sample_df.shape


# In[17]:


polarity=[]

for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[18]:


len(polarity)


# In[19]:


comments.head()


# In[20]:


comments['polarity'] = polarity


# In[21]:


comments.head()


# In[22]:


#3. worldcloud analysis of your data 


# In[23]:


filter1 = comments['polarity'] == 1


# In[24]:


comments_positive=comments[filter1]


# In[ ]:





# In[25]:


filter2 = comments['polarity'] == -1


# In[26]:


comments_negative=comments[filter2]


# In[27]:


comments_positive.head()


# In[28]:


get_ipython().system('pip install wordcloud')


# In[29]:


from wordcloud import WordCloud, STOPWORDS


# In[30]:


set(STOPWORDS)


# In[31]:


comments['comment_text']


# In[32]:


type(comments['comment_text'])


# In[33]:


total_comments_positive = ' '.join(comments_positive['comment_text'])


# In[34]:


wordcloud=WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_positive)


# In[35]:


plt.imshow(wordcloud)
plt.axis('off')


# In[36]:


total_comments_negative = ' '.join(comments_negative['comment_text'])


# In[37]:


wordcloud2 = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_negative)


# In[38]:


plt.imshow(wordcloud2)
plt.axis('off')


# In[39]:


#4 Perform Emoji,s Analysis


# In[40]:


get_ipython().system('pip install emoji==2.2.0')


# In[41]:


import emoji


# In[42]:


emoji.__version__


# In[43]:


comments['comment_text'].head(6)


# In[44]:


comment = 'trending 😉'


# In[45]:


emoji_list = []
for char in comment:
    if char in emoji.EMOJI_DATA:
        emoji_list.append(char)


# In[46]:


emoji_list


# In[47]:


all_emojis_list=[]

for comment in comments['comment_text'].dropna():
    for char in comment:
        if char in emoji.EMOJI_DATA:
            all_emojis_list.append(char)


# In[48]:


all_emojis_list[0:10]


# In[49]:


from collections import Counter


# In[50]:


Counter(all_emojis_list).most_common(10)


# In[51]:


Counter(all_emojis_list).most_common(10)[0][0]


# In[52]:


Counter(all_emojis_list).most_common(10)[1][0]


# In[53]:


Counter(all_emojis_list).most_common(10)[2][0]


# In[54]:


Counter(all_emojis_list).most_common(10)[0][1]


# In[55]:


Counter(all_emojis_list).most_common(10)[1][1]


# In[56]:


emojis=[Counter(all_emojis_list).most_common(10)[i][0] for i in range(10)]


# In[ ]:





# In[57]:


freqs=[Counter(all_emojis_list).most_common(10)[i][1] for i in range(10)]


# In[58]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[59]:


trace=go.Bar(x=emojis,y=freqs)


# In[60]:


iplot([trace])


# In[ ]:





# In[61]:


#4 Collect Entire data of youtube?


# In[62]:


import os


# In[63]:


files=os.listdir(r'C:\Data_analyst_udemy_project\additional_data')


# In[64]:


files_csv=[file for file in files if '.csv' in file]


# In[65]:


files_csv


# In[66]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:





# In[67]:


full_df=pd.DataFrame()
path = r'C:\Data_analyst_udemy_project\additional_data'

for file in files_csv:
    current_df= pd.read_csv(path+'/'+file, encoding='iso-8859-1',on_bad_lines='skip')
    
    full_df = pd.concat([full_df, current_df], ignore_index=True)


# In[68]:


full_df.shape


# In[69]:


#How to export your data into (csv,json,db)


# In[70]:


full_df[full_df.duplicated()].shape


# In[71]:


full_df=full_df.drop_duplicates() 


# In[72]:


full_df.shape


# In[73]:


full_df[0:1000].to_csv(r'C:\Data_analyst_udemy_project\additional_data/youtube_sample.csv',index=False)


# In[74]:


full_df[0:1000].to_json(r'C:\Data_analyst_udemy_project\additional_data/youtube_sample.json')


# In[75]:


from sqlalchemy import create_engine


# In[76]:


engines = create_engine(r'sqlite:/// C:\Data_analyst_udemy_project\additional_data/youtibe_sample.sqlite')


# In[ ]:





# In[77]:


#5which category has the maximum likes?


# In[78]:


full_df.head()


# In[79]:


full_df['category_id'].unique()


# In[80]:


json_df=pd.read_json(r'C:\Data_analyst_udemy_project\additional_data/US_category_id.json')


# In[81]:


json_df


# In[82]:


json_df['items'][0]


# In[83]:


json_df['items'][1]


# In[84]:


cat_dict = {}
for item in json_df['items'].values:
    cat_dict[int(item['id'])]=item['snippet']['title']


# In[85]:


cat_dict


# In[87]:


full_df['category_name']= full_df['category_id'].map(cat_dict)


# In[88]:


full_df.head()


# In[94]:


plt.figure(figsize=(12,8))
sns.boxplot(x='category_name', y='likes', data=full_df)

plt.xticks(rotation='vertical')


# In[ ]:


#6 find out whether audience is engaged or not


# In[97]:


full_df['like_rate']=(full_df['likes']/full_df['views'])*100
full_df['dislike_rate']=(full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate']=(full_df['comment_count']/full_df['views'])*100


# In[98]:


full_df.columns


# In[99]:


full_df.head()


# In[101]:


plt.figure(figsize=(8,6))
sns.boxplot(x='category_name', y='like_rate', data=full_df)

plt.xticks(rotation='vertical')
plt.show()


# In[102]:


sns.regplot(x='views', y='likes', data=full_df)


# In[103]:


full_df.columns


# In[105]:


full_df[['views','likes','dislikes']].corr()


# In[107]:


sns.heatmap(full_df[['views','likes','dislikes']].corr(),annot=True)


# In[ ]:


#7 which channels have the largest number of trending videos?


# In[108]:


full_df.head(6)


# In[109]:


full_df['channel_title'].value_counts()


# In[114]:


cdf=full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()


# In[115]:


cdf=cdf.rename(columns={0:'total_videos'})


# In[116]:


cdf


# In[117]:


import plotly.express as px


# In[118]:


px.bar(data_frame=cdf[0:20],x='channel_title', y='total_videos')


# In[ ]:


#8. Does Panctuation in title and tags have any relation with views,likes, dislikes comments?


# In[119]:


full_df['title'][0]


# In[120]:


import string


# In[121]:


string.punctuation


# In[124]:


len([char for char in full_df['title'][0] if char in string.punctuation])


# In[125]:


def punc_count(text):
    return len([char for char in text if char in string.punctuation])


# In[126]:


full_df['title'].apply(punc_count)


# In[127]:


sample=full_df[0:10000]


# In[128]:


sample['count_punc']=sample['title'].apply(punc_count)


# In[129]:


sample['count_punc']


# In[131]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc', y = 'views', data = sample)
plt.show()


# In[132]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc', y = 'likes', data = sample)
plt.show()


# In[ ]:





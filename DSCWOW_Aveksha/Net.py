import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.stem import PorterStemmer
nltk.download('punkt')
from collections import Counter
import re
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()


# In[3]:


consumerKey = "m20YJPqehD4J9Qvdq426Wn8yb"
consumerSecret = "ONI1K90Pq0NNaEbm3Jn3CxLO5ViDxnj5s163fij21jcnsQ81UC"
accessToken = "1171293910175383554-c1zFJv4Hlx8Air1ZodaWdWlhh53oyS"
accessTokenSecret = "fPtou2yiZGh4fUpy36Vzqy82Jk5VQHmC0MYitJEoIvsdr"


# In[4]:


# Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
    
# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret) 
    
# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit = True)


# In[5]:


from datetime import date

today = date.today()


# In[8]:


data = pd.read_csv("Data.csv")


# In[9]:


data


# In[10]:


data.drop(data.columns[[3,4,5]], axis = 1, inplace = True) 


# In[11]:


data


# In[12]:


def findlat(city):
    latitude=data.Lat[data['City']==city]
    return latitude
def findlong(city):
    longitude=data.Long[data['City']==city]
    return longitude


# In[13]:


def scraptweets(search_words,numTweets):
    max_range = 100
    for i in range(0, 1):
        tweets = tweepy.Cursor(api.search, lang="en",q=search_words, geocode="%f,%f,%dkm" % (latitude, longitude, max_range), tweet_mode='extended').items(numTweets)
        tweet_list = [tweet for tweet in tweets]
    for tweet in tweet_list:
        username = tweet.user.screen_name
        location = tweet.user.location
        hashtags = tweet.entities['hashtags']
        try:
            text = tweet.retweeted_status.full_text
        except:  # Not a Retweet
            text = tweet.full_text
            ith_tweet = [username, location, text, hashtags]
            db_tweets.loc[len(db_tweets)] = ith_tweet


# In[15]:


# Initialise these variables:
db_tweets = pd.DataFrame(columns = ['username','location', 'text', 'hashtags'])
search_words = "women"
date_since = today
city=input("Enter your city: ")
latitude= findlat(city)
longitude = findlong(city)
scraptweets(search_words,100)


# # Data cleaning

# In[16]:


dataset=db_tweets
dataset.head()


# In[17]:


dataset.shape


# ### Cleaning for duplicates and null values

# In[18]:


dataset['text'].isna().sum() #no null tweets


# ### Removing mentions

# In[19]:


dataset['clean_tweet'] = dataset['text'].apply(lambda x: ' '.join([tweet for tweet in x.split() if not tweet.startswith("@")]))


# In[20]:


dataset.head()


# ### Removing numbers

# In[21]:


dataset['clean_tweet'] = dataset['clean_tweet'].apply(lambda x: ' '.join([tweet for tweet in x.split() if not tweet.isnumeric()]))


# ### Correcting slang words

# In[22]:


slang = {'luv':'love','wud':'would','lyk':'like','wateva':'whatever','ttyl':'talk to you later',
          'kul':'cool','fyn':'fine','omg':'oh my god!','fam':'family','bruh':'brother', 'cud':'could',
         'fud':'food', 'u':'you', 'ur':'your', 'frm': 'from'}

dataset['clean_tweet'] = dataset['clean_tweet'].apply(lambda x : ' '.join(slang[word] if word in slang else word for word in x.split()))


# ### Finding hashtags

# In[23]:


dataset['Hashtags'] = dataset['clean_tweet'].apply(lambda x : ' '.join([word for word in x.split() if word.startswith('#')]))


# In[24]:


dataset.drop('text',axis=1,inplace=True)
dataset.drop('hashtags',axis=1,inplace=True)
dataset.head()


# ### Removing stopwords

# In[25]:


dataset['clean_tweet'] = dataset['clean_tweet'].apply(lambda x : ' '.join([word for word in x.split() if not word in set(stopwords.words('english'))]))


# ### Lemmatization

# In[26]:


lemmatizer = WordNetLemmatizer()
dataset['clean_tweet'] = dataset['clean_tweet'].apply(lambda x : ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))


# ### Stemming

# In[27]:


ps = PorterStemmer()
dataset['clean_tweet'] = dataset['clean_tweet'].apply(lambda x : ' '.join([ps.stem(word) for word in x.split()]))


# In[28]:


#dataset.drop('hashtags',axis=1,inplace=True)
dataset['text']=dataset['clean_tweet']
dataset.drop('clean_tweet',axis=1,inplace=True)


# # Visualizations

# ### Wordcloud

# In[29]:


df = dataset


# In[30]:


def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity


# Create two new columns 'Subjectivity' & 'Polarity'
df['Subjectivity'] = df['text'].apply(getSubjectivity)
df['Polarity'] = df['text'].apply(getPolarity)

# Show the new dataframe with columns 'Subjectivity' & 'Polarity'


# In[31]:


allWords = ' '.join([twts for twts in df['text']])
new_stopwords=["woman","women","girl","women'","https"]
wc = WordCloud(width = 800, height = 500, max_font_size = 110, max_words=100, stopwords=new_stopwords).generate(allWords)
plt.figure(figsize=(12,8))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[32]:


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
df['Analysis'] = df['Polarity'].apply(getAnalysis)
# Show the dataframe
df


# In[33]:


ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['text']
ptweets

round( (ptweets.shape[0] / df.shape[0]) * 100 , 1)


# In[34]:


ntweets = df[df.Analysis == 'Negative']
ntext = ntweets['text']
nusers = ntweets['username']
ntweets

round( (ntweets.shape[0] / df.shape[0]) * 100, 1)


plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind = 'bar')
plt.show()


nusers


# In[ ]:





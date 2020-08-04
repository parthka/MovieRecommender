# [Modern Movie Recommender](https://modern-movie-recommender.ue.r.appspot.com/)  
***[TMDB](https://www.themoviedb.org/) is a database that hosts information about movie and TV shows. From here, you can get information like cast, budget, revenue,
 as well as basic information about a movie such as the date it was released and a brief synopsis.***

In this project, I am going to build a movie recommendation app using data pulled back from TMDB.  
TMDB has a robust API that is free to use for small sized projects. It has information on movies and TV shows. I will limit myself to movies made in 2000 or later. Using overviews of movies from TMDB, I will build recommendations for movies based on similarities (ising natural language processing or NLP) in the text used to describe a movie along with their genres. These recommendations will then all be served in a RESTful API built in Flask. The flask web app I built is located here:
### [Get Movie Recommendations](https://modern-movie-recommender.ue.r.appspot.com/)  
Below is a guide of all the steps involved in this end-to-end project. I won't go super in-depth on describing the methods or techniques, this is moreso showcasing how to replicate the work if you'd like to solve a similar problem.
  
## Pulling Data from TMDB  
[TMDB has a well documented API](https://www.themoviedb.org/documentation/api). We will be using the "discover" and "movie" libraries. From looking over documentation for the [discover](https://developers.themoviedb.org/3/discover/movie-discover) API, we can see that only the API key is required, but there are several optional values to help us pull the data we need. I've decided to use a few options:  
- release year: cycle through each year to pull back a large selection of movies from each year  
- with_original_language: focus on english movies only  
- sort_by: using revenue as a proxy for popularity, want to sort by highest revenue films for each year in descending order  
  
For the sake of keeping this manageable, I only pulled back movies from the year 2000 to now. I'm partial to newer movies, but I also am going in with the assumption that data for newer movies will be more accurate.  
#### The code below will cycle through each page of results from the API for each year, returning the top 1,000 grossing movies for each year:
```python
# import necessary libraries
import time
import pandas as pd
import numpy as np
import json
import requests
from config import tmdb_key
from rake_nltk import Rake
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
```
```python
# declare static URL for API, empty list
tmdb_url = r"https://api.themoviedb.org/3/discover/movie"
tmdb = []

# Each page has 20 results
# To limit to widely known movies - limit to top 1,000 grossing movies per year; only pull back 50 pages per year for Discover Library API call
# Sort by revenue to get top 1,000, stop if there are no more pages
for year in range(2000, 2021):
    for page in range(1, 51):

        discover_params = {
            'api_key': tmdb_key,
            'primary_release_year': year,
            'include_adult': "false",
            'include_video': "false",
            'with_original_language': 'en',
            'sort_by': 'revenue.desc',
            'page': page
        }
        try: 
            response = requests.get(tmdb_url, params = discover_params)
            results = response.json()['results']
            for item in results:
                tmdb.append(item)
            print("Year {}, Page {} done".format(year, page))
        except:
            break
```
  
```python
print("Count of movies: {:,}".format(len(tmdb)))
```

    Count of movies: 21,000
  
Once we have this metadata from the Discover library, we can use the JSON results that we've put into a list to get specific information on each movie. I'm particularly interested in the overview and genre
 information from the [movie](https://developers.themoviedb.org/3/movies/get-movie-details) library:  
#### I'm going to declare a shell for the DataFrame that will house the information from the movie API call, then for each loop, make a new record in the DataFrame based on features I'd like to pull back:  
```python
columns = ['tmdb_id', 'imdb_id', 'title', 'budget', 'revenue', 'release_date', 'release_year', 'genres', 'overview']
movies_df = pd.DataFrame(columns=columns)
```
```python
# static TMDB movie API URL
tmdb_movie_url = r"https://api.themoviedb.org/3/movie/"

print("Start time: ", time.strftime("%H:%M:%S", time.localtime(time.time())))

for movie in tmdb:
    movie_id = movie['id']
    overview = movie['overview']
    release_year = int(movie['release_date'][:4])
    
    # params specific to movie API
    tmdb_movie_params = {
        'api_key': tmdb_key,
        'language': "en-US"
    }
    
    # make request to API
    tmdb_response = requests.get(tmdb_movie_url+str(movie_id), params = tmdb_movie_params)
    # loop over only if data exists
    if tmdb_response.status_code == 200:
        tmdb_movie_json = tmdb_response.json()

        # make sure data exists - else save as NA
        # add into our DataFrame of movies
        # results show genre data needs to be flattened
        # return empty if no genre data
        try:
            genres_pd = pd.json_normalize(tmdb_movie_json['genres'])
            genres = genres_pd['name'].str.cat(sep = ', ')
            movies_df.loc[len(movies_df)] = [movie_id,
                                            tmdb_movie_json['imdb_id'],
                                            tmdb_movie_json['title'],
                                            tmdb_movie_json['budget'],
                                            tmdb_movie_json['revenue'],
                                            tmdb_movie_json['release_date'],
                                            release_year,
                                            genres,
                                            overview]
        except:
            movies_df.loc[len(movies_df)] = [movie_id,
                                    tmdb_movie_json['imdb_id'],
                                    tmdb_movie_json['title'],
                                    tmdb_movie_json['budget'],
                                    tmdb_movie_json['revenue'],
                                    tmdb_movie_json['release_date'],
                                    release_year,
                                    "NA",
                                    overview]
        print(".", end = " ")
        
print("/nEnded loop at: ", time.strftime("%H:%M:%S", time.localtime(time.time())))
```  
This took roughly 50 minutes. From here, I'm adding a column that will have each genre split separately:
```python
# add column which treats genres as a list
movies_df['genre_list'] = movies_df.genres.str.split(',')
movies_df.head(10)
```
After this was done, I saved to a data folder, so I could have a place to pick up from:
```python
# Save data
movies_df.to_pickle('../Data/movies_df.pkl')
```  
  
## Creating Our Recommendation Data  
***To make accurate recommendations, we'll be using a few different NLP techniques to clean and vectorize overviews for each movie, and use the keywords from each overview along with the genres to make movie recommendations***  
Before I go any further, I want to make sure that I will have good data to make recommendations with. So I'm going to limit to movies that have a genre, as well as minimum length for the overview. I'm also using the IMDB ID as an indicator for whether the data for the specifc movie is reliable or not, with the assumption that movies without one do not have reliable information.
```python
tmdb_df = pd.read_pickle('../Data/movies_df.pkl')
```
```python
# Before processing for key words, remove any movie titles from database that do not have an overview or genre

# Going to link to IMDB - exclude anything without an IMDB ID populated
imdb_mask =  tmdb_df['imdb_id'].str.len() > 0
# Overview must at least have at least a three word summary
overview_length_mask = tmdb_df['overview'].str.split().apply(lambda x: len(x)) >= 3
# Remove values of NA in genre column
genre_not_empty_mask = tmdb_df['genres'] != "NA"
tmdb_df1 = tmdb_df[(imdb_mask) & (overview_length_mask) & (genre_not_empty_mask)].copy()
# reset index
tmdb_df1.reset_index(inplace=True)
tmdb_df1.drop('index', axis=1, inplace=True)
tmdb_df1
```  
After doing so, we are left with 9,650 movies:
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tmdb_id</th>
      <th>imdb_id</th>
      <th>title</th>
      <th>budget</th>
      <th>revenue</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>genres</th>
      <th>overview</th>
      <th>genre_list</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>955</td>
      <td>tt0120755</td>
      <td>Mission: Impossible II</td>
      <td>125000000</td>
      <td>546388105</td>
      <td>2000-05-24</td>
      <td>2000</td>
      <td>Action, Adventure, Thriller</td>
      <td>With computer genius Luther Stickell at his si...</td>
      <td>[Action,  Adventure,  Thriller]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>98</td>
      <td>tt0172495</td>
      <td>Gladiator</td>
      <td>103000000</td>
      <td>460583960</td>
      <td>2000-05-01</td>
      <td>2000</td>
      <td>Action, Adventure, Drama</td>
      <td>In the year 180, the death of emperor Marcus A...</td>
      <td>[Action,  Adventure,  Drama]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8358</td>
      <td>tt0162222</td>
      <td>Cast Away</td>
      <td>90000000</td>
      <td>429632142</td>
      <td>2000-12-22</td>
      <td>2000</td>
      <td>Adventure, Drama</td>
      <td>Chuck Nolan, a top international manager for F...</td>
      <td>[Adventure,  Drama]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3981</td>
      <td>tt0207201</td>
      <td>What Women Want</td>
      <td>70000000</td>
      <td>374111707</td>
      <td>2000-12-15</td>
      <td>2000</td>
      <td>Comedy, Romance</td>
      <td>Advertising executive Nick Marshall is as cock...</td>
      <td>[Comedy,  Romance]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10567</td>
      <td>tt0130623</td>
      <td>Dinosaur</td>
      <td>127500000</td>
      <td>354248063</td>
      <td>2000-05-19</td>
      <td>2000</td>
      <td>Animation, Family</td>
      <td>An orphaned dinosaur raised by lemurs joins an...</td>
      <td>[Animation,  Family]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9645</th>
      <td>714996</td>
      <td>tt11957868</td>
      <td>Peach</td>
      <td>0</td>
      <td>0</td>
      <td>2020-01-13</td>
      <td>2020</td>
      <td>Comedy</td>
      <td>A socially anxious young woman lands a hot dat...</td>
      <td>[Comedy]</td>
    </tr>
    <tr>
      <th>9646</th>
      <td>714936</td>
      <td>tt11754128</td>
      <td>Atlas</td>
      <td>0</td>
      <td>0</td>
      <td>2020-01-13</td>
      <td>2020</td>
      <td>Science Fiction</td>
      <td>Atlas and his dog Charlie are both locked into...</td>
      <td>[Science Fiction]</td>
    </tr>
    <tr>
      <th>9647</th>
      <td>714847</td>
      <td>tt12299114</td>
      <td>Trapped</td>
      <td>0</td>
      <td>0</td>
      <td>2020-04-24</td>
      <td>2020</td>
      <td>Thriller</td>
      <td>A stressed young drug addicted person who is h...</td>
      <td>[Thriller]</td>
    </tr>
    <tr>
      <th>9648</th>
      <td>714842</td>
      <td>tt12498618</td>
      <td>8:46</td>
      <td>0</td>
      <td>0</td>
      <td>2020-06-11</td>
      <td>2020</td>
      <td>Comedy, Documentary</td>
      <td>From Dave: Normally I wouldn't show you someth...</td>
      <td>[Comedy,  Documentary]</td>
    </tr>
    <tr>
      <th>9649</th>
      <td>714836</td>
      <td>tt12525356</td>
      <td>Brock: Over the Top</td>
      <td>0</td>
      <td>0</td>
      <td>2020-06-22</td>
      <td>2020</td>
      <td>Documentary</td>
      <td>Brock: Over the Top is a feature length docume...</td>
      <td>[Documentary]</td>
    </tr>
  </tbody>
</table>
<p>9650 rows × 10 columns</p>
</div>
  
### The next part is to start preprocessing for our text data. [Lemmatization](https://en.wikipedia.org/wiki/Lemmatisation) is the process of removing inflections to return a word to its root form. This way, similar words can be analysed as a single item, as identified by the word's lemma. For reference, see the table below:
  
| Original Word | Word Lemma |
| ------------- | ---------- |
| Copied        | Copy       |
| Copying       | Copy       |
| Copies        | Copy       |
  
Let's go ahead and lemmatize our data. First, we'll build a function to recognize parts of speech for each text, and lemma each word based on their part of speech tag:
```python
# More pre-processing
# 1: Noise removal - get rid of non alphanumeric text
# 2: lemmatize text so root stays, but not different tenses or versions of same word (i.e. terrifying -> terrify)
    ## Much more powerful when part of speech for word is accurately identified
    ## Build function to accurately identify pos_tag, anothor to lemmatize text and return sentence lemmatized
lemm = WordNetLemmatizer()

def get_pos(tag):
    lemma_tag = tag[0].lower()
    return {
        "n": "n",
        "v": "v",
        "r": "r",
        "j": "a"
    }.get(lemma_tag, 'n')

def lemmatize(series):
    # RegEx removal of anything that is not alphanumeric
    s1 = series.str.replace('[^a-zA-Z\d\s:]', '')
    # tokenize
    s2 = s1.str.split()
    # get parts of speech tags for each word in each overview
    s3 = s2.apply(lambda x: pos_tag(x))
    # lemmatization
    s4 = s3.apply(lambda x:[lemm.lemmatize(word, pos=get_pos(tag)) for word, tag in x])
    # convert back to series
    lemma_series = s4.apply(lambda x: ' '.join(x))
    # lemmatized series
    return lemma_series
```


```python
tmdb_df1['overview_lemma'] = lemmatize(tmdb_df1['overview'])
```
***Lemma text versus non lemma text:***
```python
# Inspect
tmdb_df1['overview_lemma'][0]
```




    'With computer genius Luther Stickell at his side and a beautiful thief on his mind agent Ethan Hunt race across Australia and Spain to stop a former IMF agent from unleash a genetically engineer biological weapon call Chimera This mission should Hunt choose to accept it plunge him into the center of an international crisis of terrify magnitude'
    
```python
t1 = tmdb_df1['overview'].str.replace('[^a-zA-Z\d\s:]', '')
t1[0]
```




    'With computer genius Luther Stickell at his side and a beautiful thief on his mind agent Ethan Hunt races across Australia and Spain to stop a former IMF agent from unleashing a genetically engineered biological weapon called Chimera This mission should Hunt choose to accept it plunges him into the center of an international crisis of terrifying magnitude'

Overall, five words were lemmatized in MI2's overview *(races, unleashing, engineered, plunges, terrifying)*.  
From here, I'm going to use a method built to get keywords from text based on word occurence and co-occurence to make sure our recommender is only using important keywords to make recommendations:  
```python
from rake_nltk import Rake
# Rake - rapid automatic keyword extraction (semantically similar to TF-IDF)
    ## Gets keywords based on frequency of word occurence and co-occurence with other words in text
key_words = []
RAKE = Rake() 
for index, row in tmdb_df1.iterrows():
    RAKE.extract_keywords_from_text(row['overview_lemma'])
    key_word_scores = RAKE.get_word_degrees()
    key_words.append(list(key_word_scores.keys()))
```
  
Let's check what was extracted:
```python
# Check
print(tmdb_df1['overview_lemma'][0])
print(key_words[0])
print('Total words in overview: {:,}'.format(len(tmdb_df1['overview_lemma'][0].split())))
print('Total extracted: {:,}'.format(len(key_words[0])))
```

    With computer genius Luther Stickell at his side and a beautiful thief on his mind agent Ethan Hunt race across Australia and Spain to stop a former IMF agent from unleash a genetically engineer biological weapon call Chimera This mission should Hunt choose to accept it plunge him into the center of an international crisis of terrify magnitude
    ['genetically', 'engineer', 'biological', 'weapon', 'call', 'chimera', 'mission', 'plunge', 'terrify', 'magnitude', 'former', 'imf', 'agent', 'beautiful', 'thief', 'side', 'stop', 'spain', 'center', 'hunt', 'choose', 'computer', 'genius', 'luther', 'stickell', 'unleash', 'international', 'crisis', 'mind', 'ethan', 'race', 'across', 'australia', 'accept']
    Total words in overview: 58
    Total extracted: 34

Overall, I think this is a good enough step for this problem. Let's go ahead and append back to our dataframe.
```python
# add back to dataframe
tmdb_df1['Key_Words'] = pd.Series(key_words)
```
### The last step is to take the key words and the genres, and combine into a "bag of words" for each movie. 
```python
# Final combined DF with title and final list of words
recommend_df = pd.DataFrame(columns = ['Title', 'Recommender_BOW'])
```


```python
# Iterate through each row of movie data, combine overview text and genre tags into one text column as bag of words
for i in range(len(tmdb_df1)):
    combined_row = [*tmdb_df1['Key_Words'].tolist()[i], *tmdb_df1['genre_list'].tolist()[i]]
    # join genre & key words from overview while removing double spaced characters
    recommend_df.loc[len(recommend_df)] = [tmdb_df1.loc[i, 'title'], ' '.join(combined_row).lower().replace('  ', ' ')]
```
  
## Build recommendation
An easy way to understand how related a movie is, is to see if similar descriptions are used to describe two movies. However, before we can do that, we need a way to numerically represent the data. An easy way is to use CountVectorizer() from sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
# Vector representation of our bag of words using Count_Vectorizer: convert raw text into a sparse matrix to numerically represent words
# (can also use Python's collections library counter class)
    ## Because we have extracted key words - should just be binary i.e. whether word exists, rather than count of words
    ## Will have min document frequency of 2, max document frequency of 85%
    ## Another option would have been to include bigrams, but RAKE re-ordered key words from overview when extracting
#CV = CountVectorizer(min_df = 2, max_df = 0.85, ngram_range = (1,2))
CV = CountVectorizer(min_df = 2, max_df = 0.85)
CV_Matrix = CV.fit_transform(recommend_df['Recommender_BOW'])
```


```python
# Take a look at the CV after processing words for our recommender
print("Count Vectorizer number of documents: {:,}".format(CV_Matrix.shape[0]))
print("Count Vectorizer number of unique words (vocabulary size): {:,}".format(CV_Matrix.shape[1]))
```

    Count Vectorizer number of documents: 9,650
    Count Vectorizer number of unique words (vocabulary size): 13,615
    


```python
# Dictionary of word and position representing place in sparse matrix
print("Word: {} \nPosition: {:,}".format(list(CV.vocabulary_.keys())[0], list(CV.vocabulary_.values())[0]))
```

    Word: genetically 
    Position: 5,171

We can now numerically compare the 'bag of words' for each movie to each other. To get similarity, we can calculate the cosine similarity. It is a common similarity metric for measuring similarity between categorical data.
```python
from sklearn.metrics.pairwise import cosine_similarity
# Matrix representing cosine_similarity once our vocabulary is transformed into a numeric vector reprsentation
cosine_similarities = cosine_similarity(CV_Matrix, CV_Matrix)
```


```python
# Inspect
print(cosine_similarities)
# Top 10 similarity scores for first (should be mission impossible)
print(cosine_similarities[0].argsort())
```

    [[1.         0.05484085 0.05976143 ... 0.04364358 0.         0.02020305]
     [0.05484085 1.         0.11470787 ... 0.         0.         0.03877834]
     [0.05976143 0.11470787 1.         ... 0.04564355 0.         0.02112886]
     ...
     [0.04364358 0.         0.04564355 ... 1.         0.         0.        ]
     [0.         0.         0.         ... 0.         1.         0.07559289]
     [0.02020305 0.03877834 0.02112886 ... 0.         0.07559289 1.        ]]
    [4824 3701 3700 ... 7066 2769    0]
    
#### At this point - save cleaned TMDB DF & cosine_similarities to mark checkpoint to come back to


```python
# Save data
tmdb_df1.to_pickle('../Data/tmdb_movies.pkl')
```


```python
with open("../Data/movie_similarities.npy", 'wb') as npy:
    np.save(npy, cosine_similarities)
```


## Inspect our results  
**The last part is to build a function to get recommendations for each movie, and review the results:**
```python
# Build recommender based on above
def recommend(title):
    idx = tmdb_df1[tmdb_df1['title'] == title].index[0]
    # top 50 recommendations
    similar_movies = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)[1:51]
    # add similarity scores for top 50 - instead of iterating, pull back all data and drop null values
    recommend = pd.concat([tmdb_df1['imdb_id'], tmdb_df1['title'], tmdb_df1['release_year'], tmdb_df1['overview'], similar_movies], axis=1)
    recommend.columns = ['IMDB ID', 'Title', 'Year', 'Overview', 'Similarity Score']
    recommend = recommend.dropna()
    recommend = recommend.sort_values(by='Similarity Score', ascending=False)
    
    return recommend
```


```python
recommend('Mission: Impossible II')[:20]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IMDB ID</th>
      <th>Title</th>
      <th>Year</th>
      <th>Overview</th>
      <th>Similarity Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2769</th>
      <td>tt0317919</td>
      <td>Mission: Impossible III</td>
      <td>2006</td>
      <td>Retired from active duty to train new IMF agen...</td>
      <td>0.308607</td>
    </tr>
    <tr>
      <th>7066</th>
      <td>tt2381249</td>
      <td>Mission: Impossible - Rogue Nation</td>
      <td>2015</td>
      <td>Ethan and team take on their most impossible m...</td>
      <td>0.271052</td>
    </tr>
    <tr>
      <th>9059</th>
      <td>tt5033998</td>
      <td>Charlie's Angels</td>
      <td>2019</td>
      <td>When a systems engineer blows the whistle on a...</td>
      <td>0.216225</td>
    </tr>
    <tr>
      <th>8541</th>
      <td>tt4912910</td>
      <td>Mission: Impossible - Fallout</td>
      <td>2018</td>
      <td>When an IMF mission ends badly, the world is f...</td>
      <td>0.212512</td>
    </tr>
    <tr>
      <th>5194</th>
      <td>tt1509767</td>
      <td>The Three Musketeers</td>
      <td>2011</td>
      <td>The hot-headed young D'Artagnan along with thr...</td>
      <td>0.198898</td>
    </tr>
    <tr>
      <th>5138</th>
      <td>tt1229238</td>
      <td>Mission: Impossible - Ghost Protocol</td>
      <td>2011</td>
      <td>Ethan Hunt and his team are racing against tim...</td>
      <td>0.197245</td>
    </tr>
    <tr>
      <th>6204</th>
      <td>tt1517260</td>
      <td>The Host</td>
      <td>2013</td>
      <td>A parasitic alien soul is injected into the bo...</td>
      <td>0.195180</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>tt0283160</td>
      <td>Extreme Ops</td>
      <td>2002</td>
      <td>While filming an advertisement, some extreme s...</td>
      <td>0.187523</td>
    </tr>
    <tr>
      <th>4807</th>
      <td>tt1032751</td>
      <td>The Warrior's Way</td>
      <td>2010</td>
      <td>A warrior-assassin is forced to hide in a smal...</td>
      <td>0.187523</td>
    </tr>
    <tr>
      <th>5227</th>
      <td>tt0993842</td>
      <td>Hanna</td>
      <td>2011</td>
      <td>A 16-year-old girl raised by her father to be ...</td>
      <td>0.184428</td>
    </tr>
    <tr>
      <th>989</th>
      <td>tt0280486</td>
      <td>Bad Company</td>
      <td>2002</td>
      <td>When a Harvard-educated CIA agent is killed du...</td>
      <td>0.180702</td>
    </tr>
    <tr>
      <th>7567</th>
      <td>tt0918940</td>
      <td>The Legend of Tarzan</td>
      <td>2016</td>
      <td>Tarzan, having acclimated to life in London, i...</td>
      <td>0.180702</td>
    </tr>
    <tr>
      <th>6469</th>
      <td>tt6703928</td>
      <td>A Fool's Paradise</td>
      <td>2013</td>
      <td>James Bond is sent on a mission to investigate...</td>
      <td>0.179284</td>
    </tr>
    <tr>
      <th>8673</th>
      <td>tt4669264</td>
      <td>Beirut</td>
      <td>2018</td>
      <td>In 1980s Beirut, Mason Skiles is a former U.S....</td>
      <td>0.176547</td>
    </tr>
    <tr>
      <th>4745</th>
      <td>tt1245526</td>
      <td>RED</td>
      <td>2010</td>
      <td>When his peaceful life is threatened by a high...</td>
      <td>0.176227</td>
    </tr>
    <tr>
      <th>8054</th>
      <td>tt3501632</td>
      <td>Thor: Ragnarok</td>
      <td>2017</td>
      <td>Thor is imprisoned on the other side of the un...</td>
      <td>0.176227</td>
    </tr>
    <tr>
      <th>497</th>
      <td>tt0266987</td>
      <td>Spy Game</td>
      <td>2001</td>
      <td>On the day of his retirement, a veteran CIA ag...</td>
      <td>0.172516</td>
    </tr>
    <tr>
      <th>6221</th>
      <td>tt2312718</td>
      <td>Homefront</td>
      <td>2013</td>
      <td>Phil Broker is a former DEA agent who has gone...</td>
      <td>0.170367</td>
    </tr>
    <tr>
      <th>8952</th>
      <td>tt9314132</td>
      <td>When They Run</td>
      <td>2018</td>
      <td>A survivor of a zombie apocalypse is on the ru...</td>
      <td>0.169031</td>
    </tr>
    <tr>
      <th>8647</th>
      <td>tt5177088</td>
      <td>The Girl in the Spider's Web</td>
      <td>2018</td>
      <td>In Stockholm, Sweden, hacker Lisbeth Salander ...</td>
      <td>0.169031</td>
    </tr>
  </tbody>
</table>
</div>
  
***Not too shabby! We seem to recommend other Mission Impossible movies, as well as other actions movies. To get better results, we could use different methods to extract keywords or other preprocessing steps, as well as modifying our vector representation of our dictionary.***

**However, upon further inspection, I noticed that I was using a 9,650 x 9,650 2D Numpy array inside the recommend function. This is an incredibly large file, and would be hard to serve over the web without bogging down memory resources. I'm going to recreate the top 20 suggestions for each movie, save them to a dictionary with an index representation of each movie and their similarity. This will be used to get the recommendations on our web application.**

## Build recommender function
Below, I create a dictionary that takes the index value for each movie as the key, with the indexes of the top 20 most similar movies based on the cosine similarity between their key words and genres for each movie as the dictionary values.
```python
recommender_dict = {}
for i in range (len(tmdf1)):
    recommender_dict.update({i: pd.Series(cosine_similarities[i]).sort_values(ascending = False)[1:21].to_dict()})
```
Let's go ahead and save for future use:
```python
import pickle

f = open("../Data/cosine_dict.pkl","wb")
pickle.dump(recommender_dict,f)
f.close()
```
  
Our new recommender function is going to use the newly created dictionary. It will still search for the index of the title requested by the user, but this time, it will use that as the dictionary key. From there, we're mapping back to the original dataframe, and if there is a match on the dataframe index and an index value within the dictionary, a score will be returned. All other movies will show NaN. We'll remove null values, and sort by highest similarity, and return those top 20 results:
```python
def recommend(title):
    idx = tmdf1[tmdf1['title'] == title].index[0]
    dict_ref = recommender_dict[idx]
    df_copy = tmdf1.copy()
    df_copy['similarity'] = df_copy.index.map(dict_ref)
    df_cleaned = df_copy[df.similarity.notna()]
    df_sorted = df_cleaned.sort_values(by='similarity', ascending=False)
    return df_sorted
```
Are the results the same? Let's take a look at our example movie Mission: Impossible II:
```python
recommend('Mission: Impossible II')
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tmdb_id</th>
      <th>imdb_id</th>
      <th>title</th>
      <th>budget</th>
      <th>revenue</th>
      <th>release_date</th>
      <th>release_year</th>
      <th>genres</th>
      <th>overview</th>
      <th>genre_list</th>
      <th>overview_lemma</th>
      <th>Key_Words</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2769</th>
      <td>956</td>
      <td>tt0317919</td>
      <td>Mission: Impossible III</td>
      <td>150000000</td>
      <td>397850012</td>
      <td>2006-05-03</td>
      <td>2006</td>
      <td>Action, Adventure, Thriller</td>
      <td>Retired from active duty to train new IMF agen...</td>
      <td>[Action,  Adventure,  Thriller]</td>
      <td>Retired from active duty to train new IMF agen...</td>
      <td>[mission, call, back, retired, train, new, imf...</td>
      <td>0.308607</td>
    </tr>
    <tr>
      <th>7066</th>
      <td>177677</td>
      <td>tt2381249</td>
      <td>Mission: Impossible - Rogue Nation</td>
      <td>150000000</td>
      <td>682330139</td>
      <td>2015-07-23</td>
      <td>2015</td>
      <td>Action, Adventure</td>
      <td>Ethan and team take on their most impossible m...</td>
      <td>[Action,  Adventure]</td>
      <td>Ethan and team take on their most impossible m...</td>
      <td>[imf, syndicate, destroy, team, take, highlysk...</td>
      <td>0.271052</td>
    </tr>
    <tr>
      <th>9059</th>
      <td>458897</td>
      <td>tt5033998</td>
      <td>Charlie's Angels</td>
      <td>48000000</td>
      <td>73279888</td>
      <td>2019-11-14</td>
      <td>2019</td>
      <td>Action, Adventure, Comedy</td>
      <td>When a systems engineer blows the whistle on a...</td>
      <td>[Action,  Adventure,  Comedy]</td>
      <td>When a system engineer blow the whistle on a d...</td>
      <td>[system, engineer, blow, line, across, protect...</td>
      <td>0.216225</td>
    </tr>
    <tr>
      <th>8541</th>
      <td>353081</td>
      <td>tt4912910</td>
      <td>Mission: Impossible - Fallout</td>
      <td>178000000</td>
      <td>791017452</td>
      <td>2018-07-13</td>
      <td>2018</td>
      <td>Action, Adventure</td>
      <td>When an IMF mission ends badly, the world is f...</td>
      <td>[Action,  Adventure]</td>
      <td>When an IMF mission end badly the world be fac...</td>
      <td>[time, hunt, loyalty, assassin, world, race, f...</td>
      <td>0.212512</td>
    </tr>
    <tr>
      <th>5194</th>
      <td>52451</td>
      <td>tt1509767</td>
      <td>The Three Musketeers</td>
      <td>75000000</td>
      <td>132274484</td>
      <td>2011-08-31</td>
      <td>2011</td>
      <td>Action, Adventure, Thriller</td>
      <td>The hot-headed young D'Artagnan along with thr...</td>
      <td>[Action,  Adventure,  Thriller]</td>
      <td>The hotheaded young DArtagnan along with three...</td>
      <td>[engulf, europe, hotheaded, young, dartagnan, ...</td>
      <td>0.198898</td>
    </tr>
    <tr>
      <th>5138</th>
      <td>56292</td>
      <td>tt1229238</td>
      <td>Mission: Impossible - Ghost Protocol</td>
      <td>145000000</td>
      <td>694713380</td>
      <td>2011-12-07</td>
      <td>2011</td>
      <td>Action, Adventure, Thriller</td>
      <td>Ethan Hunt and his team are racing against tim...</td>
      <td>[Action,  Adventure,  Thriller]</td>
      <td>Ethan Hunt and his team be race against time t...</td>
      <td>[bombing, force, disavow, kremlin, stop, ethan...</td>
      <td>0.197245</td>
    </tr>
    <tr>
      <th>6204</th>
      <td>72710</td>
      <td>tt1517260</td>
      <td>The Host</td>
      <td>44000000</td>
      <td>63327201</td>
      <td>2013-03-22</td>
      <td>2013</td>
      <td>Action, Adventure, Romance, Science Fiction, T...</td>
      <td>A parasitic alien soul is injected into the bo...</td>
      <td>[Action,  Adventure,  Romance,  Science Fictio...</td>
      <td>A parasitic alien soul be inject into the body...</td>
      <td>[melanie, stryder, instead, inject, body, carr...</td>
      <td>0.195180</td>
    </tr>
    <tr>
      <th>4807</th>
      <td>46528</td>
      <td>tt1032751</td>
      <td>The Warrior's Way</td>
      <td>42000000</td>
      <td>11087569</td>
      <td>2010-12-02</td>
      <td>2010</td>
      <td>Action, Adventure, Fantasy, Thriller, Western</td>
      <td>A warrior-assassin is forced to hide in a smal...</td>
      <td>[Action,  Adventure,  Fantasy,  Thriller,  Wes...</td>
      <td>A warriorassassin be force to hide in a small ...</td>
      <td>[mission, hide, refuse, american, badlands, fo...</td>
      <td>0.187523</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>15074</td>
      <td>tt0283160</td>
      <td>Extreme Ops</td>
      <td>40000000</td>
      <td>10959475</td>
      <td>2002-11-27</td>
      <td>2002</td>
      <td>Action, Adventure, Drama, Thriller</td>
      <td>While filming an advertisement, some extreme s...</td>
      <td>[Action,  Adventure,  Drama,  Thriller]</td>
      <td>While film an advertisement some extreme sport...</td>
      <td>[advertisement, terrorist, film, group, extrem...</td>
      <td>0.187523</td>
    </tr>
    <tr>
      <th>5227</th>
      <td>50456</td>
      <td>tt0993842</td>
      <td>Hanna</td>
      <td>30000000</td>
      <td>63782078</td>
      <td>2011-04-07</td>
      <td>2011</td>
      <td>Action, Adventure, Thriller</td>
      <td>A 16-year-old girl raised by her father to be ...</td>
      <td>[Action,  Adventure,  Thriller]</td>
      <td>A 16yearold girl raise by her father to be the...</td>
      <td>[mission, across, europe, tracked, dispatch, h...</td>
      <td>0.184428</td>
    </tr>
    <tr>
      <th>7567</th>
      <td>258489</td>
      <td>tt0918940</td>
      <td>The Legend of Tarzan</td>
      <td>180000000</td>
      <td>356743061</td>
      <td>2016-06-06</td>
      <td>2016</td>
      <td>Action, Adventure</td>
      <td>Tarzan, having acclimated to life in London, i...</td>
      <td>[Action,  Adventure]</td>
      <td>Tarzan have acclimate to life in London be cal...</td>
      <td>[investigate, call, back, acclimate, tarzan, m...</td>
      <td>0.180702</td>
    </tr>
    <tr>
      <th>989</th>
      <td>3132</td>
      <td>tt0280486</td>
      <td>Bad Company</td>
      <td>70000000</td>
      <td>65977295</td>
      <td>2002-06-07</td>
      <td>2002</td>
      <td>Action, Adventure, Comedy, Thriller</td>
      <td>When a Harvard-educated CIA agent is killed du...</td>
      <td>[Action,  Adventure,  Comedy,  Thriller]</td>
      <td>When a Harvardeducated CIA agent be kill durin...</td>
      <td>[kill, twin, brother, harvardeducated, cia, ag...</td>
      <td>0.180702</td>
    </tr>
    <tr>
      <th>6469</th>
      <td>699220</td>
      <td>tt6703928</td>
      <td>A Fool's Paradise</td>
      <td>0</td>
      <td>0</td>
      <td>2013-05-04</td>
      <td>2013</td>
      <td>Action, Thriller</td>
      <td>James Bond is sent on a mission to investigate...</td>
      <td>[Action,  Thriller]</td>
      <td>James Bond be send on a mission to investigate...</td>
      <td>[mission, send, investigate, michael, kristato...</td>
      <td>0.179284</td>
    </tr>
    <tr>
      <th>8673</th>
      <td>399248</td>
      <td>tt4669264</td>
      <td>Beirut</td>
      <td>0</td>
      <td>7258534</td>
      <td>2018-04-11</td>
      <td>2018</td>
      <td>Action, Drama, Thriller</td>
      <td>In 1980s Beirut, Mason Skiles is a former U.S....</td>
      <td>[Action,  Drama,  Thriller]</td>
      <td>In 1980s Beirut Mason Skiles be a former US di...</td>
      <td>[mission, former, us, diplomat, call, back, ci...</td>
      <td>0.176547</td>
    </tr>
    <tr>
      <th>4745</th>
      <td>39514</td>
      <td>tt1245526</td>
      <td>RED</td>
      <td>58000000</td>
      <td>71664962</td>
      <td>2010-10-13</td>
      <td>2010</td>
      <td>Action, Adventure, Comedy, Crime, Thriller</td>
      <td>When his peaceful life is threatened by a high...</td>
      <td>[Action,  Adventure,  Comedy,  Crime,  Thriller]</td>
      <td>When his peaceful life be threaten by a highte...</td>
      <td>[peaceful, life, uncover, old, team, assailant...</td>
      <td>0.176227</td>
    </tr>
    <tr>
      <th>8054</th>
      <td>284053</td>
      <td>tt3501632</td>
      <td>Thor: Ragnarok</td>
      <td>180000000</td>
      <td>853977126</td>
      <td>2017-10-25</td>
      <td>2017</td>
      <td>Action, Adventure, Comedy, Fantasy</td>
      <td>Thor is imprisoned on the other side of the un...</td>
      <td>[Action,  Adventure,  Comedy,  Fantasy]</td>
      <td>Thor be imprison on the other side of the univ...</td>
      <td>[asgardian, civilization, end, destruction, th...</td>
      <td>0.176227</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1535</td>
      <td>tt0266987</td>
      <td>Spy Game</td>
      <td>115000000</td>
      <td>143049560</td>
      <td>2001-11-18</td>
      <td>2001</td>
      <td>Action, Crime, Thriller</td>
      <td>On the day of his retirement, a veteran CIA ag...</td>
      <td>[Action,  Crime,  Thriller]</td>
      <td>On the day of his retirement a veteran CIA age...</td>
      <td>[former, protg, die, arrest, international, sc...</td>
      <td>0.172516</td>
    </tr>
    <tr>
      <th>6221</th>
      <td>204082</td>
      <td>tt2312718</td>
      <td>Homefront</td>
      <td>22000000</td>
      <td>43058898</td>
      <td>2013-11-12</td>
      <td>2013</td>
      <td>Action, Thriller</td>
      <td>Phil Broker is a former DEA agent who has gone...</td>
      <td>[Action,  Thriller]</td>
      <td>Phil Broker be a former DEA agent who have go ...</td>
      <td>[school, recently, widow, event, cost, 9yearso...</td>
      <td>0.170367</td>
    </tr>
    <tr>
      <th>3316</th>
      <td>1620</td>
      <td>tt0465494</td>
      <td>Hitman</td>
      <td>24000000</td>
      <td>99965753</td>
      <td>2007-11-21</td>
      <td>2007</td>
      <td>Action, Crime, Drama, Thriller</td>
      <td>The best-selling videogame, Hitman, roars to l...</td>
      <td>[Action,  Crime,  Drama,  Thriller]</td>
      <td>The bestselling videogame Hitman roar to life ...</td>
      <td>[prey, international, intrigue, barrel, blaze,...</td>
      <td>0.169031</td>
    </tr>
    <tr>
      <th>8647</th>
      <td>446807</td>
      <td>tt5177088</td>
      <td>The Girl in the Spider's Web</td>
      <td>43000000</td>
      <td>17894345</td>
      <td>2018-10-25</td>
      <td>2018</td>
      <td>Action, Crime, Thriller</td>
      <td>In Stockholm, Sweden, hacker Lisbeth Salander ...</td>
      <td>[Action,  Crime,  Thriller]</td>
      <td>In Stockholm Sweden hacker Lisbeth Salander be...</td>
      <td>[exist, computer, engineer, stockholm, sweden,...</td>
      <td>0.169031</td>
    </tr>
  </tbody>
</table>
</div>

### **It worked! Time do build a flask app and deploy.**

## Build a Flask App to Serve Recommendations
Since front end/HTML isn't my forte, I'll just disclose that I used [Bootstrap](https://getbootstrap.com/docs/4.4/getting-started/download/) to design the front end. IMO, it is the quickest way to get up and running with having a functioning, responsive site without worrying about customizing heavily or advanced frameworks.

I'll just showcase the app.py file to build the Flask app. In it, I do a few things:  
- Use Flask's render_template function to serve an HTML file, which takes variables and uses the Jinja2 templating language to help return results.  
- Build helper functions to:  
    - Populate a list of movies within the search bar  
    - Get the overview, IMDB ID and title of the movie requested to serve as a summary before recommendations  
    - And of course, the recommend function created earlier  
  
```python
from flask import Flask, render_template, request, Response
import pandas as pd
from helper import get_movies, choose, overview, imdb, recommend

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    movielist = get_movies()
    if request.method == 'GET':
        return render_template('index.html', recommendations = pd.DataFrame(columns=['title', 'ID', 'year', 'overview']), movielist=movielist, movie="", overview = "", imdb="")
    
    if request.method == 'POST':

        if request.form.get('submit') == 'search':
            movie = request.form.get('movie')
            if movie in movielist:
                recs = recommend(movie)[:20]
                return render_template('index.html', recommendations = recs, movielist=movielist, movie=movie, overview = overview(movie), imdb=imdb(movie))
            else:
                return render_template('error.html')
        elif request.form.get('submit') == 'random':
            movie = choose()
            recs = recommend(movie)[:20]
            return render_template('index.html', recommendations = recs, movielist=movielist, movie=movie, overview = overview(movie), imdb=imdb(movie))

        

if __name__ == '__main__':
	app.run(debug=True)
```

## After all this, I used Google Cloud Engine to host and serve this Flask app. Link is at the top. Any questions? [Contact](https://parthka.github.io/#contact) me!

import pandas as pd
import numpy as np
import scipy as sp

from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import svm
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import re
from sklearn.feature_extraction.text import CountVectorizer



def bagwords (df):
       
    
    listname = ['artist_genres', 'artist_name', 'album_name', 'song_name']
    
  
    dicname = {name: [] for name in listname}
    dicname['song_id'] = []
    
    listid = []
    
    dicdf = {}
    vol = {}
    print(df.shape)
 
        
    for name in listname:
        
        for idx in range(df[name].size):
            
            l = df[name][idx]
          
            r = names_to_words(l)
            
            dicname[name].append(r) 
            
      
            
        
        vectorizer = CountVectorizer(analyzer='word',max_features=30)
        feature = vectorizer.fit_transform(dicname[name]).toarray().tolist()
        vol[name] = vectorizer.get_feature_names()
        
        
        dicdf[name] = pd.DataFrame({name: feature}) 
        
        dicdf[name] = dicdf[name][name].apply(pd.Series)
        ### dicdf[name] already lose column tag name, but change into 0 1 2 
        
        l = len(dicdf[name].columns)
        
        dicdf[name].columns = [(name + '_' +vol[name][x]) for x in range(l)]
        
        print('good')
     
   
    
    for idx in range(df['song_id'].size):
        sid = df['song_id'][idx]
        listid.append(sid)
       
            
        dicdf['song_id'] = pd.DataFrame({'song_id':listid}) 

        
            
            
        
    result = pd.concat(dicdf.values(), axis =1)
    
    return result
    


def names_to_words(names):
    words = re.sub("[^a-zA-Z0-9]"," ",names).lower().split()
    
    words = [i for i in words if i not in set(stopwords.words("english"))]
    ## Need join as string for countvectorizer!
    return (" ".join(words))

def reduce_genres(gen):
    genre = re.sub("[^a-zA-Z0-9]"," ",gen).lower().split()
    genre = [i for i in genre if i not in set(stopwords.words("english"))]
    mode1 = str(stats.mode(genre)).split('[')[1].split(']')[0]
    return mode1
def generic_cleanup (df):
    df = df.dropna()#drops null values
    df = df.drop(['album_genres'],axis =1 )#column not needed
    df['explicit'] = df['explicit'].map( {True: 1, False: 0} ).astype(int)
    threshold = df['popularity'].quantile(0.8)
    df['class'] = df['popularity'].apply(lambda x: 1 if x >= threshold else 0) #takes 20% of all tracks
    df = df[(df.astype(str)['artist_genres'] != '[]')].reset_index()
    df['reduced_genres'] = df['artist_genres'].apply(lambda x: reduce_genres(x))
    df['year'] = [x.split('-')[0] for x in df['album_release_date']]
    return df
def merge ():
    warnings.filterwarnings('ignore')
    df = pd.read_csv('//Users/Ilya/PycharmProjects/SpotifySongAnalisis/data/1995.csv', sep='\t')
    df = generic_cleanup(df)
    df1 = bagwords(df)
    df = df.merge(df1, on='song_id', how='outer')
    return df
def train (df):
    Y = df['class'].values
    df = df.drop(['Unnamed: 0', 'song_id', 'artist_id', 'album_id', 'song_name',
                  'artist_name', 'album_name', 'type', 'artist_genres', 'album_release_date', 'popularity',
                  'class', 'index', 'reduced_genres'], axis=1)
    X = df.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_features=17)
    model.fit(X_train, Y_train)
    train_predict = model.predict(X_train)
    mse_train = mean_squared_error(Y_train, train_predict)
    print("training error:", mse_train)
    test_predict = model.predict(X_test)
    mse_test = mean_squared_error(Y_test, test_predict)
    print("test error:", mse_test)





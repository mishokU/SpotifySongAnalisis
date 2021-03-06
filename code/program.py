import collections
import os
import re
import time
import warnings
from tkinter import Label, Button, Tk, Scale, Entry, HORIZONTAL

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import spotipy
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from spotipy import SpotifyClientCredentials
from wordcloud import WordCloud

dataPath = '/Users/m.usov/PycharmProjects/SpotifySongAnalisis/data/'

years = ['1995', '1996', '1997', '1998', '1999', '2000', '2001',
         '2002', '2003', '2004', '2005', '2006', '2007', '2008',
         '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

num_tracks_per_query = 10000

start_analysis_year = 2013
end_analysis_year = 2014
year_step = 1

show = True

artist_genres_info = pd.DataFrame()

firstColumn = ['popularity',
               'song_id', 'artist_id', 'album_id',
               'song_name', 'artist_name', 'album_name',
               'explicit', 'disc_number', 'track_number']

secondColumn = ['song_id', 'uri',
                'tempo', 'type',
                'key', 'loudness',
                'mode', 'speechiness',
                'liveness', 'valence',
                'danceability', 'energy',
                'track_href', 'analysis_url',
                'duration_ms', 'time_signature',
                'acousticness', 'instrumentalness']

thirdColumn = ['artist_id', 'artist_genres', 'artist_popularity']

fourthColumn = ['album_id', 'album_genres', 'album_popularity', 'album_release_date']

tracks_limit = 50
loop_step = 50

albums_step = 20
time_sleep = 0.3
start_value = 0

g_search_type = 'track'


def getSpotify():
    client_id = '60ba58fe2d914022a2a43967d9217771'
    client_secret = '4837bd2dd3ad4cf0b1ab30e4696e8b2e'
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)  # spotify object to access API


def createFile(tracks, audios, artist_data, album_data, year):
    df1 = pd.DataFrame(tracks, columns=firstColumn)

    df2 = pd.DataFrame(audios, columns=secondColumn)

    df3 = pd.DataFrame(artist_data, columns=thirdColumn)

    df4 = pd.DataFrame(album_data, columns=fourthColumn)

    df = df1.merge(df2, on='song_id', how='outer').merge(df3, on='artist_id', how='outer').merge(df4, on='album_id',
                                                                                                 how='outer')
    print(df)
    filename = '../data/' + year + '.csv'

    df.to_csv(filename, sep=',', index=False)

    print('finish')
    print(year)


def readAllCsv(startYear, endYear):
    songsYearData = []
    for year in range(start_analysis_year, end_analysis_year, year_step):
        df = pd.read_csv(dataPath + str(year) + '.csv', on_bad_lines='skip')
        songsYearData.append(df)
    frame = pd.concat(songsYearData, axis=0, ignore_index=True)
    frame = frame.dropna()
    frame['year'] = [x.split('-')[0] for x in frame['album_release_date']]
    print(frame.columns)
    print(frame)

    start_time = time.time()
    drawPlots(frame, startYear, endYear)
    drawMostFrequentlyGenres(startYear, endYear)
    drawDominateGenresWords(frame, startYear, endYear)
    drawArtistPopularityByAlbumPopularity(frame)
    drawArtistPopularityBySongsCount(frame)
    print("--- %s seconds draw task ---" % (time.time() - start_time))


def readModel():
    start_time = time.time()
    df = merge()
    Y = df['class'].values
    df = df.drop(['song_id', 'artist_id', 'album_id', 'song_name', 'uri', 'track_href', 'analysis_url',
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
    drawImportantFeatures(df, model)
    print("--- %s seconds classifier task ---" % (time.time() - start_time))


def mean_data_generation(df, start, end):
    loudness_over_years = []
    energy_over_years = []
    valence_over_years = []
    acoustics_over_years = []
    instrumentalness_over_years = []
    track_number_over_years = []
    for year in range(start, end):
        df = pd.read_csv(dataPath + str(year) + '.csv', on_bad_lines='skip')
        loudness_over_years.append(df['loudness'].mean())
        energy_over_years.append(df['energy'].mean())
        valence_over_years.append(df['valence'].mean())
        acoustics_over_years.append(df['acousticness'].mean())
        instrumentalness_over_years.append(df['instrumentalness'].mean())
        track_number_over_years.append(df['track_number'].mean())

    # Energy data frame
    df_energy = generate_dataframe(energy_over_years, 'energy', start, end)
    # Loudness data frame
    df_loudness = generate_dataframe(loudness_over_years, 'loudness', start, end)
    # Valence data frame
    df_valence = generate_dataframe(valence_over_years, 'valence', start, end)
    # Acousticness data frame
    df_acousticness = generate_dataframe(acoustics_over_years, 'acousticness', start, end)
    # Instrumentalness data frame
    df_instrumentalness = generate_dataframe(instrumentalness_over_years, 'instrumentalness', start, end)
    # Ttrack_numbers data frame
    df_track_numbers = generate_dataframe(track_number_over_years, 'track_number', start, end)
    return {
        'Energy': df_energy,
        'Loudness': df_loudness,
        'Valence': df_valence,
        'Acousticness': df_acousticness,
        'Instrumentalness': df_instrumentalness,
        'Track number': df_track_numbers
    }


def generate_dataframe(data, key, start, end):
    values = list(range(start, end, 1))
    print(values)
    df_tmp = pd.DataFrame(data, index=[str(int) for int in values], columns=['key'])
    # Converting the index as date
    df_tmp.index = pd.to_datetime(df_tmp.index)
    return df_tmp


def drawPlots(df, start, end):
    warnings.filterwarnings(action='once')
    mean_map = mean_data_generation(df, start, end)
    # Create the plot space upon which to plot the data
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 7))
    color_map = {
        'Energy': 'purple',
        'Loudness': 'pink',
        'Valence': 'green',
        'Acousticness': 'cyan',
        'Instrumentalness': 'lime',
        'Track number': 'coral'
    }

    r = 0
    c = 0
    print(mean_map)
    print(mean_map.keys())
    for key in mean_map.keys():
        sns.set()
        data_frame = mean_map[key]
        # Add the x-axis and the y-axis to the plot
        ax[r][c].plot(data_frame.index.values, data_frame, '-o', color=color_map[key])
        # Set title and labels for axes
        ax[r][c].set(xlabel="Date", ylabel=key, title=key + " mean comparison over 1995 to 2017")
        ax[r][c].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        c = c + 1
        if c >= 2:
            r = r + 1
            c = 0
        if r > 2:
            break

    fig.tight_layout()
    if show:
        plt.show()


def drawMostFrequentlyGenres(start, end):
    count = collections.Counter()

    for year in range(start, end):
        df = pd.read_csv(dataPath + str(year) + '.csv', on_bad_lines='skip')
        artist_genres_info[str(year)] = (df.loc[:, 'artist_genres'])

    x = (artist_genres_info.loc[:, str(year)])
    for elem in x:
        v = (elem[1:-1].split(", "))
        v = [year[1:-1] for year in v]
        count.update(v)

    df = pd.DataFrame(count.most_common(25), columns=['genre', 'count'])
    ax = sns.barplot(x="genre", y="count", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    if show:
        plt.show()


def drawDominateGenresWords(df, start, end):
    for year in range(start, end):
        artist_genres_info[str(year)] = (df.loc[:, 'artist_genres'])

    x = (artist_genres_info.loc[:, str(start)])
    counter = collections.Counter()
    for elem in x:
        v = (elem[1:-1].split(", "))
        v = [year[1:-1] for year in v]
        counter.update(v)
    wc = WordCloud(width=500, height=500, background_color='white', min_font_size=10)
    wc.generate_from_frequencies(counter)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    if show:
        plt.show()


def drawImportantFeatures(df, model):
    importance = model.feature_importances_
    df.drop_duplicates(inplace=True)
    dfi = pd.DataFrame(importance, index=df.columns, columns=["Importance"])
    dfi = dfi.sort_values(['Importance'], ascending=False)
    # showing those features which are at least significant.
    df_plot = dfi[dfi.Importance > 0.01]
    # Let's visualize the importance
    sns.set()
    plt.figure(figsize=(15, 10))
    plt.barh(df_plot.index, df_plot['Importance'], align='center', alpha=0.5,
             color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.xlabel("Importance")
    plt.ylabel("Audio features")
    plt.title("Importance of audio features")

    plt.tight_layout()
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords)
    weights = {}
    idx = 0
    arr = np.array(dfi)
    keys = np.array(dfi.index)

    idx = 0
    for i in keys:
        weights[i] = arr[idx][0]
        idx += 1

    wc.generate_from_frequencies(weights)
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)

    if show:
        plt.show()


def drawArtistPopularityByAlbumPopularity(frame):
    sns.scatterplot(data=frame, x="artist_popularity", y="album_popularity", hue="popularity")


def drawArtistPopularityBySongsCount(frame):
    plt.figure(figsize=(10, 10))
    sns.countplot(x="artist_popularity", data=frame)
    plt.ylabel('Songs Count')
    plt.xticks(rotation=45, ha='right')
    if show:
        plt.show()


def extract_songs(startYear, endYear, tracks_per_year):
    # Query and request from API are different!
    # Number of track query need to make
    start_time = time.time()
    for year in range(startYear, endYear):

        tracks = []
        song_ids = []
        artist_ids = []
        album_ids = []

        audios = []
        artist_data = []
        album_data = []

        for index in range(start_value, int(tracks_per_year), loop_step):
            searchRequest(str(year), g_search_type, tracks_limit, index, tracks, song_ids, artist_ids, album_ids)
            print(('\n>> this is No ' + str(index) + ' search End '))
            # Limit API requests to at most 3ish calls / second
            time.sleep(time_sleep)

        ## spotify API "search" option vs here track/audiofeature query
        for index in range(start_value, len(song_ids), loop_step):
            getAudios(song_ids[index: index + loop_step], audios)
            time.sleep(time_sleep)

        for index in range(start_value, len(artist_ids), loop_step):
            getArtists(artist_ids[index: index + loop_step], artist_data)
            time.sleep(time_sleep)

        for index in range(start_value, len(album_ids), albums_step):
            getAlbums(album_ids[index: index + albums_step], album_data)
            time.sleep(time_sleep)

        createFile(tracks, audios, artist_data, album_data, str(year))
    print("--- %s seconds ---" % (time.time() - start_time))


def searchRequest(year, search_type, results_limit, results_offset, tracksFile, song_ids, artist_ids, album_ids):
    try:
        result = getSpotify().search(q='year:' + year, type=search_type, limit=results_limit, offset=results_offset)

        items = result['tracks']['items']

        for item in items:

            if item['id'] not in song_ids:
                song_ids.append(item['id'])

            if item['artists'][0]['id'] not in artist_ids:
                artist_ids.append(item['artists'][0]['id'])

            if item['album']['id'] not in album_ids:
                album_ids.append(item['album']['id'])

            k = [item['popularity'],

                 item['id'],
                 item['artists'][0]['id'],
                 item['album']['id'],

                 item['name'].replace(',', ''),
                 item['artists'][0]['name'].replace(',', ''),
                 item['album']['name'].replace(',', ''),

                 item['explicit'],
                 item['disc_number'],
                 item['track_number']]

            tracksFile.append(k)
    except:
        ValueError


def getAudios(songIds, audioFile):
    track_ids = ','.join(songIds)

    audioFeatures = getSpotify().audio_features(tracks=track_ids)

    try:
        for features in audioFeatures:
            columns = [features['id'], features['uri'],
                       features['tempo'], features['type'],
                       features['key'], features['loudness'],
                       features['mode'], features['speechiness'],
                       features['liveness'], features['valence'],
                       features['danceability'], features['energy'],
                       features['track_href'], features['analysis_url'],
                       features['duration_ms'], features['time_signature'],
                       features['acousticness'], features['instrumentalness']]
            audioFile.append(columns)
    except:
        ValueError


def getAlbums(album_ids, album_data):
    result = getSpotify().albums(albums=album_ids)

    albums = result['albums']

    try:
        for album in albums:
            k = [album['id'],
                 album['genres'],
                 album['popularity'],
                 album['release_date']]

            album_data.append(k)

    except:
        ValueError


def getArtists(artist_ids, artist_data):
    result = getSpotify().artists(artist_ids)  # search query
    artists = result['artists']

    try:
        for artist in artists:
            k = [artist['id'], artist['genres'], artist['popularity']]
            artist_data.append(k)

    except:
        ValueError


class SongAnalysisUi:
    years = ['1995', '1996', '1997', '1998', '1999', '2000', '2001',
             '2002', '2003', '2004', '2005', '2006', '2007', '2008',
             '2009', '2010', '2011', '2012', '2013', '2014', '2015',
             '2016', '2017'
             ]

    num_tracks_per_query = 10000

    window = Tk()
    start = Scale(window, from_=years[0], length=800, to=years[-1], tickinterval=1, orient=HORIZONTAL)
    end = Scale(window, from_=years[0], length=800, to=years[-1], tickinterval=1, orient=HORIZONTAL)
    entry1 = Entry(window)

    def __init__(self, years, tracks_per_year):
        self.years = years
        self.num_tracks_per_query = tracks_per_year

    def createMainWindow(self):
        self.window.title("?????????? ???????????????????? ?? ???????????????????? PythonRu")
        self.window.geometry('1000x450')
        self.initMethods()
        self.window.mainloop()

    def initMethods(self):
        self.createWelcomeLabel()
        self.createYearsSlider()
        self.createTakeSongsPerYear()
        self.createButton()

    def createYearsSlider(self):
        labelFrom = Label(self.window, anchor='w', text="???? ???????????? ????????", font=("Arial Bold", 18))
        labelFrom.grid(column=0, row=1)
        self.start.set(0)
        self.start.grid(column=0, row=2)
        labelTo = Label(self.window, text="???? ???????????? ????????", font=("Arial Bold", 18))
        labelTo.grid(column=0, row=3)
        self.end.set(self.years[-1])
        self.end.grid(column=0, row=4)

    def createWelcomeLabel(self):
        welcomeLabel = Label(self.window, text="?????? ???? ???????????? ?????????????? ?? ???????????? ???? ?????????? ?????? ???????????? ???????????? ??????????",
                             font=("Arial Bold", 18))
        welcomeLabel.grid(column=0, row=0)

    def createTakeSongsPerYear(self):
        welcomeLabel = Label(self.window, text="?????????? ?? ??????",
                             font=("Arial Bold", 18))
        welcomeLabel.grid(column=0, row=5)
        self.entry1.grid(column=0, row=6)

    def createButton(self):
        analysisButton = Button(self.window, text="?????????????? ????????????", command=self.startAnalysis)
        analysisButton.grid(column=0, row=7)

    def startAnalysis(self):
        if self.checkSliders() and self.checkSongsPerYear():
            if os.path.exists('../data'):
                extract_songs(self.getStartYear(), self.getEndYear(), tracks_per_year=self.entry1.get())
            readAllCsv(self.getStartYear(), self.getEndYear())
            readModel()

    def getStartYear(self):
        return self.start.get()

    def getEndYear(self):
        return self.end.get()

    def checkSliders(self):
        return True
        print("sliders good")

    def checkSongsPerYear(self):
        return True
        print("songs per year")

def bagwords(df):
    list_name = ['artist_genres', 'artist_name', 'album_name', 'song_name']
    final_dict = {name: [] for name in list_name}
    final_dict['song_id'] = []
    listid = []
    dicdf = {}
    val = {}
    print(df.shape)
    for name in list_name:
        for idx in range(df[name].size):
            original_val = df[name][idx]
            changed_val = names_to_words(original_val)
            final_dict[name].append(changed_val)

        vectorizer = CountVectorizer(analyzer='word', max_features=30)
        feature = vectorizer.fit_transform(final_dict[name]).toarray().tolist()
        val[name] = vectorizer.get_feature_names()
        dicdf[name] = pd.DataFrame({name: feature})
        dicdf[name] = dicdf[name][name].apply(pd.Series)
        ### dicdf[name] already lose column tag name, but change into 0 1 2
        length = len(dicdf[name].columns)
        dicdf[name].columns = [(name + '_' + val[name][x]) for x in range(length)]
        # print('good')
    for idx in range(df['song_id'].size):
        sid = df['song_id'][idx]
        listid.append(sid)
        dicdf['song_id'] = pd.DataFrame({'song_id': listid})
    result = pd.concat(dicdf.values(), axis=1)
    return result


def names_to_words(names):
    words = re.sub("[^a-zA-Z0-9]", " ", names).lower().split()
    words = [i for i in words if i not in set(stopwords.words("english"))]
    return (" ".join(words))  # joined for use with countvectorizer


def reduce_genres(gen):
    genre = re.sub("[^a-zA-Z0-9]", " ", gen).lower().split()
    genre = [i for i in genre if i not in set(stopwords.words("english"))]
    most_frequent = collections.Counter(genre).most_common(1)[0][0]
    most_frequent = "'" + most_frequent + "'"
    return most_frequent


def generic_cleanup(df):
    df = df.dropna()  # drops null values
    df = df.drop(['album_genres'], axis=1)  # column not needed
    df['explicit'] = df['explicit'].map({True: 1, False: 0}).astype(int)
    threshold = df['popularity'].quantile(0.8)
    df['class'] = df['popularity'].apply(lambda x: 1 if x >= threshold else 0)  # takes 20% of all tracks
    df = df[(df.astype(str)['artist_genres'] != '[]')].reset_index()
    df['reduced_genres'] = df['artist_genres'].apply(lambda x: reduce_genres(x))
    df['year'] = [x.split('-')[0] for x in df['album_release_date']]
    return df


def merge():
    warnings.filterwarnings('ignore')
    df = pd.read_csv(dataPath + '2014.csv', error_bad_lines=False)
    df = generic_cleanup(df)
    df1 = bagwords(df)
    df = df.merge(df1, on='song_id', how='outer')
    return df


def train(df):
    Y = df['class'].values
    df = df.drop(['Unnamed: 0', 'song_id', 'artist_id', 'album_id', 'song_name', 'uri', 'track_href', 'analysis_url',
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
    return model

if __name__ == '__main__':
    window = SongAnalysisUi(years=years, tracks_per_year=num_tracks_per_query)
    window.createMainWindow()

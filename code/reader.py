import collections
import time
import warnings

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

import classifier
from constants import dataPath

start_analysis_year = 2013
end_analysis_year = 2014
year_step = 1

show = True

artist_genres_info = pd.DataFrame()


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
    df = classifier.merge()
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

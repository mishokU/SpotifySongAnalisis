import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import wordcloud
from wordcloud import WordCloud
import collections
import matplotlib.dates as mdates
from matplotlib import pyplot as plt

from PyLyrics import *
import billboard

from main import years
from columns import secondColumn

start_analysis_year = 2014  # 1995
end_analysis_year = 2015  # 2017
year_step = 1

artist_genres_info = pd.DataFrame()


def readAllCsv():
    # read your 1+ StreamingHistory files (depending on how extensive your streaming history is) into pandas dataframes

    songsYearData = []
    for year in range(start_analysis_year, end_analysis_year, year_step):
        df = pd.read_csv('/Users/m.usov/PycharmProjects/SpotifySongAnalisis/data/' + str(year) + '.csv',
                         error_bad_lines=False)
        songsYearData.append(df)

    # merge streaming dataframes
    frame = pd.concat(songsYearData, axis=0, ignore_index=True)

    ## Remove NaN
    frame = frame.dropna()

    ## Convert categorical features into numeric
    # frame['explicit'] = frame['explicit'].map({True: 1, False: 0}).astype(int)

    ## New 'year' feature
    frame['year'] = [x.split('-')[0] for x in frame['album_release_date']]
    # reduce_genres(frame)
    print(frame.columns)
    print(frame)

    drawPlots(frame)
    drawMostFrequentlyGenres(frame)
    drawDominateGenresWords(frame)
    drawArtistPopularityByAlbumPopularity(frame)


def fetch_data():
    year = 1969
    data = collections.defaultdict(list)
    for _ in range(6):
        chart_date = str(year)+"-07-15"
        chart = billboard.ChartData('hot-100',date=chart_date)
        track_count = 0
        for i,track in zip(range(100),chart):
            if track_count == 30: break
            artist = re.sub(r' Featuring.*','',track.artist)
            try:
                a = PyLyrics.getLyrics(artist,track.title)
            except ValueError as e:
                continue
            data[year].append((track.title,track.artist,a))
            track_count +=1
        year += 10
    return data

def generate_dataframe(data, key):
    df_tmp = pd.DataFrame(data, index=years, columns=['key'])
    # Converting the index as date
    df_tmp.index = pd.to_datetime(df_tmp.index)
    return df_tmp


def mean_data_generation(df):
    loudness_over_years = []
    energy_over_years = []
    valence_over_years = []
    acoustics_over_years = []
    instrumentalness_over_years = []
    track_number_over_years = []
    for _ in years:
        loudness_over_years.append(df['loudness'].mean())
        energy_over_years.append(df['energy'].mean())
        valence_over_years.append(df['valence'].mean())
        acoustics_over_years.append(df['acousticness'].mean())
        instrumentalness_over_years.append(df['instrumentalness'].mean())
        track_number_over_years.append(df['track_number'].mean())

    # Energy data frame
    df_energy = generate_dataframe(energy_over_years, 'energy')
    # Loudness data frame
    df_loudness = generate_dataframe(loudness_over_years, 'loudness')
    # Valence data frame
    df_valence = generate_dataframe(valence_over_years, 'valence')
    # Acousticness data frame
    df_acousticness = generate_dataframe(acoustics_over_years, 'acousticness')
    # Instrumentalness data frame
    df_instrumentalness = generate_dataframe(instrumentalness_over_years, 'instrumentalness')
    # Ttrack_numbers data frame
    df_track_numbers = generate_dataframe(track_number_over_years, 'track_number')
    return {
        'Energy': df_energy,
        'Loudness': df_loudness,
        'Valence': df_valence,
        'Acousticness': df_acousticness,
        'Instrumentalness': df_instrumentalness,
        'Track number': df_track_numbers
    }


def drawDominateGenresWords(df):
    for year in years:
        artist_genres_info[str(year)] = (df.loc[:, 'artist_genres'])

    x = (artist_genres_info.loc[:, '1995'])
    coun = collections.Counter()
    for elem in x:
        v = (elem[1:-1].split(", "))
        v = [year[1:-1] for year in v]
        coun.update(v)
    wc = WordCloud(width=500, height=500, background_color='white', min_font_size=10)
    wc.generate_from_frequencies(coun)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def drawGenderTrends():
    total_array = []
    for year in range(1969, 2020, 10):
        temp_array = []
        for title, artist, lyrics in data[year]:
            temp_array.append(artist_dict[artist])
        total_array.append(temp_array)

    total_array = np.array(total_array)
    df = pd.DataFrame(total_array.T, columns=['1969', '1979', '1989', '1999', '2009', '2019'])
    df_new = pd.melt(df)
    df_new = df_new.rename(columns={"variable": "Year", "value": "Gender"})
    sns.countplot(x="Year", hue="Gender", data=df_new)
    plt.show()


def drawMostFrequentlyGenres(df):
    count = collections.Counter()

    for year in years:
        artist_genres_info[str(year)] = (df.loc[:, 'artist_genres'])

    x = (artist_genres_info.loc[:, str(year)])
    for elem in x:
        v = ((elem[1:-1].split(", ")))
        v = [year[1:-1] for year in v]
        count.update(v)

    df = pd.DataFrame(count.most_common(25), columns=['genre', 'count'])
    ax = sns.barplot(x="genre", y="count", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.figure(figsize=(45, 35))
    plt.show()


def drawPlots(df):
    warnings.filterwarnings(action='once')
    mean_map = mean_data_generation(df)
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
    plt.show()


def drawArtistPopularityByAlbumPopularity(frame):
    sns.scatterplot(data=frame, x="artist_popularity", y="album_popularity", hue="popularity")


def drawArtistPopularityBySongsCount(frame):
    plt.figure(figsize=(10, 10))
    sns.countplot(x="artist_popularity", data=frame)
    plt.ylabel('Songs Count')
    plt.xticks(rotation=45, ha='right')
    plt.show()

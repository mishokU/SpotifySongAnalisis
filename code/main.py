import time
import auth
from createFile import createFile

years = ['1995', '1996', '1997', '1998', '1999', '2000', '2001',
         '2002', '2003', '2004', '2005', '2006', '2007', '2008',
         '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
num_tracks_per_query = 10000

# years = ['1995', '1996']
#
# num_tracks_per_query = 100

tracks_limit = 50
loop_step = 50

albums_step = 20
time_sleep = 0.3
start_value = 0

g_search_type = 'track'


def extract_songs():
    # Query and request from API are different!
    # Number of track query need to make
    start_time = time.time()
    for year in years:

        tracks = []
        song_ids = []
        artist_ids = []
        album_ids = []

        audios = []
        artist_data = []
        album_data = []

        for index in range(start_value, num_tracks_per_query, loop_step):
            searchRequest(year, g_search_type, tracks_limit, index, tracks, song_ids, artist_ids, album_ids)
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

        createFile(tracks, audios, artist_data, album_data, year)
    print("--- %s seconds ---" % (time.time() - start_time))


def searchRequest(year, search_type, results_limit, results_offset, tracksFile, song_ids, artist_ids, album_ids):
    try:
        result = auth.getSpotify().search(q='year:' + year, type=search_type, limit=results_limit,
                                          offset=results_offset)

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

    audioFeatures = auth.getSpotify().audio_features(tracks=track_ids)

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
    result = auth.getSpotify().albums(albums=album_ids)

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
    result = auth.getSpotify().artists(artist_ids)  # search query
    artists = result['artists']

    try:
        for artist in artists:
            k = [artist['id'], artist['genres'], artist['popularity']]
            artist_data.append(k)

    except:
        ValueError


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    extract_songs()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

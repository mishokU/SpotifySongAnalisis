import pandas as pd

import columns


def createFile(tracks, audios, artist_data, album_data, year):
    df1 = pd.DataFrame(tracks, columns=columns.firstColumn)

    df2 = pd.DataFrame(audios, columns=columns.secondColumn)

    df3 = pd.DataFrame(artist_data, columns=columns.thirdColumn)

    df4 = pd.DataFrame(album_data, columns=columns.fourthColumn)

    df = df1.merge(df2, on='song_id', how='outer').merge(df3, on='artist_id', how='outer').merge(df4, on='album_id', how='outer')
    print(df)
    filename = '../data/' + year + '.csv'

    df.to_csv(filename, sep=',', index=False)

    print('finish')
    print(year)

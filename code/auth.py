import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def getSpotify():
    client_id = '60ba58fe2d914022a2a43967d9217771'
    client_secret = '4837bd2dd3ad4cf0b1ab30e4696e8b2e'
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)  # spotify object to access API

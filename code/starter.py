from constants import years, num_tracks_per_query
from mainWindow import SongAnalysisUi

if __name__ == '__main__':

    window = SongAnalysisUi(years = years, tracks_per_year= num_tracks_per_query)
    window.createMainWindow()

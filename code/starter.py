
# Press the green button in the gutter to run the script.
import os

from reader import readModel, readAllCsv
from main import extract_songs

if __name__ == '__main__':
    if not os.path.exists('../data'):
        extract_songs()
    readAllCsv()
    readModel()

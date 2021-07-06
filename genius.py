import os
from pathlib import Path
from unicodedata import name
import lyricsgenius
import sys

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

class Genius:
    def __init__(self, client_access_token, timeout, retries, song_foldername = 'songs'):
        self.songs_foldername = song_foldername
        self._api_ = lyricsgenius.Genius(client_access_token, timeout=timeout, retries=retries)
        folder_exists = os.path.exists(Path(ROOT_DIR, self.songs_foldername))
        if not folder_exists:
            os.mkdir(Path(ROOT_DIR, self.songs_foldername))
    def scrape_artist(self, artist_name: str):
        artist = self._api_.search_artist(artist_name)
        print(f"stvaram file {artist.name.strip()}")
        f = open(self.songs_foldername + f'/{artist.name.strip()}.txt', 'w')  
        for song in artist.songs:
            f.write("\n" + '*'*50 + "\n")
            f.write(song.title)
            f.write(song.lyrics)
        f.close()
        print(f"done with {artist_name}")

#genius = Genius(sys.argv[1])
genius = Genius("Kkk5BYIapNxzTdhK1fxsRzOC6_-pohR6_as_Y5dWx8fho4JTbehcPRJeQIj34WMt", timeout = 15, retries = 30)

with open(Path(ROOT_DIR, 'data', 'zenske.txt')) as f:
    for rapper in [line.strip() for line in f.readlines()]:
        genius.scrape_artist(rapper)
"""
artist = genius.search_artist("Chief Keef", sort="title")
print(artist.songs)

with open(Path(ROOT_DIR, 'gpt2/data', 'rappers.txt')) as f:
    for rapper in [line.strip() for line in f.readlines()]:
        print("reper mi je {}".format(rapper))
        genius.scrape_artist(rapper)


genius = lyricsgenius.Genius("Kkk5BYIapNxzTdhK1fxsRzOC6_-pohR6_as_Y5dWx8fho4JTbehcPRJeQIj34WMt", timeout = 20, retries = 30)
artist = genius.search_artist("Chief Keef", max_songs=3, per_page=50)
print(len(artist.songs))
print(type(artist.songs[1]))
for kurac in artist.songs:
    print(kurac)
#print(artist.songs[)

"""
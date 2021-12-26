import os
from tkinter import *

from main import extract_songs

from reader import readAllCsv, readModel


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
        self.window.title("Добро пожаловать в приложение PythonRu")
        self.window.geometry('1000x450')
        self.initMethods()
        self.window.mainloop()

    def initMethods(self):
        self.createWelcomeLabel()
        self.createYearsSlider()
        self.createTakeSongsPerYear()
        self.createButton()

    def createYearsSlider(self):
        labelFrom = Label(self.window, anchor='w', text="От какого года", font=("Arial Bold", 18))
        labelFrom.grid(column=0, row=1)
        self.start.set(0)
        self.start.grid(column=0, row=2)
        labelTo = Label(self.window, text="До какого года", font=("Arial Bold", 18))
        labelTo.grid(column=0, row=3)
        self.end.set(self.years[-1])
        self.end.grid(column=0, row=4)

    def createWelcomeLabel(self):
        welcomeLabel = Label(self.window, text="Тут вы можете выбрать с какого по какой год делать анализ песен",
                             font=("Arial Bold", 18))
        welcomeLabel.grid(column=0, row=0)

    def createTakeSongsPerYear(self):
        welcomeLabel = Label(self.window, text="Песен в год",
                             font=("Arial Bold", 18))
        welcomeLabel.grid(column=0, row=5)
        self.entry1.grid(column=0, row=6)

    def createButton(self):
        analysisButton = Button(self.window, text="Сделать анализ", command=self.startAnalysis)
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

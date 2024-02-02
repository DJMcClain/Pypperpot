import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtWidgets import QWidget
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
from skimage import data, io, filters, feature
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gc
from PIL import Image
from IPython import display
import csv
import os
import scipy
import ImageData
import Mainapp

class ImFields(QMainWindow):
    def __init__(self):
        super(ImFields, self).__init__()
        self.central_widget = QWidget()
        self.layoutV1 = QVBoxLayout()
        self.layoutH1 = QHBoxLayout() # Load/Save Image
        self.layoutH2 = QHBoxLayout() # peak nums
        self.layoutH3 = QHBoxLayout() # max/min pixels
        self.layoutH3b = QHBoxLayout() # max/min pixels
        self.layoutH4 = QHBoxLayout() # Final Prompts
        self.layoutH5 = QHBoxLayout() # Window Frac
        self.layoutH5 = QHBoxLayout() # Window Frac
        self.layoutH6 = QHBoxLayout() # Offsets
        self.setCentralWidget(self.central_widget)

        loadImagePrompt = QPushButton('Load Image')
        saveImagePrompt = QPushButton('*Save Image*')
        loadImagePrompt.clicked.connect(ImageData.ImageReader.on_LoadIm_clicked)
        saveImagePrompt.clicked.connect(ImageData.ImageReader.on_SaveIm_clicked)

        
        FindPeaksPrompt = QPushButton('Find Peaks')
        ReducePrompt = QPushButton('Reduce Peak Number')
        FindPeaksPrompt.clicked.connect(ImageData.ImageReader.on_FindPeaks_clicked)
        ReducePrompt.clicked.connect(ImageData.ImageReader.on_Reduce_clicked)

        ImFields.xpeaksIn = x_peak_read()
        ImFields.ypeaksIn = y_peak_read()
        ImFields.xminIn = x_min_read()
        ImFields.yminIn = y_min_read()
        ImFields.xmaxIn = x_max_read()
        ImFields.ymaxIn = y_max_read()

        ImFields.winfrac = win_frac_read()
        ImFields.x_offsetIn = x_offset_read()
        ImFields.y_offsetIn = y_offset_read()

        self.central_widget.setLayout(self.layoutV1)
        # self.layoutV1.addLayout(self.layoutH1)
        # self.layoutH1.addWidget(loadImagePrompt)
        #self.layoutH1.addWidget(saveImagePrompt)

        self.layoutV1.addLayout(self.layoutH5)
        self.layoutH5.addWidget(QLabel('Window Fraction 1/'))
        self.layoutH5.addWidget(ImFields.winfrac)
        self.layoutV1.addLayout(self.layoutH6)
        self.layoutH6.addWidget(QLabel('X offset'))
        self.layoutH6.addWidget(ImFields.x_offsetIn)
        self.layoutH6.addWidget(QLabel('Y offset'))
        self.layoutH6.addWidget(ImFields.y_offsetIn)

        self.layoutV1.addLayout(self.layoutH4)
        # self.layoutH4.addWidget(FindPeaksPrompt)
        self.layoutH4.addWidget(ReducePrompt)
        
        self.layoutV1.addLayout(self.layoutH2)
        self.layoutH2.addWidget(QLabel('X-peaks'))
        self.layoutH2.addWidget(ImFields.xpeaksIn)
        self.layoutH2.addWidget(QLabel('Y-peaks'))
        self.layoutH2.addWidget(ImFields.ypeaksIn)

        self.layoutV1.addLayout(self.layoutH3)
        self.layoutH3.addWidget(QLabel('Min X'))
        self.layoutH3.addWidget(ImFields.xminIn)
        self.layoutH3.addWidget(QLabel('Max X'))
        self.layoutH3.addWidget(ImFields.xmaxIn)

        self.layoutV1.addLayout(self.layoutH3b)
        self.layoutH3b.addWidget(QLabel('Min Y'))
        self.layoutH3b.addWidget(ImFields.yminIn)
        self.layoutH3b.addWidget(QLabel('Max Y'))
        self.layoutH3b.addWidget(ImFields.ymaxIn)
        



class y_peak_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(2)
        self.setPlaceholderText("#")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

    def on_button_clicked():
        alert = QMessageBox()
        alert.setText('You clicked the button!')
        alert.exec()
    
    def return_pressed(self):
        print("Return pressed!")
        
    def selection_changed(self):
        return

    def text_changed(self, s):
        # print("Text changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("Text edited...")
        # print(s)
        return

class x_peak_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(2)
        self.setPlaceholderText("#")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

    
    def return_pressed(self):
        print("Return pressed!")
        
    def selection_changed(self):
        return

    def text_changed(self, s):
        # print("Text changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("Text edited...")
        # print(s)
        return

class x_min_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(4)
        self.setPlaceholderText("pix")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

    def on_button_clicked():
        alert = QMessageBox()
        alert.setText('You clicked the button!')
        alert.exec()
    
    def return_pressed(self):
        print("Return pressed!")
        
    def selection_changed(self):
        return

    def text_changed(self, s):
        # print("Text changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("Text edited...")
        # print(s)
        return

class x_max_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(4)
        self.setPlaceholderText("pix")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

    
    def return_pressed(self):
        print("Return pressed!")
        
    def selection_changed(self):
        return

    def text_changed(self, s):
        # print("Text changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("Text edited...")
        # print(s)
        return

class y_min_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(4)
        self.setPlaceholderText("pix")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)
    
    def return_pressed(self):
        print("Return pressed!")
        
    def selection_changed(self):
        return

    def text_changed(self, s):
        # print("Text changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("Text edited...")
        # print(s)
        return

class y_max_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(4)
        self.setPlaceholderText("pix")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

    def return_pressed(self):
        print("Return pressed!")
        
    def selection_changed(self):
        return

    def text_changed(self, s):
        # print("Text changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("Text edited...")
        # print(s)
        return
    
class win_frac_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(4)
        self.setPlaceholderText("2")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

    def return_pressed(self):
        print("Return pressed!")
        
    def selection_changed(self):
        return

    def text_changed(self, s):
        # print("Text changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("Text edited...")
        # print(s)
        return

class x_offset_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(4)
        self.setPlaceholderText("0")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)
    
    def return_pressed(self):
        print("Return pressed!")
        
    def selection_changed(self):
        return

    def text_changed(self, s):
        # print("Text changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("Text edited...")
        # print(s)
        return
    
class y_offset_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(4)
        self.setPlaceholderText("0")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)
    
    def return_pressed(self):
        print("Return pressed!")
        
    def selection_changed(self):
        return

    def text_changed(self, s):
        # print("Text changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("Text edited...")
        # print(s)
        return
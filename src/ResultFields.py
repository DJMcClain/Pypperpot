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
import Fitter

class ResFields(QMainWindow):
    def __init__(self):
        super(ResFields, self).__init__()
        self.central_widget = QFrame()
        self.layoutV1 = QVBoxLayout()
        self.layoutH1 = QHBoxLayout() # emitx
        self.layoutH2 = QHBoxLayout() # emity
        self.layoutH3 = QHBoxLayout() # alpha
        self.layoutH4 = QHBoxLayout() # beta
        self.layoutH5 = QHBoxLayout() # gamma
        
        self.setCentralWidget(self.central_widget)
        
        
        saveDataPrompt = QPushButton('*Save Data*')
        saveDataPrompt.clicked.connect(Fitter.PeakByPeakFits.on_SaveData_clicked)

        ResFields.xemit = QLabel('')#x_peak_read()
        ResFields.yemit = QLabel('')#y_peak_read()
        ResFields.xemiterr = QLabel('')#x_min_read()
        ResFields.yemiterr = QLabel('')#y_min_read()
        ResFields.xalph = QLabel('')#x_max_read()
        ResFields.yalph = QLabel('')
        ResFields.xbeta = QLabel('')
        ResFields.ybeta = QLabel('')
        ResFields.xgamm = QLabel('')
        ResFields.ygamm = QLabel('')
        Twiss = QLabel('Twiss Parameters')
        Twiss.setAlignment(QtCore.Qt.AlignCenter)
        self.central_widget.setLayout(self.layoutV1)
        self.central_widget.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.central_widget.setLineWidth(2)
        self.layoutV1.addLayout(self.layoutH1)
        self.layoutH1.addWidget(QLabel(f'\u03b5_x = '))
        self.layoutH1.addWidget(ResFields.xemit)
        self.layoutH1.addWidget(QLabel(f' \u00b1 '))
        self.layoutH1.addWidget(ResFields.xemiterr)
        self.layoutH1.addWidget(QLabel(f' \u03c0 mm mrad '))


        self.layoutV1.addLayout(self.layoutH2)
        self.layoutH2.addWidget(QLabel(f'\u03b5_y = '))
        self.layoutH2.addWidget(ResFields.yemit)
        self.layoutH2.addWidget(QLabel(f' \u00b1 '))
        self.layoutH2.addWidget(ResFields.yemiterr)
        self.layoutH2.addWidget(QLabel(f' \u03c0 mm mrad '))

        self.layoutV1.addWidget(Twiss)
        self.layoutV1.addLayout(self.layoutH3)
        self.layoutH3.addWidget(QLabel(f'\u03b1_x = '))
        self.layoutH3.addWidget(ResFields.xalph)
        self.layoutH3.addWidget(QLabel(f'\u03b1_y = '))
        self.layoutH3.addWidget(ResFields.yalph)

        self.layoutV1.addLayout(self.layoutH4)
        self.layoutH4.addWidget(QLabel(f'\u03b2_x = '))
        self.layoutH4.addWidget(ResFields.xbeta)
        self.layoutH4.addWidget(QLabel(f'\u03b2_y = '))
        self.layoutH4.addWidget(ResFields.ybeta)
        
        self.layoutV1.addLayout(self.layoutH5)
        self.layoutH5.addWidget(QLabel(f'\u03b3_x = '))
        self.layoutH5.addWidget(ResFields.xgamm)
        self.layoutH5.addWidget(QLabel(f'\u03b3_y = '))
        self.layoutH5.addWidget(ResFields.ygamm)

        self.layoutV1.addWidget(saveDataPrompt)

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

class y_min_read(QLineEdit):
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

class y_max_read(QLineEdit):
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


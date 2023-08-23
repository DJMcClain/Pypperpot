from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor, QIcon
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
import MaskFields
import ImageFields

path = os.getcwd()
# path = 'D:/Workspace/Images/'

class MaskWidget(QMainWindow):
    def __init__(self):
        super(MaskWidget, self).__init__()
        self.central_widget = QWidget()
        self.layoutV1 = QVBoxLayout() # File Params column
        self.layoutH4 = QHBoxLayout() # Save/Load Mask
        self.layoutH5 = QHBoxLayout() # Number of holes in mask
        self.layoutH6 = QHBoxLayout() # Hole Diameter
        self.layoutH7 = QHBoxLayout() # Hole Separation
        self.layoutH8 = QHBoxLayout() # Mask to Screen
        self.layoutH9 = QHBoxLayout() # Calibration
        self.setCentralWidget(self.central_widget)

        self.diamIn = hole_diam_read()
        self.numHoles = hole_num_read()
        self.sepIn = hole_sep_read()
        self.Mask2ScrnIn = mask2Scrn_read()
        self.Calibration = calib_read()

        loadMaskPrompt = QPushButton('Load Mask')
        saveMaskPrompt = QPushButton('Save Mask')
        loadMaskPrompt.clicked.connect(self.on_LoadMa_clicked)
        saveMaskPrompt.clicked.connect(self.on_SaveMa_clicked)

        self.central_widget.setLayout(self.layoutV1)
        self.layoutV1.addLayout(self.layoutH4)
        self.layoutH4.addWidget(QLabel('Mask'))
        self.layoutH4.addWidget(loadMaskPrompt)
        self.layoutH4.addWidget(saveMaskPrompt)

        self.layoutV1.addLayout(self.layoutH5)
        self.layoutH5.addWidget(QLabel('Number of Holes'))
        self.layoutH5.addWidget(self.numHoles)

        self.layoutV1.addLayout(self.layoutH6)
        self.layoutH6.addWidget(QLabel('Hole Diameter (mm)'))
        self.layoutH6.addWidget(self.diamIn)

        self.layoutV1.addLayout(self.layoutH7)
        self.layoutH7.addWidget(QLabel('Hole Separation (mm)'))
        self.layoutH7.addWidget(self.sepIn)

        self.layoutV1.addLayout(self.layoutH8)
        self.layoutH8.addWidget(QLabel('Mask to Screen Distance (mm)'))
        self.layoutH8.addWidget(self.Mask2ScrnIn) 

        self.layoutV1.addLayout(self.layoutH9)
        self.layoutH9.addWidget(QLabel('Calibration (pix/mm)'))
        self.layoutH9.addWidget(self.Calibration) 

    def on_LoadMa_clicked(self):
        loadMaskName = QFileDialog.getOpenFileName(caption="Load Mask", directory=path, filter="*.csv")
        print(loadMaskName[0])
        file = open(loadMaskName[0], 'r')
        data = list(csv.reader(file, delimiter=","))
        print(data[0][0])
        print(data[0][1])
        print(data[0][2])
        print(data[0][3])
        print(data[0][4])
        self.numHoles.setText(data[0][0])
        self.diamIn.setText(data[0][1])
        self.sepIn.setText(data[0][2])
        self.Mask2ScrnIn.setText(data[0][3])
        self.Calibration.setText(data[0][4])
        file.close()

    def on_SaveMa_clicked(self):
        saveMaskName = QFileDialog.getSaveFileName(caption="Save Mask", directrory=path, filter="*.csv")
        file = open(saveMaskName[0], 'w')
        writer =csv.writer(file)
        writer.writwrow([self.numHoles.text(), self.diamIn.text(), self.sepIn.text(),self.Mask2ScrnIn.text(), self.Calibration.text()])
        file.close()
class mask2Scrn_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
        self.setPlaceholderText("mm")
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
        # print("m2s changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("m2s edited...")
        # print(s)
        return

class calib_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
        self.setPlaceholderText("pix/mm")
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
        # print("calib changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("calib edited...")
        # print(s)
        return
class hole_diam_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        # self.setMaxLength(6)
        self.setPlaceholderText("mm")
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
        # print("hole diam changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("hole diam edited...")
        # print(s)
        return

class hole_sep_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
        self.setPlaceholderText("mm")
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
        return
    
    def text_edited(self, s):
        return

class hole_num_read(QLineEdit):
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
        # print("num changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("num edited...")
        # print(s)
        return
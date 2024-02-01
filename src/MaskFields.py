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
        self.layoutH1 = QHBoxLayout() # Load/Save Image
        self.layoutH4 = QHBoxLayout() # Save/Load Mask
        self.layoutH5 = QHBoxLayout() # Number of holes in mask
        self.layoutH6 = QHBoxLayout() # Hole Diameter
        self.layoutH7 = QHBoxLayout() # Hole Separation
        self.layoutH8 = QHBoxLayout() # Mask to Screen
        self.layoutH9 = QHBoxLayout() # Calibration
        self.setCentralWidget(self.central_widget)

        MaskWidget.diamIn = hole_diam_read()
        MaskWidget.numHoles = hole_num_read()
        MaskWidget.sepIn = hole_sep_read()
        MaskWidget.Mask2ScrnIn = mask2Scrn_read()
        MaskWidget.Calibration = calib_read()
        MaskWidget.hole_err = hole_err_read()
        MaskWidget.sigL = mask2Scrn_err_read()
        MaskWidget.puncert = calib_err_read()

        loadMaskPrompt = QPushButton('Load Mask')
        saveMaskPrompt = QPushButton('Save Mask')
        loadMaskPrompt.clicked.connect(self.on_LoadMa_clicked)
        saveMaskPrompt.clicked.connect(self.on_SaveMa_clicked)

        # loadImagePrompt = QPushButton('Load Image')
        # saveImagePrompt = QPushButton('*Save Image*')
        # loadImagePrompt.clicked.connect(ImageData.ImageReader.on_LoadIm_clicked)
        # saveImagePrompt.clicked.connect(ImageData.ImageReader.on_SaveIm_clicked)

        self.central_widget.setLayout(self.layoutV1)
        # self.layoutV1.addLayout(self.layoutH1)
        # self.layoutH1.addWidget(loadImagePrompt)
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
        self.layoutH7.addWidget(QLabel('+/-'))
        self.layoutH7.addWidget(self.hole_err)

        self.layoutV1.addLayout(self.layoutH8)
        self.layoutH8.addWidget(QLabel('Mask to Screen Distance (mm)'))
        self.layoutH8.addWidget(self.Mask2ScrnIn) 
        self.layoutH8.addWidget(QLabel('+/-'))
        self.layoutH8.addWidget(self.sigL) 

        self.layoutV1.addLayout(self.layoutH9)
        self.layoutH9.addWidget(QLabel('Calibration (pix/mm)'))
        self.layoutH9.addWidget(self.Calibration) 
        self.layoutH9.addWidget(QLabel('+/-'))
        self.layoutH9.addWidget(self.puncert) 

    def on_LoadMa_clicked(self):
        loadMaskName = QFileDialog.getOpenFileName(caption="Load Mask", directory=path, filter="*.csv")
        print(loadMaskName[0])
        file = open(loadMaskName[0], 'r')
        data = list(csv.reader(file, delimiter=","))
        # print(data[0][0])
        # print(data[0][1])
        # print(data[0][2])
        # print(data[0][3])
        # print(data[0][4])
        #print(data)
        self.numHoles.setText(data[0][0])
        self.diamIn.setText(data[0][1])
        self.sepIn.setText(data[0][2])
        self.Mask2ScrnIn.setText(data[0][3])
        self.Calibration.setText(data[0][4])
        try:
            self.hole_err.setText(data[2][0])
            self.sigL.setText(data[2][1])
            self.puncert.setText(data[2][2])
        except:
            print('no errors in file')
            self.hole_err.setText('')
            self.sigL.setText('')
            self.puncert.setText('')
        file.close()

    def on_SaveMa_clicked(self):
        saveMaskName = QFileDialog.getSaveFileName(caption="Save Mask", filter="*.csv")
        file = open(saveMaskName[0], 'w')
        writer =csv.writer(file)
        writer.writerow([self.numHoles.text(), self.diamIn.text(), self.sepIn.text(),self.Mask2ScrnIn.text(), self.Calibration.text()])
        writer.writerow([self.hole_err.text(), self.sigL.text(), self.puncert.text()])
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

class calib_err_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(8)
        self.setPlaceholderText("0.0005")
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

class hole_err_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(8)
        self.setPlaceholderText("0.00005")
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
    
class mask2Scrn_err_read(QLineEdit):

    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(8)
        self.setPlaceholderText("0.05")
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
class mask_width_read(QLineEdit):

    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(8)
        self.setPlaceholderText("0.1")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

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
class noise_read(QLineEdit):

    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(8)
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
        # print("m2s changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("m2s edited...")
        # print(s)
        return
class noise_uncert_read(QLineEdit):

    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(8)
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
        # print("m2s changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("m2s edited...")
        # print(s)
        return
class MaskSimDat(QWidget):
    def __init__(self):
        super(MaskSimDat, self).__init__()
        # self.central_widget = QWidget()
        self.layoutV1 = QVBoxLayout() #main layout
        self.layoutH1 = QHBoxLayout() # Mask Width
        self.layoutH2 = QHBoxLayout() # Noise Level
        # self.setCentralWidget(self.central_widget)

        MaskSimDat.maskwidth = mask_width_read()
        MaskSimDat.noiseLevel = noise_read()
        MaskSimDat.noiseUncert = noise_uncert_read()
        self.setLayout(self.layoutV1)
        self.layoutV1.addLayout(self.layoutH1)
        self.layoutH1.addWidget(QLabel('Mask Width (mm)'))
        self.layoutH1.addWidget(MaskSimDat.maskwidth)
        self.layoutV1.addLayout(self.layoutH2)
        self.layoutH2.addWidget(QLabel('Added Noise Level'))
        self.layoutH2.addWidget(MaskSimDat.noiseLevel)
        self.layoutH2.addWidget(QLabel('+/-'))
        self.layoutH2.addWidget(MaskSimDat.noiseUncert)
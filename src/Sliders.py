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
import Fitter

prombool = False
threshbool = False

class slider1(QWidget):#Threshold
    def __init__(self):
        QWidget.__init__(self)
        # self.setFixedWidth(40)#Height 50
        self.layoutS0 = QVBoxLayout()#Hbox
        self.setLayout(self.layoutS0)

        #Slider
        self.sl = QSlider(Qt.Vertical)#Horizontal
        self.sl.setMinimum(1)
        self.sl.setMaximum(10)
        self.sl.setValue(5)
        self.sl.setTickPosition(QSlider.TicksRight)#Below
        self.sl.setTickInterval(1)

        #Label
        self.slValue = QLineEdit()
        self.slValue.setAlignment(Qt.AlignCenter) 
        self.slValue.setFixedSize(30,20)
        self.slValue.setText(str(self.sl.value()))
        self.slValue.setMaxLength(2)
        
        #Connections
        self.slValue.returnPressed.connect(self.return_pressed)
        self.sl.valueChanged.connect(self.changeSlider)
        #Format
        self.layoutS0.addWidget(QLabel('Threshold'))
        self.layoutS0.addWidget(self.sl)
        self.layoutS0.addWidget(self.slValue)

    def return_pressed(self):
        s = int(self.slValue.text())
        self.sl.setValue(s)
        if threshbool == True:
            ImageData.ImageReader.changeThreshold(self.returnValue())

    def changeSlider(self):
        self.slValue.setText(str(self.sl.value()))
        if threshbool == True:
            ImageData.ImageReader.changeThreshold(self.returnValue())

    def returnValue(self):
        return(int(self.slValue.text()))
    
    def setValue(self, s):
        self.sl.setValue(s)
        self.slValue.setText(str(self.sl.value()))

class slider2(QWidget):#Prominence
    def __init__(self):
        QWidget.__init__(self)
        # self.setFixedWidth(40)
        self.layoutS0 = QVBoxLayout()
        self.setLayout(self.layoutS0)

        #Slider
        self.sl = QSlider(Qt.Vertical)
        self.sl.setMinimum(1)
        self.sl.setMaximum(20)
        self.sl.setValue(5)
        self.sl.setTickPosition(QSlider.TicksRight)
        self.sl.setTickInterval(5)
        # self.sl.setAlignment(Qt.AlignCenter)

        #Label
        self.slValue = QLineEdit()
        self.slValue.setAlignment(Qt.AlignCenter) 
        self.slValue.setFixedSize(30,20)
        self.slValue.setText(str(self.sl.value()))
        self.slValue.setMaxLength(2)
        
        #Connections
        self.slValue.returnPressed.connect(self.return_pressed)
        self.sl.valueChanged.connect(self.changeSlider)
        #Format
        self.layoutS0.addWidget(QLabel('Prominence'))
        self.layoutS0.addWidget(self.sl)
        self.layoutS0.addWidget(self.slValue)

    def return_pressed(self):
        s = int(self.slValue.text())
        self.sl.setValue(s)
        if prombool == True:
            ImageData.ImageReader.changeProminence(self.returnValue())

    def changeSlider(self):
        self.slValue.setText(str(self.sl.value()))
        if prombool == True:
            ImageData.ImageReader.changeProminence(self.returnValue())

    def returnValue(self):
        return(int(self.slValue.text()))
    
    def setValue(self, s):
        self.sl.setValue(s)
        self.slValue.setText(str(self.sl.value()))

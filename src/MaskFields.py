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
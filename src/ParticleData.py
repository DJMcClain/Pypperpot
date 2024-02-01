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
class PDataWidget(QMainWindow):
    def __init__(self):
        super(PDataWidget, self).__init__()
        self.central_widget = QWidget()
        self.layoutV1 = QVBoxLayout() # data column
        self.layoutH4 = QHBoxLayout() # Number of Particles
        self.layoutH5 = QHBoxLayout() # Mass Number
        self.layoutH6 = QHBoxLayout() # Kinetic Energy
        self.layoutH7 = QHBoxLayout() # Beam Radius
        self.layoutH8 = QHBoxLayout() # Beam Divergence
        self.layoutH9 = QHBoxLayout() # Alignment
        self.layoutH10 = QHBoxLayout() # generate/save
        self.setCentralWidget(self.central_widget)

        PDataWidget.simnumpartIn =simnum_part_read()
        PDataWidget.simmassNum = simMass_num_read()
        PDataWidget.simKinEIn = simKin_E_read()
        PDataWidget.simradIn = simRad_read()
        PDataWidget.simdivIn = simDiv_read()
        PDataWidget.simxalign = simX_align_read()
        PDataWidget.simyalign = simY_align_read()

        genDataPrompt = QPushButton('Generate Data')
        saveDataPrompt = QPushButton('Save Data')
        genDataPrompt.clicked.connect(self.on_GenDat_clicked)
        saveDataPrompt.clicked.connect(self.on_SaveDat_clicked)

        self.central_widget.setLayout(self.layoutV1)
        self.layoutV1.addLayout(self.layoutH4)
        self.layoutH4.addWidget(QLabel('Number of Particles'))
        self.layoutH4.addWidget(self.simnumpartIn)


        self.layoutV1.addLayout(self.layoutH5)
        self.layoutH5.addWidget(QLabel('Mass Number (amu)'))
        self.layoutH5.addWidget(self.simmassNum)

        self.layoutV1.addLayout(self.layoutH6)
        self.layoutH6.addWidget(QLabel('Kinetic Energy (eV)'))
        self.layoutH6.addWidget(self.simKinEIn)

        self.layoutV1.addLayout(self.layoutH7)
        self.layoutH7.addWidget(QLabel('Beam Radius (mm)'))
        self.layoutH7.addWidget(self.simradIn)
        # self.layoutH7.addWidget(QLabel('+/-'))
        # self.layoutH7.addWidget(self.hole_err)
        # self.layoutH7.addWidget(self.hole_err)

        self.layoutV1.addLayout(self.layoutH8)
        self.layoutH8.addWidget(QLabel('Beam Divergence (degrees)'))
        self.layoutH8.addWidget(self.simdivIn) 

        self.layoutV1.addLayout(self.layoutH9)
        self.layoutH9.addWidget(QLabel('X Alignment (mm)'))
        self.layoutH9.addWidget(self.simxalign) 
        self.layoutH9.addWidget(QLabel('Y Alignment'))
        self.layoutH9.addWidget(self.simyalign)

        self.layoutV1.addLayout(self.layoutH10)
        self.layoutH10.addWidget(genDataPrompt )
        self.layoutH10.addWidget(saveDataPrompt)
    def on_GenDat_clicked(self):
        print("grumbe")
        #Fill in with generate info

    def on_SaveDat_clicked(self):
        saveDatName = QFileDialog.getSaveFileName(caption="Save Data", filter="*.csv")
        file = open(saveDatName[0], 'w')
        # writer =csv.writer(file)
        # writer.writerow([self.numHoles.text(), self.diamIn.text(), self.sepIn.text(),self.Mask2ScrnIn.text(), self.Calibration.text()])
        # writer.writerow([self.hole_err.text(), self.sigL.text(), self.puncert.text()])
        file.close()
class simRad_read(QLineEdit):
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

class simDiv_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
        self.setPlaceholderText("degrees")
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
class simnum_part_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        # self.setMaxLength(6)
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
        # print("hole diam changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("hole diam edited...")
        # print(s)
        return

class simKin_E_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
        self.setPlaceholderText("eV")
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

class simMass_num_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(2)
        self.setPlaceholderText("amu")
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

class simY_align_read(QLineEdit):
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
        # print("calib changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("calib edited...")
        # print(s)
        return

class simX_align_read(QLineEdit):
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
    

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

class Handfitting(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, Xprojdf, Yprojdf, imgData, threshold, d, x3s, y3s):
        super().__init__()
        self.Xprojdf = Xprojdf
        self.Yprojdf = Yprojdf
        ImageData.imgData = imgData
        ImageData.ImageReader.threshold = threshold
        self.d = d
        self.x3s = x3s
        self.y3s = y3s
        self.setWindowTitle('PYpperpot 2.0 - Hand Fitting')
        self.setWindowIcon(QIcon('mrsPepper.png'))

        # self.setLayout(layout)
        

######Complete classes
        self.integral = IntRead()
        self.peakMean = MeanRead()
        self.peakWidth = WidthRead()
        self.peakOffset = OffsetRead()
        self.projectionNum = CustomSlider()
        self.num_peaks = HFNumPeaksRead()
        self.fit_peak = FitPeakRead()
        self.fit = QPushButton('Fit')
        self.mainwindow = Mainapp()
        ImageData.ImageReader.plota = PlotWidget()
        ImageData.ImageReader.plota.setBackground('w')

        # self.central_widget = QWidget() # A QWidget to work as Central Widget
        self.layoutH0 = QHBoxLayout() # Main window
        self.layoutV0 = QVBoxLayout() 
        self.layoutV1 = QVBoxLayout() 
        self.layoutH1 = QHBoxLayout()#Direction x/y
        self.layoutH2 = QHBoxLayout()#Projection num
        self.layoutH3 = QHBoxLayout()# num peaks here
        self.layoutH4 = QHBoxLayout()#peak num add another?
        self.layoutH5 = QHBoxLayout()#int input
        self.layoutH6 = QHBoxLayout()#mean input
        self.layoutH7 = QHBoxLayout()#width input
        self.layoutH8 = QHBoxLayout()#offset input
        self.fit.clicked.connect(self.HandFitting)
        self.ppen = pg.mkPen(color=(0, 0, 150),width = 2)
        self.fpen = pg.mkPen(color=(255, 0, 0),width = 2)
        self.hfpen = pg.mkPen(color = (0,0,0), width = 2)

# creating a push button
        self.xybutton = QPushButton('X', self)
        # setting checkable to true
        self.xybutton.setCheckable(True)
        # setting calling method by button
        self.xybutton.clicked.connect(self.changeState)
        # setting default color of button to light-grey
        self.xybutton.setStyleSheet("background-color : lightgrey")





        # self.setCentralWidget(self.central_widget)
        self.setLayout(self.layoutH0)
        self.layoutH0.addLayout(self.layoutV0)#column 1
        self.layoutH0.addLayout(self.layoutV1)#column 2
        self.layoutV0.addLayout(self.layoutH1)
        self.layoutH1.addWidget(QLabel('Direction (X/Y)'))
        self.layoutH1.addWidget(self.xybutton)
        self.layoutV0.addLayout(self.layoutH2)
        self.layoutH2.addWidget(QLabel('Projection Num'))
        self.layoutH2.addWidget(self.projectionNum)
        self.layoutV0.addLayout(self.layoutH3)
        self.layoutH3.addWidget(QLabel('Number of Peaks'))
        self.layoutH3.addWidget(self.num_peaks)
        self.layoutV0.addLayout(self.layoutH4)
        self.layoutH4.addWidget(QLabel('Peak to fit'))
        self.layoutH4.addWidget(self.fit_peak)
        self.layoutV0.addLayout(self.layoutH5)
        self.layoutH5.addWidget(QLabel('Integral of Peak'))
        self.layoutH5.addWidget(self.integral)
        self.layoutV0.addLayout(self.layoutH6)
        self.layoutH6.addWidget(QLabel('Peak Mean'))
        self.layoutH6.addWidget(self.peakMean)
        self.layoutV0.addLayout(self.layoutH7)
        self.layoutH7.addWidget(QLabel('Peak Width'))
        self.layoutH7.addWidget(self.peakWidth)
        self.layoutV0.addLayout(self.layoutH8)
        self.layoutH8.addWidget(QLabel('Peak Offset'))
        self.layoutH8.addWidget(self.peakOffset)
        self.layoutV0.addWidget(self.fit)
        self.layoutV1.addWidget(ImageData.ImageReader.plota)



 
    # method called by button
    def changeState(self):
 
        # if button is checked
        if self.xybutton.isChecked():
 
            # setting background color to light-blue
            self.xybutton.setStyleSheet("background-color : lightblue")
            self.xybutton.setText('Y')
        # if it is unchecked
        else:
 
            # set background color back to light-grey
            self.xybutton.setStyleSheet("background-color : lightgrey")
            self.xybutton.setText('X')

    def HandFitting(self):

        if self.xybutton.isChecked():
            isX = False
            self.projectiondf = self.Yprojdf
        else:
            isX = True
            self.projectiondf = self.Xprojdf
        # print(type(projectiondf))<class 'pandas.core.frame.DataFrame'>
        # print(self.projectionNum.returnValue())
        num_projection = int(self.projectionNum.returnValue())
        integral = int(self.integral.text())
        mean = float(self.peakMean.text())
        sigma = float(self.peakWidth.text())
        delta = float(self.peakOffset.text())
        num_peaks = int(self.num_peaks.text())
        peak_2_fit = int(self.fit_peak.text())
        
        # ImageData.imgData, ImageData.ImageReader.threshold, self.d, self.x3s, self.y3s = self.mainwindow.returnImageData()
        ImageData.ImageReader.plota.clear()
        
        off_peaks = [num_projection]
        # print(off_peaks)
        print(self.projectiondf.head())
        self.projectiondf.drop(off_peaks)
        self.x_offset = ImageData.imgData.shape[0]/2
        self.y_offset = ImageData.imgData.shape[1]/2

        for peaknum_start in off_peaks:
            if isX == True:
                self.positions = self.projectiondf.Ypos[peaknum_start]
                guess1 = [integral, self.y3s[peak_2_fit],sigma,ImageData.ImageReader.threshold]
                cut1 = self.y3s[peak_2_fit] - math.ceil(self.d/2)
                cut2 = self.y3s[peak_2_fit] + math.ceil(self.d/2)
                arr2 = self.x3s
            else:
                self.positions = self.projectiondf.Xpos[peaknum_start]
                guess1 = [integral, self.x3s[peak_2_fit],sigma, ImageData.ImageReader.threshold]
                cut1 = self.x3s[peak_2_fit] - math.ceil(self.d/2)
                cut2 = self.x3s[peak_2_fit] + math.ceil(self.d/2)
                arr2 = self.y3s

            data = np.array(ImageData.imgData[:, self.positions- math.ceil(self.d/2):self.positions+ math.ceil(self.d/2)])
            temp = np.arange(data.shape[0])
            flim = np.arange(temp.min()-1, temp.max(),1)
            for i in range(temp.shape[0]):
                temp[i] = sum(data[i])
            ImageData.ImageReader.plota.plot(flim,temp, pen = self.ppen)
            errfunc1 = lambda p, x, y: (self.mainwindow.gaussian(x, *p) - y)**2
            optim1, success = optimize.leastsq(errfunc1, guess1[:], args=(flim[cut1:cut2], temp[cut1:cut2]))
            ImageData.ImageReader.plota.plot(flim[cut1:cut2], self.mainwindow.gaussian(flim[cut1:cut2], *optim1), pen = self.hfpen, label = 'Hand Fit')
            if num_peaks <= 1:
                print('peak already plotted')
            elif num_peaks == 2:
                errfunc2 = lambda p, x, y: (self.mainwindow.two_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 5, 18000, arr2[1], 5, ImageData.ImageReader.threshold]
                optim2, success2 = optimize.leastsq(errfunc2, guess[:], args=(flim, temp))
                ImageData.ImageReader.plota.plot(flim, self.mainwindow.two_gaussians(flim, *optim2),pen = self.fpen, label='fit of 2 Gaussians')
            elif num_peaks == 3:
                errfunc3 = lambda p, x, y: (self.mainwindow.three_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 5, 18000, arr2[1], 5,55000, arr2[2],5, ImageData.ImageReader.threshold]
                optim2, success2 = optimize.leastsq(errfunc3, guess[:], args=(flim, temp))
                ImageData.ImageReader.plota.plot(flim, self.mainwindow.three_gaussians(flim, *optim2),pen = self.fpen, label='fit of 3 Gaussians')
            elif num_peaks == 4:
                errfunc4 = lambda p, x, y: (self.mainwindow.four_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5, ImageData.ImageReader.threshold]
                optim2, success2 = optimize.leastsq(errfunc4, guess[:], args=(flim, temp))
                ImageData.ImageReader.plota.plot(flim, self.mainwindow.four_gaussians(flim, *optim2),pen = self.fpen, label='fit of 4 Gaussians')
            elif num_peaks == 5:
                errfunc5 = lambda p, x, y: (self.mainwindow.five_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5, ImageData.ImageReader.threshold]
                optim2, success2 = optimize.leastsq(errfunc5, guess[:], args=(flim, temp))
                ImageData.ImageReader.plota.plot(flim, self.mainwindow.five_gaussians(flim, *optim2),pen = self.fpen, label='fit of 5 Gaussians')
            elif num_peaks == 6:
                errfunc6 = lambda p, x, y: (self.mainwindow.six_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5, ImageData.ImageReader.threshold]
                optim2, success = optimize.leastsq(errfunc6, guess[:], args=(flim, temp))
                ImageData.ImageReader.plota.plot(flim, self.mainwindow.six_gaussians(flim, *optim2),pen = self.fpen, label='fit of 6 Gaussians')
            elif num_peaks == 7:
                errfunc7 = lambda p, x, y: (self.mainwindow.seven_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5,120000,arr2[6],5, ImageData.ImageReader.threshold]
                optim2, success2 = optimize.leastsq(errfunc7, guess[:], args=(flim, temp))
                ImageData.ImageReader.plota.plot(flim, self.mainwindow.seven_gaussians(flim, *optim2),pen = self.fpen, label='fit of 7 Gaussians')
            elif num_peaks == 8:
                errfunc8 = lambda p, x, y: (self.mainwindow.eight_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5,120000,arr2[6],5,18000, arr2[7],0.25, ImageData.ImageReader.threshold]
                optim2, success2 = optimize.leastsq(errfunc8, guess[:], args=(flim, temp))
                ImageData.ImageReader.plota.plot(flim, self.mainwindow.eight_gaussians(flim, *optim2),pen = self.fpen, label='fit of 8 Gaussians')
            # print(self.positions)
            self.projectiondf.Int[peaknum_start + peak_2_fit] = optim1[0]
            self.projectiondf.Mean[peaknum_start + peak_2_fit] = optim1[1]
            self.projectiondf.Sig[peaknum_start + peak_2_fit] = optim1[2]/2
            # print(self.projectiondf.head())
        return      
    def Slider2ValueChange(self, value):
        self.projectionNum.setText(f'{value}')       

class IntRead(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
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
        # print("calib changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("calib edited...")
        # print(s)
        return        

class MeanRead(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
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
        # print("calib changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("calib edited...")
        # print(s)
        return           

class WidthRead(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
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
        # print("calib changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("calib edited...")
        # print(s)
        return          

class OffsetRead(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
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
        # print("calib changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("calib edited...")
        # print(s)
        return           

class HFNumPeaksRead(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
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
        # print("calib changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("calib edited...")
        # print(s)
        return      
class FitPeakRead(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
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
        # print("calib changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("calib edited...")
        # print(s)
        return  
class CustomSlider(QWidget):
    def __init__(self, *args, **kwargs):
        super(CustomSlider, self).__init__(*args, **kwargs)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximum(12)#make num peaks
        self.slider.valueChanged.connect(self.handleSliderValueChange)
        self.numbox = QSpinBox()
        self.numbox.valueChanged.connect(self.handleNumboxValueChange)
        layout = QHBoxLayout(self)
        layout.addWidget(self.numbox)
        layout.addWidget(self.slider)


    def handleSliderValueChange(self, value):
        self.numbox.setValue(value)


    def handleNumboxValueChange(self, value):
        # Prevent values outside slider range
        if value < self.slider.minimum():
            self.numbox.setValue(self.slider.minimum())
        elif value > self.slider.maximum():
            self.numbox.setValue(self.slider.maximum())

        self.slider.setValue(self.numbox.value())

    def returnValue(self):
        return self.numbox.value()
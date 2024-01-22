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
import ImageFields
import MaskFields
import HandFitWindow
import Sliders
import Fitter
import ResultFields

# path = os.getcwd()
path = 'D:/Workspace/Images/'
scale = 1

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowIcon(QIcon("mrsPepper.png"))
        self.setWindowTitle("PYpperpot 2.1")
        
        self.central_widget = QWidget() # A QWidget to work as Central Widget
        self.layoutH0 = QHBoxLayout() # Main window
        self.layoutV0 = QVBoxLayout() # Plot Column
        self.layoutG1 = ImageData.ImageReader()
        self.layoutV1 = QVBoxLayout() # File Params column
        self.ImgFields = ImageFields.ImFields()
        self.MskFields = MaskFields.MaskWidget()
        self.layoutH10 = QHBoxLayout() # Fit/Hand Fit
        self.setCentralWidget(self.central_widget)
#button classes to be started

        multiFit = QPushButton('Multi Fit (Max 8 peaks)')
        pbpFit = QPushButton('Peak-By-Peak Fit')
        self.handfit = QPushButton('Hand Fit')
        
#field class to be defined

#Connect your fields to functions
        multiFit.clicked.connect(Fitter.MultiFits.on_MultiFit_clicked)
        pbpFit.clicked.connect(Fitter.PeakByPeakFits.on_pbpFit_clicked)
        self.handfit.clicked.connect(self.on_Hand_clicked)

#Set Highest layer layout and work down
        self.central_widget.setLayout(self.layoutH0)
        self.layoutH0.addLayout(self.layoutV1,1)#column 1
        self.layoutH0.addLayout(self.layoutV0,9)#column 2

        self.layoutV0.addWidget(self.layoutG1)

        self.layoutV1.addWidget(self.ImgFields)
        self.layoutV1.addWidget(self.MskFields)

        self.layoutV1.addLayout(self.layoutH10)
        #self.layoutH10.addWidget(multiFit)
        self.layoutH10.addWidget(pbpFit)
        #self.layoutH10.addWidget(self.handfit)   

    def changeFitplots(self,value):
        # ImageData.ImageReader.plot1.clear()
        ImageData.ImageReader.plot2.clear()
        ImageData.ImageReader.plot3.clear()
        try:
            yprojInt,yprojMean,yprojSig,yprojX = self.fitter_func(self.y3s[value:value+1],self.x3s, self.num_peaks_y,ImageData.imgData, math.ceil(self.d/2), False, True)
            xprojInt,xprojMean,xprojSig,xprojY = self.fitter_func(self.x3s[value:value+1],self.y3s, self.num_peaks_x,ImageData.imgData, math.ceil(self.d/2), True, True)
        except:
            print(f'no peak {value}')
        # ImageData.ImageReader.plot1.clear()
        # ImageData.ImageReader.plot1.setImage(ImageData.imgData)
        self.p1linev1 = pg.PlotCurveItem(x=[self.x3s[value]-math.ceil(self.d/2), self.x3s[value]-math.ceil(self.d/2)], y=[0,700], pen = self.gpen)
        self.p1linev2 = pg.PlotCurveItem(x=[self.x3s[value]+math.ceil(self.d/2), self.x3s[value]+math.ceil(self.d/2)], y=[0,700], pen = self.gpen)
        self.p1lineh1 = pg.PlotCurveItem(x=[0, 700], y=[self.y3s[value]-math.ceil(self.d/2), self.y3s[value]-math.ceil(self.d/2)], pen = self.gpen)
        self.p1lineh2 = pg.PlotCurveItem(x=[0, 700], y=[self.y3s[value]+math.ceil(self.d/2), self.y3s[value]+math.ceil(self.d/2)], pen = self.gpen)
        # self.img = pg.ImageItem(image = ImageData.imgData)
        # self.p1view = ImageData.ImageReader.plot1.getView()
        self.p1view.clear()
        self.p1view.addItem(self.img)
        self.p1view.addItem(self.p1lineh1)
        self.p1view.addItem(self.p1linev1)
        self.p1view.addItem(self.p1lineh2)
        self.p1view.addItem(self.p1linev2)
        # self.edges = feature.canny(self.binary_image, sigma=value)
        # ImageData.ImageReader.plot4.setImage(self.edges)
        # ImageData.ImageReader.plot4.show()
 
    def on_Fit_clicked(self):
        try:
            self.sl.valueChanged.disconnect()
        except:
            print('no slider connection')

        self.num_peaks_x = int(self.xpeaksIn.text())
        self.num_peaks_y = int(self.ypeaksIn.text())
        self.n_holes = int(self.numHoles.text())
        self.sl.valueChanged.connect(self.changeFitplots)#connect slider to fit plots
        self.sl.setMinimum(0)
        self.sl.setMaximum(self.n_holes-1)
        hole_diameter = float(self.diamIn.text())
        hole_separation = float(self.sepIn.text())
        mask_to_screen = float(self.Mask2ScrnIn.text())
        pixpermm = float(self.Calibration.text())
        min_x = int(self.xminIn.text())
        min_y = int(self.yminIn.text())
        max_x = int(self.xmaxIn.text())
        max_y = int(self.ymaxIn.text())
        self.x_offset = ImageData.imgData.shape[0]/2
        self.y_offset = ImageData.imgData.shape[1]/2
        locs2 =[]
        
        self.d = (hole_separation*pixpermm)/2+hole_diameter*pixpermm
        for i in range(self.n_holes):
            for j in range(self.n_holes):
                locs2.append([i*self.d-self.d*(self.n_holes-1)/2,j*self.d-(self.d*(self.n_holes-1))/2])
        locs2 = np.array(locs2).T
        y1s = []
        y2s = []
        x1s = []
        x2s = []
        self.x3s = []
        self.y3s = []
        for x1, y1 in zip(*np.where(self.edges)):
            y1s.append(y1)
            x1s.append(x1)
    
        y1s = np.array(y1s)
        x1s = np.array(x1s)
        x2s, y2s = self.cutdown(x1s,y1s,  math.ceil(self.d/2), x2s,y2s)
        self.x3s, self.y3s = self.cutdown(x2s,y2s,  math.ceil(self.d/2), self.x3s,self.y3s)
        yprojInt,yprojMean,yprojSig,yprojX = self.fitter_func(self.y3s,self.x3s, self.num_peaks_y,ImageData.imgData, math.ceil(self.d/2), False, False)
        xprojInt,xprojMean,xprojSig,xprojY = self.fitter_func(self.x3s,self.y3s, self.num_peaks_x,ImageData.imgData, math.ceil(self.d/2), True, False)
        self.Xprojdf = pd.DataFrame({'Ypos': xprojY, 'Mean': xprojMean, 'Sig': xprojSig,'Int': xprojInt})
        self.Yprojdf = pd.DataFrame({'Xpos': yprojX, 'Mean': yprojMean, 'Sig': yprojSig,'Int': yprojInt})

        #Remapping
        Meanx = []
        Intx = []
        Sigx = []
        j = 0
        for i in range(self.Xprojdf.shape[0]):
            if i < self.num_peaks_y:
                # print(i)
                Meanx.append(self.Xprojdf.Mean[i * self.num_peaks_x])
                Intx.append(self.Xprojdf.Int[i * self.num_peaks_x])
                Sigx.append(self.Xprojdf.Sig[i * self.num_peaks_x])
            else:
                if i%self.num_peaks_y == 0:
                    j=j+1
                Meanx.append(self.Xprojdf.Mean[i%self.num_peaks_y*self.num_peaks_x + j])
                Intx.append(  self.Xprojdf.Int[i%self.num_peaks_y*self.num_peaks_x + j])
                Sigx.append(  self.Xprojdf.Sig[i%self.num_peaks_y*self.num_peaks_x + j])
        Meanx = np.array(Meanx)
        Intx = np.array(Intx)
        Sigx = np.array(Sigx)
        locsdf = pd.DataFrame({'X':locs2[0]+self.x_offset, 'Y':locs2[1]+self.y_offset})
        templocs = locsdf[(locsdf.X > min_x) & (locsdf.X < max_x)]
        templocs = templocs[(locsdf.Y > min_y) & (locsdf.Y < max_y)]
        print(templocs.shape[0])
        print(Meanx.shape[0])
        
        self.projectionsdf = pd.DataFrame({'HoleX':templocs.X.to_numpy(),'MeanX': Meanx,'SigX': Sigx,'IntX': Intx,'HoleY': templocs.Y.to_numpy(), 'MeanY': yprojMean, 'SigY': yprojSig,'IntY': yprojInt})
        self.sl.setValue(4)
        return
    
    def returnProjection(self, isYdir):
        self.num_peaks_x = int(self.xpeaksIn.text())
        self.num_peaks_y = int(self.ypeaksIn.text())
        self.n_holes = int(self.numHoles.text())
        hole_diameter = float(self.diamIn.text())
        hole_separation = float(self.sepIn.text())
        pixpermm = float(self.Calibration.text())
        self.x_offset = ImageData.imgData.shape[0]/2
        self.y_offset = ImageData.imgData.shape[1]/2
        locs2 =[]
        
        self.d = (hole_separation*pixpermm)/2+hole_diameter*pixpermm
        for i in range(self.n_holes):
            for j in range(self.n_holes):
                locs2.append([i*self.d-self.d*(self.n_holes-1)/2,j*self.d-(self.d*(self.n_holes-1))/2])
        locs2 = np.array(locs2).T
        y1s = []
        y2s = []
        x1s = []
        x2s = []
        self.x3s = []
        self.y3s = []
        for x1, y1 in zip(*np.where(self.edges)):
            y1s.append(y1)
            x1s.append(x1)
    
        y1s = np.array(y1s)
        x1s = np.array(x1s)
        x2s, y2s = self.cutdown(x1s,y1s,  math.ceil(self.d/2), x2s,y2s)
        self.x3s, self.y3s = self.cutdown(x2s,y2s,  math.ceil(self.d/2), self.x3s,self.y3s)
        yprojInt,yprojMean,yprojSig,yprojX = self.fitter_func(self.y3s,self.x3s, self.num_peaks_y,ImageData.imgData, math.ceil(self.d/2), False, False)
        xprojInt,xprojMean,xprojSig,xprojY = self.fitter_func(self.x3s,self.y3s, self.num_peaks_x,ImageData.imgData, math.ceil(self.d/2), True, False)
        self.Xprojdf = pd.DataFrame({'Ypos': xprojY, 'Mean': xprojMean, 'Sig': xprojSig,'Int': xprojInt})
        self.Yprojdf = pd.DataFrame({'Xpos': yprojX, 'Mean': yprojMean, 'Sig': yprojSig,'Int': yprojInt})
        if isYdir ==False:
            return self.Xprojdf
        else:
            return self.Yprojdf
  
    def returnImageData(self):
        return ImageData.imgData, ImageData.ImageReader.threshold, self.d, self.x3s, self.y3s
    
    def on_Hand_clicked(self):
        self.w2 = HandFitWindow.Handfitting(self.Xprojdf, self.Yprojdf, ImageData.imgData, ImageData.ImageReader.threshold, self.d, self.x3s, self.y3s)
        self.w2.show()



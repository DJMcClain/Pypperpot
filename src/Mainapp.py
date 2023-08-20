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

# path = os.getcwd()
path = 'D:/Workspace/Images/'
scale = 1

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowIcon(QIcon("mrsPepper.png"))
        self.setWindowTitle("PYpperpot 2.0")
        
        self.central_widget = QWidget() # A QWidget to work as Central Widget
        self.layoutH0 = QHBoxLayout() # Main window
        self.layoutV0 = QVBoxLayout() # Plot Column
        self.layoutG1 = QGridLayout() # Plots
        self.layoutGR00 = QHBoxLayout() # Plot Row 1
        self.layoutGR01 = QHBoxLayout() # Plot Row 2
        self.layoutGC10 = QVBoxLayout() # Plot Row 2 column 1
        self.layoutGC11 = QVBoxLayout() # Plot Row 2 column 2
        self.layoutGC00 = QVBoxLayout() # Plot Row 1 column 1
        self.layoutGC01 = QVBoxLayout() # Plot Row 1 column 2
        self.layoutV1 = QVBoxLayout() # File Params column
        self.layoutH1 = QHBoxLayout() # Load/Save Image
        self.layoutH2 = QHBoxLayout() # peak nums
        self.layoutH3 = QHBoxLayout() # max/min pixels
        self.layoutH3b = QHBoxLayout() # max/min pixels
        self.layoutH4 = QHBoxLayout() # Save/Load Mask
        self.layoutH5 = QHBoxLayout() # Number of holes in mask
        self.layoutH6 = QHBoxLayout() # Hole Diameter
        self.layoutH7 = QHBoxLayout() # Hole Separation
        self.layoutH8 = QHBoxLayout() # Mask to Screen
        self.layoutH9 = QHBoxLayout() # Calibration
        self.layoutH10 = QHBoxLayout() # Fit/Hand Fit
        self.setCentralWidget(self.central_widget)

######Complete classes
        self.xpeaksIn = ImageFields.x_peak_read()
        self.ypeaksIn = ImageFields.y_peak_read()
        self.xminIn = ImageFields.x_min_read()
        self.yminIn = ImageFields.y_min_read()
        self.xmaxIn = ImageFields.x_max_read()
        self.ymaxIn = ImageFields.y_max_read()
        self.diamIn = ImageFields.hole_diam_read()
        self.numHoles = ImageFields.hole_num_read()
        self.sepIn = ImageFields.hole_sep_read()
        self.Mask2ScrnIn = MaskFields.mask2Scrn_read()
        self.Calibration = MaskFields.calib_read()
        MainWindow.Slider1 = Sliders.slider1() #Threshold
        MainWindow.Slider2 = Sliders.slider2() #Prominence
        MainWindow.slY = Sliders.slider3()
        MainWindow.slX = Sliders.slider4()
#completed functions
        loadImagePrompt = QPushButton('Load Image')
        saveImagePrompt = QPushButton('*Save Image*')
#functions to be completed
        loadMaskPrompt = QPushButton('Load Mask')
        saveMaskPrompt = QPushButton('Save Mask')
        # changeThreshold = 
#button classes to be started

        fit = QPushButton('Fit')
        self.handfit = QPushButton('Hand Fit')
        
#field class to be defined

#Connect your fields to functions
        loadImagePrompt.clicked.connect(ImageData.ImageReader.on_LoadIm_clicked)
        saveImagePrompt.clicked.connect(self.on_SaveIm_clicked)
        loadMaskPrompt.clicked.connect(self.on_LoadMa_clicked)
        saveMaskPrompt.clicked.connect(self.on_SaveMa_clicked)
        fit.clicked.connect(self.on_Fit_clicked)
        self.handfit.clicked.connect(self.on_Hand_clicked)

#plots
        MainWindow.plot2 = ImageData.xProjection()
        MainWindow.plot3 = ImageData.yProjection()
        MainWindow.plot1 = ImageData.ImagePlot1(view=pg.PlotItem())
        MainWindow.plot4 = ImageData.ImagePlot2(view=pg.PlotItem())
        

        #Set Highest layer layout and work down
        self.central_widget.setLayout(self.layoutH0)
        self.layoutH0.addLayout(self.layoutV1)#column 1
        self.layoutH0.addLayout(self.layoutV0)#column 2

        self.layoutV0.addLayout(self.layoutG1)
        
        self.layoutG1.addWidget(MainWindow.plot2, 1, 0)
        self.layoutG1.addWidget(MainWindow.plot4, 1, 1)
        self.layoutG1.addLayout(self.layoutGR00, 0, 0)
        self.layoutG1.addLayout(self.layoutGR01, 0, 1)
        self.layoutG1.addLayout(self.layoutGC10, 1, 0)
        self.layoutGR00.addLayout(self.layoutGC00)
        self.layoutGC00.addWidget(MainWindow.Slider1)
        self.layoutGC00.addWidget(MainWindow.plot1)
        self.layoutGR00.addWidget(MainWindow.Slider2)

        self.layoutGR01.addWidget(MainWindow.plot3)
        self.layoutGR01.addWidget(MainWindow.slY)

        self.layoutGC10.addWidget(MainWindow.plot2)
        self.layoutGC10.addWidget(MainWindow.slX)

        self.layoutV1.addLayout(self.layoutH1)
        self.layoutH1.addWidget(loadImagePrompt)
        self.layoutH1.addWidget(saveImagePrompt)

        # self.layoutV1.addWidget(QLabel('What You See'))
        self.layoutV1.addLayout(self.layoutH2)
        self.layoutH2.addWidget(QLabel('X-peaks'))
        self.layoutH2.addWidget(self.xpeaksIn)
        self.layoutH2.addWidget(QLabel('Y-peaks'))
        self.layoutH2.addWidget(self.ypeaksIn)

        self.layoutV1.addLayout(self.layoutH3)
        self.layoutH3.addWidget(QLabel('Min X'))
        self.layoutH3.addWidget(self.xminIn)
        self.layoutH3.addWidget(QLabel('Max X'))
        self.layoutH3.addWidget(self.xmaxIn)

        self.layoutV1.addLayout(self.layoutH3b)
        self.layoutH3b.addWidget(QLabel('Min Y'))
        self.layoutH3b.addWidget(self.yminIn)
        self.layoutH3b.addWidget(QLabel('Max Y'))
        self.layoutH3b.addWidget(self.ymaxIn)

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

        self.layoutV1.addLayout(self.layoutH10)
        self.layoutH10.addWidget(fit)
        self.layoutH10.addWidget(self.handfit)   
    
    def updateImage1(image):
        # MainWindow.plot2 = ImageData.xProjection()
        # MainWindow.plot3 = ImageData.yProjection()
        MainWindow.plot1 = ImageData.ImagePlot1(view=pg.PlotItem())
        # MainWindow.plot4 = ImageData.ImagePlot2(view=pg.PlotItem())




    def on_LoadIm_clicked(self):
        self.loadImageName = QFileDialog.getOpenFileName(self, "Open Image",path, "Image Files (*.png *.jpg *.bmp *.csv *txt)")
        # try:
        #     self.sl.valueChanged.disconnect()
        #     self.sl2.valueChanged.disconnect()
        # except:
        #     print('no slider connection')
        # self.sl.valueChanged.connect(self.changeThreshold)#connect slider to image threshold
        # self.sl2.valueChanged.connect(self.changeProminence)
        # print(self.loadImageName[0][-3:])
        if self.loadImageName[0][-3:] == "png":
            self.imgData = ImageData.ImageReader.image_GSmatrix(self.loadImageName)
            self.imgData = self.imgData[10:-10,10:-10]
        elif self.loadImageName[0][-3:] == "jpg":
            self.imgData =  ImageData.ImageReader.image_GSmatrix(self.loadImageName) 
            self.imgData = self.imgData[10:-10,10:-10]
        elif self.loadImageName[0][-3:] == "bmp":
            self.imgData =  ImageData.ImageReader.image_GSmatrix(self.loadImageName) 
            self.imgData = self.imgData[10:-10,10:-10]
        elif self.loadImageName[0][-3:] == "csv":
            self.imgData =  ImageData.ImageReader.csv_GSmatrix(self.loadImageName) 
        elif self.loadImageName[0][-3:] == "txt":
            self.imgData =  ImageData.ImageReader.csv_GSmatrix(self.loadImageName) 

        self.threshold = filters.threshold_otsu(self.imgData)
        self.binary_image = self.imgData > self.threshold/2
        self.edges = feature.canny(self.binary_image, sigma=5)
        # self.img2 = self.imgData+self.edges
        outlines = np.zeros((*self.edges.shape,4))
        outlines[:, :, 0] = 255 * self.edges
        outlines[:, :, 3] = 255.0 * self.edges

        i = 0
        xpeaks=[]
        ypeaks=[]
        for row in self.imgData:
            peaks = scipy.signal.find_peaks(row, height = self.threshold/2, prominence = 8)
            if peaks[0].shape[0] != 0:
                for peak in peaks[0]:
                    xpeaks.append(peak)
                    ypeaks.append(i)
            i+=1
        self.xpeaks = np.array(xpeaks)
        self.ypeaks = np.array(ypeaks)
        self.min_y = np.min([np.min(np.where(self.edges ==True)[1]),np.min(self.ypeaks)])
        self.max_y = np.max([np.max(np.where(self.edges ==True)[1]), np.max(self.ypeaks)])
        self.min_x = np.min([np.min(np.where(self.edges ==True)[0]), np.min(self.xpeaks)])
        self.max_x = np.max([np.max(np.where(self.edges ==True)[0]),np.max(self.xpeaks)])
        try:
            self.p1view.clear()
        except:
            print('no previous view')
        self.img = pg.ImageItem(image = self.imgData)
        # self.plot1.setImage(self.imgData)

        self.p1view = self.plot1.getView()
        self.outlines = pg.ImageItem(image = outlines)
        self.p1view.addItem(self.img)
        
        self.xmaxIn.setText(f'{self.max_x}')
        self.xminIn.setText(f'{self.min_x}')
        self.ymaxIn.setText(f'{self.max_y}')
        self.yminIn.setText(f'{self.min_y}')
        self.xpeaksIn.setText(f'{self.xpeaks.shape[0]}')
        self.ypeaksIn.setText(f'{self.ypeaks.shape[0]}')
        self.p1dots1 =  pg.ScatterPlotItem(x=self.xpeaks, y=self.ypeaks, pen = 'c', symbol = 'o')
        self.p1linev1 = pg.PlotCurveItem(x=[self.min_x, self.min_x], y=[self.min_y,self.max_y], pen = self.gpen)
        self.p1linev2 = pg.PlotCurveItem(x=[self.max_x, self.max_x], y=[self.min_y,self.max_y], pen =self.gpen)
        self.p1lineh1 = pg.PlotCurveItem(x=[self.min_x, self.max_x], y=[self.min_y,self.min_y], pen = self.gpen)
        self.p1lineh2 = pg.PlotCurveItem(x=[self.min_x, self.max_x], y=[self.max_y,self.max_y], pen =self.gpen)
        self.p1view = self.plot1.getView()
        self.p1view.addItem(self.p1dots1)
        self.p1view.addItem(self.outlines)
        self.p1view.addItem(self.p1lineh1)
        self.p1view.addItem(self.p1linev1)
        self.p1view.addItem(self.p1lineh2)
        self.p1view.addItem(self.p1linev2)
        
        self.xminIn.setText(np.min(self.edges[0]))
        self.plot4.setImage(self.img2)
        #self.plot1.setImage(dummy)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,10))
        ax[1].imshow(self.edges, origin='lower')
        im = ax[0].imshow(self.imgData, origin='lower')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im,cax=cax, orientation='vertical')
        plt.show()

    def on_SaveIm_clicked(self):
        saveImageName = QFileDialog.getSaveFileName(self, "Save Image",path, "Image Files (*.png *.jpg *.bmp)")


    def on_LoadMa_clicked(self):
        loadMaskName = QFileDialog.getOpenFileName(self, "Load Mask",path, "*.csv")
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
        saveMaskName = QFileDialog.getSaveFileName(self, "Save Mask",path, "*.csv")
        file = open(saveMaskName[0], 'w')
        writer =csv.writer(file)
        writer.writwrow([self.numHoles.text(), self.diamIn.text(), self.sepIn.text(),self.Mask2ScrnIn.text(), self.Calibration.text()])
        file.close()
   
    # def changeThreshold(self,value):
    #     # self.plot1.clear()
    #     self.edges = feature.canny(self.binary_image, sigma=value)
    #     outlines = np.zeros((*self.edges.shape,4))
    #     outlines[:, :, 0] = 255 * self.edges
    #     outlines[:, :, 3] = 255.0 * self.edges
    #     # self.plot1.setImage(self.imgData)
    #     # self.p1view = self.plot1.getView()
    #     try:
    #         self.p1view.clear()
    #     except:
    #         print('no previous view')

    #     self.p1view.addItem(self.img)
    #     self.outlines = pg.ImageItem(image = outlines)

    #     self.min_y = np.min([np.min(np.where(self.edges ==True)[1]),np.min(self.ypeaks)])
    #     self.max_y = np.max([np.max(np.where(self.edges ==True)[1]), np.max(self.ypeaks)])
    #     self.min_x = np.min([np.min(np.where(self.edges ==True)[0]), np.min(self.xpeaks)])
    #     self.max_x = np.max([np.max(np.where(self.edges ==True)[0]),np.max(self.xpeaks)])

    #     self.img = pg.ImageItem(image = self.imgData)
    #     # self.plot1.setImage(self.imgData)

    #     self.xmaxIn.setText(f'{self.max_x}')
    #     self.xminIn.setText(f'{self.min_x}')
    #     self.ymaxIn.setText(f'{self.max_y}')
    #     self.yminIn.setText(f'{self.min_y}')
        
    #     self.p1dots1 =  pg.ScatterPlotItem(x=self.xpeaks, y=self.ypeaks, pen = 'c', symbol = 'o')
    #     self.p1linev1 = pg.PlotCurveItem(x=[self.min_x, self.min_x], y=[self.min_y,self.max_y], pen = self.gpen)
    #     self.p1linev2 = pg.PlotCurveItem(x=[self.max_x, self.max_x], y=[self.min_y,self.max_y], pen =self.gpen)
    #     self.p1lineh1 = pg.PlotCurveItem(x=[self.min_x, self.max_x], y=[self.min_y,self.min_y], pen = self.gpen)
    #     self.p1lineh2 = pg.PlotCurveItem(x=[self.min_x, self.max_x], y=[self.max_y,self.max_y], pen =self.gpen)
    #     self.p1view = self.plot1.getView()
    #     self.p1view.addItem(self.p1dots1)
    #     self.p1view.addItem(self.outlines)
    #     self.p1view.addItem(self.p1lineh1)
    #     self.p1view.addItem(self.p1linev1)
    #     self.p1view.addItem(self.p1lineh2)
    #     self.p1view.addItem(self.p1linev2)

    # def changeProminence(self,value):
    #     # self.plot1.clear()
    #     # self.edges = feature.canny(self.binary_image, sigma=value)
    #     outlines = np.zeros((*self.edges.shape,4))
    #     outlines[:, :, 0] = 255 * self.edges
    #     outlines[:, :, 3] = 255.0 * self.edges
    #     # self.plot1.setImage(self.imgData)
    #     # self.p1view = self.plot1.getView()
    #     i = 0
    #     xpeaks=[]
    #     ypeaks=[]
    #     for row in self.imgData:
    #         peaks = scipy.signal.find_peaks(row, height = self.threshold/2, prominence = value)
    #         if peaks[0].shape[0] != 0:
    #             for peak in peaks[0]:
    #                 xpeaks.append(peak)
    #                 ypeaks.append(i)
    #         i+=1
    #     self.xpeaks = np.array(xpeaks)
    #     self.ypeaks = np.array(ypeaks)
    #     self.min_y = np.min([np.min(np.where(self.edges ==True)[1]),np.min(self.ypeaks)])
    #     self.max_y = np.max([np.max(np.where(self.edges ==True)[1]), np.max(self.ypeaks)])
    #     self.min_x = np.min([np.min(np.where(self.edges ==True)[0]), np.min(self.xpeaks)])
    #     self.max_x = np.max([np.max(np.where(self.edges ==True)[0]),np.max(self.xpeaks)])
    #     try:
    #         self.p1view.clear()
    #     except:
    #         print('no previous view')

    #     self.p1view.addItem(self.img)
    #     self.outlines = pg.ImageItem(image = outlines)
    #     # self.plot1.setImage(self.imgData)

    #     self.xmaxIn.setText(f'{self.max_x}')
    #     self.xminIn.setText(f'{self.min_x}')
    #     self.ymaxIn.setText(f'{self.max_y}')
    #     self.yminIn.setText(f'{self.min_y}')
    #     self.p1dots1 =  pg.ScatterPlotItem(x=self.xpeaks, y=self.ypeaks, pen = 'c', symbol = 'o')
    #     self.p1linev1 = pg.PlotCurveItem(x=[self.min_x, self.min_x], y=[self.min_y,self.max_y], pen = self.gpen)
    #     self.p1linev2 = pg.PlotCurveItem(x=[self.max_x, self.max_x], y=[self.min_y,self.max_y], pen =self.gpen)
    #     self.p1lineh1 = pg.PlotCurveItem(x=[self.min_x, self.max_x], y=[self.min_y,self.min_y], pen = self.gpen)
    #     self.p1lineh2 = pg.PlotCurveItem(x=[self.min_x, self.max_x], y=[self.max_y,self.max_y], pen =self.gpen)
    #     self.p1view = self.plot1.getView()
    #     self.p1view.addItem(self.p1dots1)
    #     self.p1view.addItem(self.outlines)
    #     self.p1view.addItem(self.p1lineh1)
    #     self.p1view.addItem(self.p1linev1)
    #     self.p1view.addItem(self.p1lineh2)
    #     self.p1view.addItem(self.p1linev2)
    #     # self.plot2.plot(self.hour, self.temperature*value, pen =self.gpen)     
    #     # self.plot3.plot(self.temperature*value, self.hour, pen =self.gpen)    

    def changeFitplots(self,value):
        # self.plot1.clear()
        self.plot2.clear()
        self.plot3.clear()
        try:
            yprojInt,yprojMean,yprojSig,yprojX = self.fitter_func(self.y3s[value:value+1],self.x3s, self.num_peaks_y,self.imgData, math.ceil(self.d/2), False, True)
            xprojInt,xprojMean,xprojSig,xprojY = self.fitter_func(self.x3s[value:value+1],self.y3s, self.num_peaks_x,self.imgData, math.ceil(self.d/2), True, True)
        except:
            print(f'no peak {value}')
        # self.plot1.clear()
        # self.plot1.setImage(self.imgData)
        self.p1linev1 = pg.PlotCurveItem(x=[self.x3s[value]-math.ceil(self.d/2), self.x3s[value]-math.ceil(self.d/2)], y=[0,700], pen = self.gpen)
        self.p1linev2 = pg.PlotCurveItem(x=[self.x3s[value]+math.ceil(self.d/2), self.x3s[value]+math.ceil(self.d/2)], y=[0,700], pen = self.gpen)
        self.p1lineh1 = pg.PlotCurveItem(x=[0, 700], y=[self.y3s[value]-math.ceil(self.d/2), self.y3s[value]-math.ceil(self.d/2)], pen = self.gpen)
        self.p1lineh2 = pg.PlotCurveItem(x=[0, 700], y=[self.y3s[value]+math.ceil(self.d/2), self.y3s[value]+math.ceil(self.d/2)], pen = self.gpen)
        # self.img = pg.ImageItem(image = self.imgData)
        # self.p1view = self.plot1.getView()
        self.p1view.clear()
        self.p1view.addItem(self.img)
        self.p1view.addItem(self.p1lineh1)
        self.p1view.addItem(self.p1linev1)
        self.p1view.addItem(self.p1lineh2)
        self.p1view.addItem(self.p1linev2)
        # self.edges = feature.canny(self.binary_image, sigma=value)
        # self.plot4.setImage(self.edges)
        # self.plot4.show()
 



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
        self.x_offset = self.imgData.shape[0]/2
        self.y_offset = self.imgData.shape[1]/2
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
        yprojInt,yprojMean,yprojSig,yprojX = self.fitter_func(self.y3s,self.x3s, self.num_peaks_y,self.imgData, math.ceil(self.d/2), False, False)
        xprojInt,xprojMean,xprojSig,xprojY = self.fitter_func(self.x3s,self.y3s, self.num_peaks_x,self.imgData, math.ceil(self.d/2), True, False)
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
        self.x_offset = self.imgData.shape[0]/2
        self.y_offset = self.imgData.shape[1]/2
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
        yprojInt,yprojMean,yprojSig,yprojX = self.fitter_func(self.y3s,self.x3s, self.num_peaks_y,self.imgData, math.ceil(self.d/2), False, False)
        xprojInt,xprojMean,xprojSig,xprojY = self.fitter_func(self.x3s,self.y3s, self.num_peaks_x,self.imgData, math.ceil(self.d/2), True, False)
        self.Xprojdf = pd.DataFrame({'Ypos': xprojY, 'Mean': xprojMean, 'Sig': xprojSig,'Int': xprojInt})
        self.Yprojdf = pd.DataFrame({'Xpos': yprojX, 'Mean': yprojMean, 'Sig': yprojSig,'Int': yprojInt})
        if isYdir ==False:
            return self.Xprojdf
        else:
            return self.Yprojdf
    def returnImageData(self):
        return self.imgData, self.threshold, self.d, self.x3s, self.y3s
    def on_Hand_clicked(self):
        self.w2 = HandFitWindow.Handfitting(self.Xprojdf, self.Yprojdf, self.imgData, self.threshold, self.d, self.x3s, self.y3s)
        self.w2.show()

    def cutdown(self, xs, ys, gate, x2s, y2s):
        for i in range(ys.shape[0]):
            temp1 = ys[i]
            temp2 = ys[(ys >= temp1 - gate) & (ys <= temp1 + gate)]
            meantemp = round(np.mean(temp2))
            #print(tempy1s, meantemp)
            y2s.append(meantemp)
        y2s = np.unique(y2s) 
        y2s = np.array(y2s)
        for i in range(xs.shape[0]):
            temp1 = xs[i]
            temp2 = xs[(xs >= temp1 - gate) & (xs <= temp1 + gate)]
            meantemp = round(np.mean(temp2))
            #print(tempy1s, meantemp)
            x2s.append(meantemp)
        x2s = np.unique(x2s) 
        x2s = np.array(x2s)
        return x2s,y2s

    def gaussian(self, x, height, center, sigma, offset):
        return height/(sigma * np.sqrt(2*np.pi))*np.exp(-(x - center)**2/(2*sigma**2)) + offset
    
    def eight_gaussians(self,x, h1, c1, w1, 
		h2, c2, w2, 
		h3, c3, w3,
		h4, c4, w4,
		h5, c5, w5,
		h6, c6, w6,
        h7, c7, w7,
		h8, c8, w8,
		offset):
        return (self.gaussian(x, h1, c1, w1, offset) +
            self.gaussian(x, h2, c2, w2, offset) +
            self.gaussian(x, h3, c3, w3, offset) + 
            self.gaussian(x, h4, c4, w4, offset) + 
            self.gaussian(x, h5, c5, w5, offset) + 
            self.gaussian(x, h6, c6, w6, offset) + 
            self.gaussian(x, h7, c7, w7, offset) +
            self.gaussian(x, h8, c8, w8, offset) +
            offset)

    def seven_gaussians(self, x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		h5, c5, w5,
    		h6, c6, w6,
            h7, c7, w7,
    		offset):
        return  (self.gaussian(x, h1, c1, w1, offset) +
                self.gaussian(x, h2, c2, w2, offset) +
                self.gaussian(x, h3, c3, w3, offset) + 
                self.gaussian(x, h4, c4, w4, offset) + 
                self.gaussian(x, h5, c5, w5, offset) + 
                self.gaussian(x, h6, c6, w6, offset) + 
                self.gaussian(x, h7, c7, w7, offset) +
                offset)

    def six_gaussians(self, x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		h5, c5, w5,
    		h6, c6, w6,
    		offset):
        return (self.gaussian(x, h1, c1, w1, offset) +
            self.gaussian(x, h3, c3, w3, offset) + 
            self.gaussian(x, h4, c4, w4, offset) + 
            self.gaussian(x, h2, c2, w2, offset) +
            self.gaussian(x, h5, c5, w5, offset) + 
            self.gaussian(x, h6, c6, w6, offset) + 
            offset)

    def five_gaussians(self,x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		h5, c5, w5,
    		offset):
        return (self.gaussian(x, h1, c1, w1, offset) +
            self.gaussian(x, h2, c2, w2, offset) +
            self.gaussian(x, h3, c3, w3, offset) + 
            self.gaussian(x, h4, c4, w4, offset) + 
            self.gaussian(x, h5, c5, w5, offset) + 
            offset)

    def four_gaussians(self, x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		offset):
        return (self.gaussian(x, h1, c1, w1, offset) +
            self.gaussian(x, h2, c2, w2, offset) +
            self.gaussian(x, h3, c3, w3, offset) + 
            self.gaussian(x, h4, c4, w4, offset) + 
            offset)
    
    def three_gaussians(self, x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		offset):
        return (self.gaussian(x, h1, c1, w1, offset) +
            self.gaussian(x, h2, c2, w2, offset) +
            self.gaussian(x, h3, c3, w3, offset) + 
            offset) 

    def two_gaussians(self, x, h1, c1, w1, 
    		h2, c2, w2, 
    		offset):
        return (self.gaussian(x, h1, c1, w1, offset) +
            self.gaussian(x, h2, c2, w2, offset) + 
            offset)
    
    def fitter_func(self,arr,arr2, num_peaks,img,pixs, isX, plot):
        #initialize outputs
        projsInt = [] 
        projsMean = []
        projsSig = []
        projsPos = []
        #scan over the direction
        for i in arr:
            i = int(i)
            #we handle x and y separately, so check if it is X
            if isX == True:
                data = np.array(self.imgData[i-pixs:i+pixs,:])
                temp = np.arange(data.shape[1])
                flim = np.arange(temp.min()-1,temp.max(),1)
                for j in range(temp.shape[0]):
                    temp[j] = sum(data[0:,j])
            else:
                data = np.array(self.imgData[:,i-pixs:i+pixs])
                temp = np.arange(data.shape[0])
                flim = np.arange(temp.min()-1, temp.max(),1)
                for j in range(temp.shape[0]):
                    temp[j] = sum(data[j])
            if plot ==True:
                ppen = pg.mkPen(color=(0, 0, 150),width = 2)
                fpen = pg.mkPen(color=(255, 0, 0),width = 2)
                if isX == True:
                    self.plot2.plot(flim,temp, pen = ppen) 
                else:    
                    self.plot3.plot(temp,flim, pen = ppen) 
                # fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(14,10))
                # plt.title(i)
                # ax[0].plot(flim,temp)
                # im = ax[1].imshow(data)
                # divider = make_axes_locatable(ax[1])
                # cax = divider.append_axes('right', size='5%',pad = 0.05)
                # fig.colorbar(im,cax=cax, orientation='vertical')

            if num_peaks == 1:
                errfunc1 = lambda p, x, y: (self.gaussian(x, *p) - y)**2
                guess = [18000, arr2[0], 5,self.threshold]
                optim, success = optimize.leastsq(errfunc1, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        self.plot2.plot(flim, self.gaussian(flim, *optim), pen = fpen, label='fit of Gaussian')
                    else:    
                        self.plot3.plot(self.gaussian(flim, *optim),flim, pen = fpen, label='fit of Gaussian')
                    # ax[0].plot(flim, gaussian(flim, *optim),c='red', label='fit of Gaussian')
            elif num_peaks == 2:
                errfunc2 = lambda p, x, y: (self.two_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 5, 18000, arr2[1], 5, self.threshold]
                optim, success = optimize.leastsq(errfunc2, guess[:], args=(flim, temp))
                # ax[0].plot(flim, two_gaussians(flim, *optim),c='red', label='fit of 2 Gaussians')
                if plot ==True:
                    if isX == True:
                        self.plot2.plot(flim, self.two_gaussians(flim, *optim),pen = fpen, label='fit of 2 Gaussians')
                    else:    
                        self.plot3.plot(self.two_gaussians(flim, *optim),flim,pen = fpen, label='fit of 2 Gaussians')
            elif num_peaks == 3:
                errfunc3 = lambda p, x, y: (self.three_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 5, 18000, arr2[1], 5,55000, arr2[2],5, self.threshold]
                optim, success = optimize.leastsq(errfunc3, guess[:], args=(flim, temp))
                # ax[0].plot(flim, three_gaussians(flim, *optim),c='red', label='fit of 3 Gaussians')
                if plot ==True:
                    if isX == True:
                        self.plot2.plot(flim, self.three_gaussians(flim, *optim),pen = fpen, label='fit of 3 Gaussians')
                    else:    
                        self.plot3.plot(self.three_gaussians(flim, *optim),flim,pen = fpen, label='fit of 3 Gaussians')
            elif num_peaks == 4:
                errfunc4 = lambda p, x, y: (self.four_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5, self.threshold]
                optim, success = optimize.leastsq(errfunc4, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        self.plot2.plot(flim, self.four_gaussians(flim, *optim),pen = fpen, label='fit of 4 Gaussians')
                    else:    
                        self.plot3.plot(self.four_gaussians(flim, *optim),flim,pen = fpen, label='fit of 4 Gaussians')
                # ax[0].plot(flim, four_gaussians(flim, *optim),c='red', label='fit of 4 Gaussians') 
            elif num_peaks == 5:
                errfunc5 = lambda p, x, y: (self.five_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5, self.threshold]
                optim, success = optimize.leastsq(errfunc5, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        self.plot2.plot(flim, self.five_gaussians(flim, *optim),pen = fpen, label='fit of 5 Gaussians')
                    else:    
                        self.plot3.plot(self.five_gaussians(flim, *optim),flim,pen = fpen, label='fit of 5 Gaussians')
                # ax[0].plot(flim, five_gaussians(flim, *optim),c='red', label='fit of 5 Gaussians') 
            elif num_peaks == 6:
                errfunc6 = lambda p, x, y: (self.six_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5, self.threshold]
                optim, success = optimize.leastsq(errfunc6, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        self.plot2.plot(flim, self.six_gaussians(flim, *optim),pen = fpen, label='fit of 6 Gaussians')
                    else:    
                        self.plot3.plot(self.six_gaussians(flim, *optim),flim,pen = fpen, label='fit of 6 Gaussians')
                # ax[0].plot(flim, six_gaussians(flim, *optim),c='red', label='fit of 6 Gaussians') 
            elif num_peaks == 7:
                errfunc7 = lambda p, x, y: (self.seven_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5,120000,arr2[6],5, self.threshold]
                optim, success = optimize.leastsq(errfunc7, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        self.plot2.plot(flim, self.seven_gaussians(flim, *optim),pen = fpen, label='fit of 7 Gaussians')
                    else:    
                        self.plot3.plot(self.seven_gaussians(flim, *optim),flim,pen = fpen, label='fit of 7 Gaussians')
                # ax[0].plot(flim, seven_gaussians(flim, *optim),c='red', label='fit of 7 Gaussians')
            elif num_peaks == 8:
                errfunc8 = lambda p, x, y: (self.eight_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5,120000,arr2[6],5,18000, arr2[7],0.25, self.threshold]
                optim, success = optimize.leastsq(errfunc8, guess[:], args=(flim, temp))
                # ax[0].plot(flim, eight_gaussians(flim, *optim),c='red', label='fit of 8 Gaussians')
                if plot ==True:
                    if isX == True:
                        self.plot2.plot(flim, self.eight_gaussians(flim, *optim),pen = fpen, label='fit of 8 Gaussians')
                    else:    
                        self.plot3.plot(self.eight_gaussians(flim, *optim),flim,pen = fpen, label='fit of 8 Gaussians')
            n = 0
            for k in range(num_peaks):
                projsInt.append(optim[n])
                projsMean.append(optim[n+1])
                projsSig.append(optim[n+2])
                projsPos.append(i)
                n = n+3
        return projsInt,projsMean,projsSig,projsPos



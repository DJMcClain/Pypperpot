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
import Sliders
import ResultFields

# path = os.getcwd()
path = 'D:/Workspace/Images/'
scale = 1
imgData = np.array([[0, 0, 0], [0, 0, 0]])


class ImageReader(QMainWindow):
    gpen = pg.mkPen(color=(0, 255, 0))
    hour = np.array([1,2,3,4,5,6,7,8,9,10])
    temperature = np.array([30,32,34,32,33,31,29,32,35,45])
    def __init__(self):
        super(ImageReader, self).__init__()
        self.central_widget = QFrame() # A QWidget to work as Central Widget
        self.central_widget.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.central_widget.setLineWidth(2)
        self.layoutG1 = QGridLayout() # Plots
        self.layoutH0 = QHBoxLayout() # Central Widget
        self.layoutV0 = QVBoxLayout() # Plots
        self.layoutV1 = QVBoxLayout() # Sliders and Results
        self.layoutH1 = QHBoxLayout() #Sliders
        self.layoutV2 = QVBoxLayout() #Threshold
        self.layoutV3 = QVBoxLayout() #Prominence
        # self.layoutGR00 = QHBoxLayout() # Plot Row 1
        # self.layoutGR01 = QHBoxLayout() # Plot Row 2
        # self.layoutGC10 = QVBoxLayout() # Plot Row 2 column 1
        # self.layoutGC11 = QVBoxLayout() # Plot Row 2 column 2
        # self.layoutGC00 = QVBoxLayout() # Plot Row 1 column 1
        # self.layoutGC01 = QVBoxLayout() # Plot Row 1 column 2
        self.setCentralWidget(self.central_widget)

        self.ResFields = ResultFields.ResFields()
        self.threshBox = QFrame()
        self.promBox = QFrame()
        ImageReader.Slider1 = Sliders.slider1() #Threshold
        ImageReader.Slider2 = Sliders.slider2() #Prominence

        ImageReader.plot2 = ImageData.xProjection()
        ImageReader.plot3 = ImageData.yProjection()
        ImageReader.plot1 = pg.PlotWidget(plotItem=pg.PlotItem())#ImageData.ImagePlot1(view=pg.PlotItem())
        ImageReader.plot4 = pg.PlotWidget(plotItem=pg.PlotItem())#ImageData.ImagePlot2(view=pg.PlotItem())
        label_style = {'color': '#000', 'font-size': '10pt'}
        ImageReader.plot2.getAxis('bottom').setPen('k')
        ImageReader.plot2.getAxis('left').setPen('k')
        ImageReader.plot2.getAxis('bottom').setTextPen('k')
        ImageReader.plot2.getAxis('left').setTextPen('k')
        ImageReader.plot2.setLabel('bottom', "Position (mm)", **label_style)
        ImageReader.plot2.setLabel('left', "Divergence (mrad)", **label_style)
        # ImageReader.plot4.set
        self.central_widget.setLayout(self.layoutH0)
        self.layoutH0.addLayout(self.layoutV1,1)#Sliders and Results
        self.layoutH0.addLayout(self.layoutV0,9)#Plots
        
        self.layoutV0.addWidget(ImageReader.plot4,5)
        self.layoutV0.addWidget(ImageReader.plot2,5)
        self.layoutV1.addLayout(self.layoutH1)
        
        self.threshBox.setLayout(self.layoutV2)
        self.threshBox.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.threshBox.setLineWidth(2)
        self.threshBox.setStyleSheet("background-color : lightgrey")
        self.layoutV2.addWidget(ImageReader.Slider1,alignment=Qt.AlignHCenter)
        self.layoutH1.addWidget(self.threshBox,5)

        self.promBox.setLayout(self.layoutV3)
        self.promBox.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.promBox.setLineWidth(2)
        self.promBox.setStyleSheet("background-color : lightblue")
        self.layoutV3.addWidget(ImageReader.Slider2,alignment=Qt.AlignHCenter)
        self.layoutH1.addWidget(self.promBox,5)
        self.layoutV1.addWidget(self.ResFields)

    def on_LoadIm_clicked():
        loadImageName = QFileDialog.getOpenFileName( caption="Open Image",directory=path, filter="Image Files (*.png *.jpg *.bmp *.csv *txt)")

        Sliders.prombool = True
        Sliders.threshbool = True

        if loadImageName[0][-3:] == "png":
            ImageData.imgData = ImageReader.image_GSmatrix(loadImageName)
            ImageData.imgData = ImageData.imgData[10:-10,10:-10]
        elif loadImageName[0][-3:] == "jpg":
            ImageData.imgData = ImageReader.image_GSmatrix(loadImageName) 
            ImageData.imgData = ImageData.imgData[10:-10,10:-10]
        elif loadImageName[0][-3:] == "bmp":
            ImageData.imgData = ImageReader.image_GSmatrix(loadImageName) 
            ImageData.imgData = ImageData.imgData[10:-10,10:-10]
        elif loadImageName[0][-3:] == "csv":
            ImageData.imgData = ImageReader.csv_GSmatrix(loadImageName) 
        elif loadImageName[0][-3:] == "txt":
            ImageData.imgData = ImageReader.csv_GSmatrix(loadImageName) 

        try:
            ImageReader.plot4.clear()
        except:
            print('no previous view')
        ImageData.imgData = ImageData.imgData.T
        ImageData.x_offset = ImageData.imgData.shape[0]/2
        ImageData.y_offset = ImageData.imgData.shape[1]/2
        ImageReader.img = pg.ImageItem(image = ImageData.imgData, invertY = False)
        # ImageReader.img2 = pg.ImageItem(image = ImageData.imgData) 

        ImageReader.threshold = filters.threshold_otsu(ImageData.imgData)
        ImageReader.binary_image = ImageData.imgData > ImageReader.threshold/2
        # ImageReader.p1view = ImageReader.plot1.getView()
        # ImageReader.p1view.addItem(ImageReader.img)
        # ImageReader.p4view = ImageReader.plot4.getView()
        ImageReader.plot4.addItem(ImageReader.img)
        try:
            ImageReader.edges = feature.canny(ImageReader.binary_image, sigma=5)
            ImageReader.outlines = np.zeros((*ImageReader.edges.shape,4))
            ImageReader.outlines[:, :, 0] = 255 * ImageReader.edges
            ImageReader.outlines[:, :, 3] = 255.0 * ImageReader.edges
            ImageReader.edminy = np.min(np.where(ImageReader.edges ==True)[1])
            ImageReader.edmaxy = np.max(np.where(ImageReader.edges ==True)[1])
            ImageReader.edminx = np.min(np.where(ImageReader.edges ==True)[0])
            ImageReader.edmaxx = np.max(np.where(ImageReader.edges ==True)[0])
        except:
            ImageReader.edges = [ImageData.x_offset,ImageData.y_offset]
            ImageReader.edminy = ImageData.y_offset
            ImageReader.edmaxy = ImageData.y_offset
            ImageReader.edminx = ImageData.x_offset
            ImageReader.edmaxx = ImageData.x_offset
        i = 0
        xpeaks=[]
        ypeaks=[]
        for row in ImageData.imgData:
            peaks = scipy.signal.find_peaks(row, height = ImageReader.threshold/3, prominence = 8)
            if peaks[0].shape[0] != 0:
                for peak in peaks[0]:
                    xpeaks.append(peak)
                    ypeaks.append(i)
            i+=1
        #Define Values
        ImageReader.xpeaks = np.array(xpeaks)
        ImageReader.ypeaks = np.array(ypeaks)
        ImageReader.min_y = np.min([ImageReader.edminy, np.min(ImageReader.ypeaks)])
        ImageReader.max_y = np.max([ImageReader.edmaxy, np.max(ImageReader.ypeaks)])
        ImageReader.min_x = np.min([ImageReader.edminx, np.min(ImageReader.xpeaks)])
        ImageReader.max_x = np.max([ImageReader.edmaxx, np.max(ImageReader.xpeaks)])
        #Write Values to GUI
        ImageFields.ImFields.xmaxIn.setText(f'{ImageReader.max_x}')
        ImageFields.ImFields.xminIn.setText(f'{ImageReader.min_x}')
        ImageFields.ImFields.ymaxIn.setText(f'{ImageReader.max_y}')
        ImageFields.ImFields.yminIn.setText(f'{ImageReader.min_y}')
        ImageFields.ImFields.xpeaksIn.setText(f'{ImageReader.xpeaks.shape[0]}')
        ImageFields.ImFields.ypeaksIn.setText(f'{ImageReader.ypeaks.shape[0]}')
        #define variables from GUI
        ImageData.num_peaks_x = int(ImageFields.ImFields.xpeaksIn.text())
        ImageData.num_peaks_y = int(ImageFields.ImFields.ypeaksIn.text())
        ImageReader.p1linev1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.min_x], y=[ImageReader.min_y,ImageReader.max_y], pen = ImageReader.gpen)
        ImageReader.p1linev2 = pg.PlotCurveItem(x=[ImageReader.max_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1lineh1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.min_y], pen = ImageReader.gpen)
        ImageReader.p1lineh2 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.max_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.plot4.addItem(ImageReader.p1lineh1)
        ImageReader.plot4.addItem(ImageReader.p1linev1)
        ImageReader.plot4.addItem(ImageReader.p1lineh2)
        ImageReader.plot4.addItem(ImageReader.p1linev2)
        # except:
            # print("Initial Peak Finding Failure, please fill out manually")

    def on_FindPeaks_clicked():
        # ImageReader.edges = feature.canny(ImageReader.binary_image, sigma=5)
        # ImageReader.outlines = np.zeros((*ImageReader.edges.shape,4))
        # ImageReader.outlines[:, :, 0] = 255 * ImageReader.edges
        # ImageReader.outlines[:, :, 3] = 255.0 * ImageReader.edges
        i = 0
        xpeaks=[]
        ypeaks=[]
        for row in ImageData.imgData:
            peaks = scipy.signal.find_peaks(row, height = ImageReader.threshold/3, prominence = 8)
            if peaks[0].shape[0] != 0:
                for peak in peaks[0]:
                    xpeaks.append(peak)
                    ypeaks.append(i)
            i+=1

        #Define Values
        ImageReader.xpeaks = np.array(xpeaks)
        ImageReader.ypeaks = np.array(ypeaks)
        ImageReader.min_y = np.min([ImageReader.edminy, np.min(ImageReader.ypeaks)])
        ImageReader.max_y = np.max([ImageReader.edmaxy, np.max(ImageReader.ypeaks)])
        ImageReader.min_x = np.min([ImageReader.edminx, np.min(ImageReader.xpeaks)])
        ImageReader.max_x = np.max([ImageReader.edmaxx, np.max(ImageReader.xpeaks)])
        #Write Values to GUI
        ImageFields.ImFields.xmaxIn.setText(f'{ImageReader.max_x}')
        ImageFields.ImFields.xminIn.setText(f'{ImageReader.min_x}')
        ImageFields.ImFields.ymaxIn.setText(f'{ImageReader.max_y}')
        ImageFields.ImFields.yminIn.setText(f'{ImageReader.min_y}')
        ImageFields.ImFields.xpeaksIn.setText(f'{ImageReader.xpeaks.shape[0]}')
        ImageFields.ImFields.ypeaksIn.setText(f'{ImageReader.ypeaks.shape[0]}')
        #define variables from GUI
        ImageData.num_peaks_x = int(ImageFields.ImFields.xpeaksIn.text())
        ImageData.num_peaks_y = int(ImageFields.ImFields.ypeaksIn.text())
        
        try:
            ImageReader.p1view.clear()
        except:
            print('no previous view')

        ImageReader.p1view = ImageReader.plot4#.getView()
        ImageReader.outlines = pg.ImageItem(image = ImageReader.outlines)
        ImageReader.p1view.addItem(ImageReader.img)
        
        ImageReader.p1dots1 =  pg.ScatterPlotItem(x=ImageReader.xpeaks, y=ImageReader.ypeaks, pen = 'c', symbol = 'o')
        ImageReader.p1linev1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.min_x], y=[ImageReader.min_y,ImageReader.max_y], pen = ImageReader.gpen)
        ImageReader.p1linev2 = pg.PlotCurveItem(x=[ImageReader.max_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1lineh1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.min_y], pen = ImageReader.gpen)
        ImageReader.p1lineh2 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.max_y,ImageReader.max_y], pen =ImageReader.gpen)
        # ImageReader.p1view = ImageReader.plot1.getView()
        # ImageReader.p1view.addItem(ImageReader.p1dots1)
        # ImageReader.p1view.addItem(ImageReader.outlines)
        ImageReader.p1view.addItem(ImageReader.p1lineh1)
        ImageReader.p1view.addItem(ImageReader.p1linev1)
        ImageReader.p1view.addItem(ImageReader.p1lineh2)
        ImageReader.p1view.addItem(ImageReader.p1linev2)

    def on_SaveIm_clicked(ImageReader):
        saveImageName = QFileDialog.getSaveFileName(ImageReader, "Save Image",path, "Image Files (*.png *.jpg *.bmp)")

    def changeThreshold(value):
        ImageData.reduced = False
        # ImageReader.plot1.clear()
        try:
            ImageReader.edges = feature.canny(ImageReader.binary_image, sigma=value)
            ImageReader.edminy = np.min(np.where(ImageReader.edges ==True)[1])
            ImageReader.edmaxy = np.max(np.where(ImageReader.edges ==True)[1])
            ImageReader.edminx = np.min(np.where(ImageReader.edges ==True)[0])
            ImageReader.edmaxx = np.max(np.where(ImageReader.edges ==True)[0])
        except:
            print("Something went wrong changing thresholds")
        # outlines = np.zeros((*ImageReader.edges.shape,4))
        # outlines[:, :, 0] = 255 * ImageReader.edges
        # outlines[:, :, 3] = 255.0 * ImageReader.edges
        # ImageReader.plot1.setImage(ImageReader.imgData)
        # ImageReader.p1view = ImageReader.plot1.getView()
        try:
            ImageReader.plot4.clear()
        except:
            print('no previous view')
        ImageReader.plot4.addItem(ImageReader.img)
        # ImageReader.outlines = pg.ImageItem(image = outlines)
        ImageReader.min_y = np.min([ImageReader.edminy,np.min(ImageReader.ypeaks)])
        ImageReader.max_y = np.max([ImageReader.edmaxy, np.max(ImageReader.ypeaks)])
        ImageReader.min_x = np.min([ImageReader.edminx, np.min(ImageReader.xpeaks)])
        ImageReader.max_x = np.max([ImageReader.edmaxx,np.max(ImageReader.xpeaks)])
        ImageReader.img = pg.ImageItem(image = ImageData.imgData)
        # ImageReader.plot1.setImage(ImageReader.imgData)
        ImageFields.ImFields.xmaxIn.setText(f'{ImageReader.max_x}')
        ImageFields.ImFields.xminIn.setText(f'{ImageReader.min_x}')
        ImageFields.ImFields.ymaxIn.setText(f'{ImageReader.max_y}')
        ImageFields.ImFields.yminIn.setText(f'{ImageReader.min_y}')

        # ImageReader.p1dots1 =  pg.ScatterPlotItem(x=ImageReader.xpeaks, y=ImageReader.ypeaks, pen = 'c', symbol = 'o')
        ImageReader.p1linev1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.min_x], y=[ImageReader.min_y,ImageReader.max_y], pen = ImageReader.gpen)
        ImageReader.p1linev2 = pg.PlotCurveItem(x=[ImageReader.max_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1lineh1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.min_y], pen = ImageReader.gpen)
        ImageReader.p1lineh2 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.max_y,ImageReader.max_y], pen =ImageReader.gpen)
        # ImageReader.p4view = ImageReader.plot4.getView()
        # ImageReader.p1view.addItem(ImageReader.p1dots1)
        # ImageReader.p4view.addItem(ImageReader.outlines)
        ImageReader.plot4.addItem(ImageReader.p1lineh1)
        ImageReader.plot4.addItem(ImageReader.p1linev1)
        ImageReader.plot4.addItem(ImageReader.p1lineh2)
        ImageReader.plot4.addItem(ImageReader.p1linev2)
        ImageReader.plot2.plot(ImageReader.hour*value, ImageReader.temperature, pen =ImageReader.gpen)     
        ImageReader.plot3.plot(ImageReader.temperature, ImageReader.hour*value, pen =ImageReader.gpen) 

    def changeProminence(value):
        # ImageReader.plot1.clear()
        # ImageReader.edges = feature.canny(ImageReader.binary_image, sigma=value)
        # outlines = np.zeros((*ImageReader.edges.shape,4))
        # outlines[:, :, 0] = 255 * ImageReader.edges
        # outlines[:, :, 3] = 255.0 * ImageReader.edges
        # ImageReader.plot1.setImage(ImageReader.imgData)
        # ImageReader.p1view = ImageReader.plot1.getView()
        i = 0
        xpeaks=[]
        ypeaks=[]
        for row in ImageData.imgData:
            peaks = scipy.signal.find_peaks(row, height = ImageReader.threshold/3, prominence = value)
            if peaks[0].shape[0] != 0:
                for peak in peaks[0]:
                    xpeaks.append(peak)
                    ypeaks.append(i)
            i+=1
        ImageReader.xpeaks = np.array(xpeaks)
        ImageReader.ypeaks = np.array(ypeaks)
        ImageReader.min_y = np.min([np.min(np.where(ImageReader.edges ==True)[1]),np.min(ImageReader.ypeaks)])
        ImageReader.max_y = np.max([np.max(np.where(ImageReader.edges ==True)[1]), np.max(ImageReader.ypeaks)])
        ImageReader.min_x = np.min([np.min(np.where(ImageReader.edges ==True)[0]), np.min(ImageReader.xpeaks)])
        ImageReader.max_x = np.max([np.max(np.where(ImageReader.edges ==True)[0]),np.max(ImageReader.xpeaks)])
        try:
            ImageReader.plot4.clear()
        except:
            print('no previous view')
        ImageReader.plot4.addItem(ImageReader.img)
        # ImageReader.outlines = pg.ImageItem(image = outlines)

        ImageFields.ImFields.xmaxIn.setText(f'{ImageReader.max_x}')
        ImageFields.ImFields.xminIn.setText(f'{ImageReader.min_x}')
        ImageFields.ImFields.ymaxIn.setText(f'{ImageReader.max_y}')
        ImageFields.ImFields.yminIn.setText(f'{ImageReader.min_y}')

        # ImageReader.p1dots1 =  pg.ScatterPlotItem(x=ImageReader.xpeaks, y=ImageReader.ypeaks, pen = 'c', symbol = 'o')
        ImageReader.p1linev1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.min_x], y=[ImageReader.min_y,ImageReader.max_y], pen = ImageReader.gpen)
        ImageReader.p1linev2 = pg.PlotCurveItem(x=[ImageReader.max_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1lineh1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.min_y], pen = ImageReader.gpen)
        ImageReader.p1lineh2 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.max_y,ImageReader.max_y], pen =ImageReader.gpen)
        # ImageReader.p4view = ImageReader.plot4.getView()
        # ImageReader.p1view.addItem(ImageReader.p1dots1)
        # ImageReader.p1view.addItem(ImageReader.outlines)
        ImageReader.plot4.addItem(ImageReader.p1lineh1)
        ImageReader.plot4.addItem(ImageReader.p1linev1)
        ImageReader.plot4.addItem(ImageReader.p1lineh2)
        ImageReader.plot4.addItem(ImageReader.p1linev2)
        ImageReader.plot2.plot(ImageReader.hour, ImageReader.temperature*value, pen =ImageReader.gpen)     
        ImageReader.plot3.plot(ImageReader.temperature*value, ImageReader.hour, pen =ImageReader.gpen) 
    
    def image_GSmatrix(path):
        img = Image.open(path[0]).convert('L')  # convert image to 8-bit grayscale
        WIDTH, HEIGHT = img.size
        data = list(img.getdata()) # convert image data to a list of integers
        # convert that to 2D list (list of lists of integers)
        data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
        data = np.array(data)
        #dataset = pd.DataFrame({'1': data[:, 0], '2': data[:, 1]})
        #for i in range(data.shape[1]-2):
        #    dataset['%i'%(i+2)] = data[:, i+2]
        print(f'min:{data.min()}  max:{data.max()}')
        return data

    def csv_GSmatrix(path):
        data = np.genfromtxt(path[0], delimiter = ',') # convert image to 8-bit grayscale
        print(f'min:{data.min()}  max:{data.max()}')
        return data   

    def on_Reduce_clicked():
        try:
            ImageData.n_holes = int(MaskFields.MaskWidget.numHoles.text())
            ImageData.hole_diameter = float(MaskFields.MaskWidget.diamIn.text())
            ImageData.hole_separation = float(MaskFields.MaskWidget.sepIn.text())
            ImageData.mask_to_screen = float(MaskFields.MaskWidget.Mask2ScrnIn.text())
            ImageData.pixpermm = float(MaskFields.MaskWidget.Calibration.text())
            ImageData.d = (ImageData.hole_separation*ImageData.pixpermm)+ImageData.hole_diameter*ImageData.pixpermm #old version had hole separation /2 consistent with simulation, inconsistent with reality
            ImageData.reduced = True


        except:
            msgBox = QMessageBox()
            msgBox.setWindowIcon(QIcon("mrsPepper.ico"))
            msgBox.setWindowTitle('Mask Read Error')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText('Fill out mask information before reducing peaks')
            msgBox.exec_()
        try:
            ImageData.winfrac = float(ImageFields.ImFields.winfrac.text())
            ImageData.xshift = int(ImageFields.ImFields.x_offsetIn.text())
            ImageData.yshift = int(ImageFields.ImFields.y_offsetIn.text())
        except:
            print("Field Failure, Defaulting to 2")
            ImageData.winfrac = 2
        try:
            ImageData.yshift = int(ImageFields.ImFields.y_offsetIn.text())
        except:
            print("Y Shift Failure, Defaulting to 0")
            ImageData.yshift = 2
        try:
            ImageData.xshift = int(ImageFields.ImFields.x_offsetIn.text())
        except:
            print("X Shift Failure, Defaulting to 0")
            ImageData.xshift = 0
        y1s = []
        y2s = []
        x1s = []
        x2s = []

        for x1, y1 in zip(*np.where(ImageReader.edges)):
            y1s.append(y1+ImageData.xshift)
            x1s.append(x1+ImageData.yshift)
    
        y1s = np.array(y1s)
        x1s = np.array(x1s)
        print(f'edges x1: {x1s.shape[0]}')
        x2s, y2s = ImageReader.cutdown(x1s,y1s,  math.ceil(ImageData.d/ImageData.pixpermm))
        print(f'edges x2: {x2s.shape[0]}')
        print(f'peaks x1: {ImageReader.xpeaks.shape[0]}')
        x1s,y1s = ImageReader.cutdown(ImageReader.xpeaks+ImageData.yshift,ImageReader.ypeaks+ImageData.xshift,  math.ceil(ImageData.d/ImageData.pixpermm))
        print(f'peaks x2: {x1s.shape[0]}')
        ImageData.y3s = []
        ImageData.x3s = []
        ImageData.x3s, ImageData.y3s = ImageReader.cutdown(x2s,y2s,  math.ceil(ImageData.d/ImageData.winfrac))
        print(f'edges x3: {ImageData.x3s.shape[0]}')
        x2s, y2s = ImageReader.cutdown(x1s,y1s,  math.ceil(ImageData.d/ImageData.winfrac))
        print(f'peaks x3: {x2s.shape[0]}')
        peakbool = Mainapp.MainWindow.edgeboolbutt.isChecked()
        # print(peakbool)
        if peakbool == True:
            unusedSpots = np.array(np.meshgrid(ImageData.x3s, ImageData.y3s)).T.reshape(-1,2).T  
            ImageData.x3s, ImageData.y3s = ImageReader.cutdown(x2s,y2s,  math.ceil(ImageData.d/ImageData.winfrac))
        else:
            unusedSpots = np.array(np.meshgrid(x2s, y2s)).T.reshape(-1,2).T
        fitSpots = np.array(np.meshgrid(ImageData.x3s, ImageData.y3s)).T.reshape(-1,2).T
          
        ImageFields.ImFields.ypeaksIn.setText(f'{ImageData.y3s.shape[0]}')
        ImageFields.ImFields.xpeaksIn.setText(f'{ImageData.x3s.shape[0]}')
        locs2 =[]
        for i in range(ImageData.n_holes):
            for j in range(ImageData.n_holes):
                locs2.append([i*ImageData.d-ImageData.d*(ImageData.n_holes-1)/2,j*ImageData.d-(ImageData.d*(ImageData.n_holes-1))/2])
        locs2 = np.array(locs2).T
        ImageData.locsdf = pd.DataFrame({'X':locs2[0]+ImageData.x_offset, 'Y':locs2[1]+ImageData.y_offset})

        # print(fitSpots)
        # print(fitSpots[0])
        ImageData.p4dots1 =  pg.ScatterPlotItem(x=ImageData.locsdf.X, y=ImageData.locsdf.Y, pen = 'c', symbol = 'o')
        ImageData.p4spots1 =  pg.ScatterPlotItem(x=fitSpots[1],y=fitSpots[0], pen = 'g', symbol = 'x')
        ImageData.p4spots2 =  pg.ScatterPlotItem(x=unusedSpots[1],y=unusedSpots[0], pen = 'r', symbol = 'x')
        try:
            ImageReader.plot4.clear()
        except:
            print('no previous view')
        # ImageReader.p4view = ImageReader.plot4.getView()
        ImageReader.plot4.addItem(ImageReader.img)
        ImageReader.plot4.addItem(ImageData.p4dots1)
        ImageReader.plot4.addItem(ImageData.p4spots1)
        ImageReader.plot4.addItem(ImageData.p4spots2)
        # ImageReader.p1linev1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.min_x], y=[ImageReader.min_y,ImageReader.max_y], pen = ImageReader.gpen)
        # ImageReader.p1linev2 = pg.PlotCurveItem(x=[ImageReader.max_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.max_y], pen =ImageReader.gpen)
        # ImageReader.p1lineh1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.min_y], pen = ImageReader.gpen)
        # ImageReader.p1lineh2 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.max_y,ImageReader.max_y], pen =ImageReader.gpen)

        ImageReader.plot4.addItem(ImageReader.p1lineh1)
        ImageReader.plot4.addItem(ImageReader.p1linev1)
        ImageReader.plot4.addItem(ImageReader.p1lineh2)
        ImageReader.plot4.addItem(ImageReader.p1linev2)
    def cutdown( xs, ys, gate):
        arr = np.copy(ys)
        for i in range(arr.shape[0]):
            temp1 = arr[i]
            temp2 = arr[(arr >= temp1 - gate) & (arr <= temp1 + gate)]
            meantemp = round(np.mean(temp2))
            arr[(arr >= temp1 - gate) & (arr <= temp1 + gate)] = meantemp
        arr = np.unique(arr)
        y2s = np.copy(arr)
        arr = np.copy(xs)
        for i in range(arr.shape[0]):
            temp1 = arr[i]
            temp2 = arr[(arr >= temp1 - gate) & (arr <= temp1 + gate)]
            meantemp = round(np.mean(temp2))
            arr[(arr >= temp1 - gate) & (arr <= temp1 + gate)] = meantemp
        arr = np.unique(arr)
        x2s = np.copy(arr) 
        return x2s,y2s
    
class xProjection(PlotWidget):
    def __init__(self, parent=None, background='w', plotItem=None, **kargs):
        super().__init__(parent, background, plotItem, **kargs)

        self.setBackground('w')
        self.hour = np.array([1,2,3,4,5,6,7,8,9,10])
        self.temperature = np.array([30,32,34,32,33,31,29,32,35,45])
        self.rpen = pg.mkPen(color=(255, 0, 0))
        self.plot(self.hour, self.temperature, pen = self.rpen)

class yProjection(PlotWidget):
    def __init__(self, parent=None, background='w', plotItem=None, **kargs):
        super().__init__(parent, background, plotItem, **kargs)
        
        self.setBackground('w')
        self.hour = np.array([1,2,3,4,5,6,7,8,9,10])
        self.temperature = np.array([30,32,34,32,33,31,29,32,35,45])
        self.gpen = pg.mkPen(color=(0, 255, 0))
        self.plot(self.temperature, self.hour, pen = self.gpen)

    
class ImagePlot2(pg.ImageView):
    def __init__(self, parent=None, name="ImageView2", view=None, imageItem=None, levelMode='mono', discreteTimeLine=False, roi=None, normRoi=None, *args):
        super().__init__(parent, name, view, imageItem, levelMode, discreteTimeLine, roi, normRoi, *args)
    
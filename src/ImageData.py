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
        self.central_widget = QWidget() # A QWidget to work as Central Widget
        self.layoutG1 = QGridLayout() # Plots
        self.layoutGR00 = QHBoxLayout() # Plot Row 1
        self.layoutGR01 = QHBoxLayout() # Plot Row 2
        self.layoutGC10 = QVBoxLayout() # Plot Row 2 column 1
        self.layoutGC11 = QVBoxLayout() # Plot Row 2 column 2
        self.layoutGC00 = QVBoxLayout() # Plot Row 1 column 1
        self.layoutGC01 = QVBoxLayout() # Plot Row 1 column 2
        self.setCentralWidget(self.central_widget)

        ImageReader.Slider1 = Sliders.slider1() #Threshold
        ImageReader.Slider2 = Sliders.slider2() #Prominence
        ImageReader.slY = Sliders.slider3()
        ImageReader.slX = Sliders.slider4()

        ImageReader.plot2 = ImageData.xProjection()
        ImageReader.plot3 = ImageData.yProjection()
        ImageReader.plot1 = pg.ImageView(view=pg.PlotItem())#ImageData.ImagePlot1(view=pg.PlotItem())
        ImageReader.plot4 = pg.ImageView(view=pg.PlotItem())#ImageData.ImagePlot2(view=pg.PlotItem())

        self.central_widget.setLayout(self.layoutG1)
        
        self.layoutG1.addWidget(ImageReader.plot2, 1, 0)
        self.layoutG1.addWidget(ImageReader.plot4, 1, 1)
        self.layoutG1.addLayout(self.layoutGR00, 0, 0)
        self.layoutG1.addLayout(self.layoutGR01, 0, 1)
        self.layoutG1.addLayout(self.layoutGC10, 1, 0)
        self.layoutGR00.addLayout(self.layoutGC00)
        self.layoutGC00.addWidget(ImageReader.Slider1)
        self.layoutGC00.addWidget(ImageReader.plot1)
        self.layoutGR00.addWidget(ImageReader.Slider2)

        self.layoutGR01.addWidget(ImageReader.plot3)
        self.layoutGR01.addWidget(ImageReader.slY)

        self.layoutGC10.addWidget(ImageReader.plot2)
        self.layoutGC10.addWidget(ImageReader.slX)

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
            ImageReader.p1view.clear()
        except:
            print('no previous view')
        ImageData.imgData = ImageData.imgData.T
        ImageData.x_offset = ImageData.imgData.shape[0]/2
        ImageData.y_offset = ImageData.imgData.shape[1]/2
        ImageReader.img = pg.ImageItem(image = ImageData.imgData)
        ImageReader.img2 = pg.ImageItem(image = ImageData.imgData) 
        ImageReader.p4view = ImageReader.plot4.getView()
        ImageReader.p4view.addItem(ImageReader.img2)

        ImageReader.p1view = ImageReader.plot1.getView()
        ImageReader.p1view.addItem(ImageReader.img)


    def on_FindPeaks_clicked():
        ImageReader.threshold = filters.threshold_otsu(ImageData.imgData)
        ImageReader.binary_image = ImageData.imgData > ImageReader.threshold/2
        ImageReader.edges = feature.canny(ImageReader.binary_image, sigma=5)
        ImageReader.outlines = np.zeros((*ImageReader.edges.shape,4))
        ImageReader.outlines[:, :, 0] = 255 * ImageReader.edges
        ImageReader.outlines[:, :, 3] = 255.0 * ImageReader.edges
        i = 0
        xpeaks=[]
        ypeaks=[]
        for row in ImageData.imgData:
            peaks = scipy.signal.find_peaks(row, height = ImageReader.threshold/2, prominence = 8)
            if peaks[0].shape[0] != 0:
                for peak in peaks[0]:
                    xpeaks.append(peak)
                    ypeaks.append(i)
            i+=1

        #Define Values
        ImageReader.xpeaks = np.array(xpeaks)
        ImageReader.ypeaks = np.array(ypeaks)
        ImageReader.min_y = np.min([np.min(np.where(ImageReader.edges ==True)[1]), np.min(ImageReader.ypeaks)])
        ImageReader.max_y = np.max([np.max(np.where(ImageReader.edges ==True)[1]), np.max(ImageReader.ypeaks)])
        ImageReader.min_x = np.min([np.min(np.where(ImageReader.edges ==True)[0]), np.min(ImageReader.xpeaks)])
        ImageReader.max_x = np.max([np.max(np.where(ImageReader.edges ==True)[0]), np.max(ImageReader.xpeaks)])
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

        ImageReader.p1view = ImageReader.plot1.getView()
        ImageReader.outlines = pg.ImageItem(image = ImageReader.outlines)
        ImageReader.p1view.addItem(ImageReader.img)
        
        ImageReader.p1dots1 =  pg.ScatterPlotItem(x=ImageReader.xpeaks, y=ImageReader.ypeaks, pen = 'c', symbol = 'o')
        ImageReader.p1linev1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.min_x], y=[ImageReader.min_y,ImageReader.max_y], pen = ImageReader.gpen)
        ImageReader.p1linev2 = pg.PlotCurveItem(x=[ImageReader.max_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1lineh1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.min_y], pen = ImageReader.gpen)
        ImageReader.p1lineh2 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.max_y,ImageReader.max_y], pen =ImageReader.gpen)
        # ImageReader.p1view = ImageReader.plot1.getView()
        ImageReader.p1view.addItem(ImageReader.p1dots1)
        ImageReader.p1view.addItem(ImageReader.outlines)
        ImageReader.p1view.addItem(ImageReader.p1lineh1)
        ImageReader.p1view.addItem(ImageReader.p1linev1)
        ImageReader.p1view.addItem(ImageReader.p1lineh2)
        ImageReader.p1view.addItem(ImageReader.p1linev2)

    def on_SaveIm_clicked(ImageReader):
        saveImageName = QFileDialog.getSaveFileName(ImageReader, "Save Image",path, "Image Files (*.png *.jpg *.bmp)")

    def changeThreshold(value):
        ImageData.reduced = False
        # ImageReader.plot1.clear()
        ImageReader.edges = feature.canny(ImageReader.binary_image, sigma=value)
        outlines = np.zeros((*ImageReader.edges.shape,4))
        outlines[:, :, 0] = 255 * ImageReader.edges
        outlines[:, :, 3] = 255.0 * ImageReader.edges
        # ImageReader.plot1.setImage(ImageReader.imgData)
        # ImageReader.p1view = ImageReader.plot1.getView()
        try:
            ImageReader.p1view.clear()
        except:
            print('no previous view')
        ImageReader.p1view.addItem(ImageReader.img)
        ImageReader.outlines = pg.ImageItem(image = outlines)
        ImageReader.min_y = np.min([np.min(np.where(ImageReader.edges ==True)[1]),np.min(ImageReader.ypeaks)])
        ImageReader.max_y = np.max([np.max(np.where(ImageReader.edges ==True)[1]), np.max(ImageReader.ypeaks)])
        ImageReader.min_x = np.min([np.min(np.where(ImageReader.edges ==True)[0]), np.min(ImageReader.xpeaks)])
        ImageReader.max_x = np.max([np.max(np.where(ImageReader.edges ==True)[0]),np.max(ImageReader.xpeaks)])
        ImageReader.img = pg.ImageItem(image = ImageData.imgData)
        # ImageReader.plot1.setImage(ImageReader.imgData)
        ImageFields.ImFields.xmaxIn.setText(f'{ImageReader.max_x}')
        ImageFields.ImFields.xminIn.setText(f'{ImageReader.min_x}')
        ImageFields.ImFields.ymaxIn.setText(f'{ImageReader.max_y}')
        ImageFields.ImFields.yminIn.setText(f'{ImageReader.min_y}')

        ImageReader.p1dots1 =  pg.ScatterPlotItem(x=ImageReader.xpeaks, y=ImageReader.ypeaks, pen = 'c', symbol = 'o')
        ImageReader.p1linev1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.min_x], y=[ImageReader.min_y,ImageReader.max_y], pen = ImageReader.gpen)
        ImageReader.p1linev2 = pg.PlotCurveItem(x=[ImageReader.max_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1lineh1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.min_y], pen = ImageReader.gpen)
        ImageReader.p1lineh2 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.max_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1view = ImageReader.plot1.getView()
        ImageReader.p1view.addItem(ImageReader.p1dots1)
        ImageReader.p1view.addItem(ImageReader.outlines)
        ImageReader.p1view.addItem(ImageReader.p1lineh1)
        ImageReader.p1view.addItem(ImageReader.p1linev1)
        ImageReader.p1view.addItem(ImageReader.p1lineh2)
        ImageReader.p1view.addItem(ImageReader.p1linev2)
        ImageReader.plot2.plot(ImageReader.hour*value, ImageReader.temperature, pen =ImageReader.gpen)     
        ImageReader.plot3.plot(ImageReader.temperature, ImageReader.hour*value, pen =ImageReader.gpen) 

    def changeProminence(value):
        # ImageReader.plot1.clear()
        # ImageReader.edges = feature.canny(ImageReader.binary_image, sigma=value)
        outlines = np.zeros((*ImageReader.edges.shape,4))
        outlines[:, :, 0] = 255 * ImageReader.edges
        outlines[:, :, 3] = 255.0 * ImageReader.edges
        # ImageReader.plot1.setImage(ImageReader.imgData)
        # ImageReader.p1view = ImageReader.plot1.getView()
        i = 0
        xpeaks=[]
        ypeaks=[]
        for row in ImageData.imgData:
            peaks = scipy.signal.find_peaks(row, height = ImageReader.threshold/2, prominence = value)
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
            ImageReader.p1view.clear()
        except:
            print('no previous view')
        ImageReader.p1view.addItem(ImageReader.img)
        ImageReader.outlines = pg.ImageItem(image = outlines)

        ImageFields.ImFields.xmaxIn.setText(f'{ImageReader.max_x}')
        ImageFields.ImFields.xminIn.setText(f'{ImageReader.min_x}')
        ImageFields.ImFields.ymaxIn.setText(f'{ImageReader.max_y}')
        ImageFields.ImFields.yminIn.setText(f'{ImageReader.min_y}')

        ImageReader.p1dots1 =  pg.ScatterPlotItem(x=ImageReader.xpeaks, y=ImageReader.ypeaks, pen = 'c', symbol = 'o')
        ImageReader.p1linev1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.min_x], y=[ImageReader.min_y,ImageReader.max_y], pen = ImageReader.gpen)
        ImageReader.p1linev2 = pg.PlotCurveItem(x=[ImageReader.max_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1lineh1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.min_y], pen = ImageReader.gpen)
        ImageReader.p1lineh2 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.max_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1view = ImageReader.plot1.getView()
        ImageReader.p1view.addItem(ImageReader.p1dots1)
        ImageReader.p1view.addItem(ImageReader.outlines)
        ImageReader.p1view.addItem(ImageReader.p1lineh1)
        ImageReader.p1view.addItem(ImageReader.p1linev1)
        ImageReader.p1view.addItem(ImageReader.p1lineh2)
        ImageReader.p1view.addItem(ImageReader.p1linev2)
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
            ImageData.d = (ImageData.hole_separation*ImageData.pixpermm)/2+ImageData.hole_diameter*ImageData.pixpermm

            ImageData.reduced = True
        except:
            msgBox = QMessageBox()
            msgBox.setWindowIcon(QIcon("mrsPepper.png"))
            msgBox.setWindowTitle('Mask Read Error')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText('Fill out mask information before reducing peaks')
            msgBox.exec_()

        y1s = []
        y2s = []
        x1s = []
        x2s = []

        for x1, y1 in zip(*np.where(ImageReader.edges)):
            y1s.append(y1)
            x1s.append(x1)
    
        y1s = np.array(y1s)
        x1s = np.array(x1s)
        x2s, y2s = ImageReader.cutdown(x1s,y1s,  math.ceil(ImageData.d/2))
        ImageData.y3s = []
        ImageData.x3s = []
        ImageData.x3s, ImageData.y3s = ImageReader.cutdown(x2s,y2s,  math.ceil(ImageData.d/2))
        ImageFields.ImFields.ypeaksIn.setText(f'{ImageData.y3s.shape[0]}')
        ImageFields.ImFields.xpeaksIn.setText(f'{ImageData.x3s.shape[0]}')
        locs2 =[]
        for i in range(ImageData.n_holes):
            for j in range(ImageData.n_holes):
                locs2.append([i*ImageData.d-ImageData.d*(ImageData.n_holes-1)/2,j*ImageData.d-(ImageData.d*(ImageData.n_holes-1))/2])
        locs2 = np.array(locs2).T
        ImageData.locsdf = pd.DataFrame({'X':locs2[0]+ImageData.x_offset, 'Y':locs2[1]+ImageData.y_offset})
        ImageData.p4dots1 =  pg.ScatterPlotItem(x=ImageData.locsdf.X, y=ImageData.locsdf.Y, pen = 'c', symbol = 'o')
        try:
            ImageReader.p4view.clear()
        except:
            print('no previous view')
        ImageReader.p4view = ImageReader.plot4.getView()
        ImageReader.p4view.addItem(ImageReader.img2)
        ImageReader.p4view.addItem(ImageData.p4dots1)

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
    
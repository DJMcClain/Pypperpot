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

class ImageReader():
    gpen = pg.mkPen(color=(0, 255, 0))

    def __init__(self):
        super(ImageReader, self).__init__()
    
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
        #print(ImageData.imgData)
        Mainapp.MainWindow.updateImage1(ImageData.imgData)

        ImageReader.threshold = filters.threshold_otsu(ImageData.imgData)
        ImageReader.binary_image = ImageData.imgData > ImageReader.threshold/2
        ImageReader.edges = feature.canny(ImageReader.binary_image, sigma=5)
        # ImageReader.img2 = ImageReader.imgData+ImageReader.edges
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
        ImageReader.xpeaks = np.array(xpeaks)
        ImageReader.ypeaks = np.array(ypeaks)
        ImageReader.min_y = np.min([np.min(np.where(ImageReader.edges ==True)[1]), np.min(ImageReader.ypeaks)])
        ImageReader.max_y = np.max([np.max(np.where(ImageReader.edges ==True)[1]), np.max(ImageReader.ypeaks)])
        ImageReader.min_x = np.min([np.min(np.where(ImageReader.edges ==True)[0]), np.min(ImageReader.xpeaks)])
        ImageReader.max_x = np.max([np.max(np.where(ImageReader.edges ==True)[0]), np.max(ImageReader.xpeaks)])
        try:
            ImageReader.p1view.clear()
        except:
            print('no previous view')
        image = pg.ImageItem(image = ImageData.imgData.T)

        # ImagePlot1.setImage(img = ImageReader.imgData, self = ImagePlot1)
        ImageReader.p1view = ImagePlot1(imageItem=image)
        ImageReader.outlines = pg.ImageItem(image = ImageReader.outlines)
        ImageReader.p1view.addItem(image)

        ImageFields.x_max_read.setText(f'{ImageReader.max_x}')
        Mainapp.MainWindow.xminIn.setText(f'{ImageReader.min_x}')
        Mainapp.MainWindow.ymaxIn.setText(f'{ImageReader.max_y}')
        Mainapp.MainWindow.yminIn.setText(f'{ImageReader.min_y}')
        Mainapp.MainWindow.xpeaksIn.setText(f'{ImageReader.xpeaks.shape[0]}')
        Mainapp.MainWindow.ypeaksIn.setText(f'{ImageReader.ypeaks.shape[0]}')
        ImageReader.p1dots1 =  pg.ScatterPlotItem(x=ImageReader.xpeaks, y=ImageReader.ypeaks, pen = 'c', symbol = 'o')
        ImageReader.p1linev1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.min_x], y=[ImageReader.min_y,ImageReader.max_y], pen = ImageReader.gpen)
        ImageReader.p1linev2 = pg.PlotCurveItem(x=[ImageReader.max_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1lineh1 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.min_y,ImageReader.min_y], pen = ImageReader.gpen)
        ImageReader.p1lineh2 = pg.PlotCurveItem(x=[ImageReader.min_x, ImageReader.max_x], y=[ImageReader.max_y,ImageReader.max_y], pen =ImageReader.gpen)
        ImageReader.p1view = ImagePlot1.getView()
        ImageReader.p1view.addItem(ImageReader.p1dots1)
        ImageReader.p1view.addItem(ImageReader.outlines)
        ImageReader.p1view.addItem(ImageReader.p1lineh1)
        ImageReader.p1view.addItem(ImageReader.p1linev1)
        ImageReader.p1view.addItem(ImageReader.p1lineh2)
        ImageReader.p1view.addItem(ImageReader.p1linev2)
        ImagePlot1.setView(ImageReader.p1view)
        # ImageReader.xminIn.setText(np.min(ImageReader.edges[0]))
        # ImageReader.plot4.setImage(ImageReader.img2)
        # ImageReader.plot1.setImage(dummy)
        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,10))
        # ax[1].imshow(ImageReader.edges, origin='lower')
        # im = ax[0].imshow(ImageReader.imgData, origin='lower')
        # divider = make_axes_locatable(ax[0])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im,cax=cax, orientation='vertical')
        # plt.show()
    def on_SaveIm_clicked(ImageReader):
        saveImageName = QFileDialog.getSaveFileName(ImageReader, "Save Image",path, "Image Files (*.png *.jpg *.bmp)")

    def changeThreshold(ImageReader,value):
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
        ImageReader.img = pg.ImageItem(image = ImageReader.imgData)
        # ImageReader.plot1.setImage(ImageReader.imgData)
        ImageReader.xmaxIn.setText(f'{ImageReader.max_x}')
        ImageReader.xminIn.setText(f'{ImageReader.min_x}')
        ImageReader.ymaxIn.setText(f'{ImageReader.max_y}')
        ImageReader.yminIn.setText(f'{ImageReader.min_y}')

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
    def changeProminence(ImageReader,value):
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
        for row in ImageReader.imgData:
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
        # ImageReader.plot1.setImage(ImageReader.imgData)
        ImageReader.xmaxIn.setText(f'{ImageReader.max_x}')
        ImageReader.xminIn.setText(f'{ImageReader.min_x}')
        ImageReader.ymaxIn.setText(f'{ImageReader.max_y}')
        ImageReader.yminIn.setText(f'{ImageReader.min_y}')
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
        # ImageReader.plot2.plot(ImageReader.hour, ImageReader.temperature*value, pen =ImageReader.gpen)     
        # ImageReader.plot3.plot(ImageReader.temperature*value, ImageReader.hour, pen =ImageReader.gpen) 
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
        data = data 
        print(f'min:{data.min()}  max:{data.max()}')
        return data   

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

class ImagePlot1(pg.ImageView):
    def __init__(self, parent=None, name="ImageView1", view=None, imageItem=None, levelMode='mono', discreteTimeLine=False, roi=None, normRoi=None, *args):
        super().__init__(parent, name, view, imageItem, levelMode, discreteTimeLine, roi, normRoi, *args)
        self.setImage(img = ImageData.imgData)
        self.p1view = self.getView()
        
    def setView(viewer):
        ImagePlot1.setImage(img = viewer)

    def gettheView():
        print("we made it here")
        return ImagePlot1.view
    
class ImagePlot2(pg.ImageView):
    def __init__(self, parent=None, name="ImageView2", view=None, imageItem=None, levelMode='mono', discreteTimeLine=False, roi=None, normRoi=None, *args):
        super().__init__(parent, name, view, imageItem, levelMode, discreteTimeLine, roi, normRoi, *args)
    
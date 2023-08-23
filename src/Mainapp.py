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
        self.layoutG1 = ImageData.ImageReader()
        self.layoutV1 = QVBoxLayout() # File Params column
        self.ImgFields = ImageFields.ImFields()
        self.MskFields = MaskFields.MaskWidget()
        self.layoutH10 = QHBoxLayout() # Fit/Hand Fit
        self.setCentralWidget(self.central_widget)

#button classes to be started

        fit = QPushButton('Fit')
        self.handfit = QPushButton('Hand Fit')
        
#field class to be defined

#Connect your fields to functions
        fit.clicked.connect(self.on_Fit_clicked)
        self.handfit.clicked.connect(self.on_Hand_clicked)

#Set Highest layer layout and work down
        self.central_widget.setLayout(self.layoutH0)
        self.layoutH0.addLayout(self.layoutV1)#column 1
        self.layoutH0.addLayout(self.layoutV0)#column 2

        self.layoutV0.addWidget(self.layoutG1)

        self.layoutV1.addWidget(self.ImgFields)
        self.layoutV1.addWidget(self.MskFields)

        self.layoutV1.addLayout(self.layoutH10)
        self.layoutH10.addWidget(fit)
        self.layoutH10.addWidget(self.handfit)   

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



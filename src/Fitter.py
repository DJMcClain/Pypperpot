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


class Fits(QMainWindow):
    fitbool = False
    def changeFitplots(value, isX):
        ImageData.ImageReader.plot2.clear()
        ImageData.ImageReader.plot3.clear()
        if isX:
            try:
                xprojInt,xprojMean,xprojSig,xprojY = Fits.fitter_func(ImageData.x3s[value:value+1],ImageData.y3s, ImageData.num_peaks_x,ImageData.imgData, math.ceil(ImageData.d/2), True, True)
            except:
                print(f'no peak {value}')
        else:
            try:
                yprojInt,yprojMean,yprojSig,yprojX = Fits.fitter_func(ImageData.y3s[value:value+1],ImageData.x3s, ImageData.num_peaks_y,ImageData.imgData, math.ceil(ImageData.d/2), False, True)
            except:
                print(f'no peak {value}')
                
        # ImageData.ImageReader.plot1.clear()

        bpen = pg.mkPen(color=(0, 0, 255))
        # ImageData.ImageReader.plot1.clear()
        ImageData.p4dots1 =  pg.ScatterPlotItem(x=ImageData.locsdf.X, y=ImageData.locsdf.Y, pen = 'c', symbol = 'o')
        Fits.p4linev1 = pg.PlotCurveItem(x=[ImageData.x3s[value]-math.ceil(ImageData.d/2), ImageData.x3s[value]-math.ceil(ImageData.d/2)], y=[ImageData.ImageReader.min_y-ImageData.d,ImageData.ImageReader.max_y+ImageData.d], pen =  bpen)
        Fits.p4linev2 = pg.PlotCurveItem(x=[ImageData.x3s[value]+math.ceil(ImageData.d/2), ImageData.x3s[value]+math.ceil(ImageData.d/2)], y=[ImageData.ImageReader.min_y-ImageData.d,ImageData.ImageReader.max_y+ImageData.d], pen =  bpen)
        Fits.p4lineh1 = pg.PlotCurveItem(x=[ImageData.ImageReader.min_x-ImageData.d,ImageData.ImageReader.max_x+ImageData.d], y=[ImageData.y3s[value]-math.ceil(ImageData.d/2), ImageData.y3s[value]-math.ceil(ImageData.d/2)], pen = bpen)
        Fits.p4lineh2 = pg.PlotCurveItem(x=[ImageData.ImageReader.min_x-ImageData.d,ImageData.ImageReader.max_x+ImageData.d], y=[ImageData.y3s[value]+math.ceil(ImageData.d/2), ImageData.y3s[value]+math.ceil(ImageData.d/2)], pen = bpen)
        # self.img = pg.ImageItem(image = ImageData.imgData)
        # self.p1view = ImageData.ImageReader.plot1.getView()
        ImageData.ImageReader.p4view.clear()
        ImageData.ImageReader.p4view.addItem(ImageData.ImageReader.img2)
        ImageData.ImageReader.p4view.addItem(ImageData.p4dots1)
        ImageData.ImageReader.p4view.addItem(Fits.p4lineh1)
        ImageData.ImageReader.p4view.addItem(Fits.p4linev1)
        ImageData.ImageReader.p4view.addItem(Fits.p4lineh2)
        ImageData.ImageReader.p4view.addItem(Fits.p4linev2)
        # self.edges = feature.canny(self.binary_image, sigma=value)
        # ImageData.ImageReader.plot4.setImage(self.edges)
        # ImageData.ImageReader.plot4.show()
 
    def on_Fit_clicked():

        Fits.fitbool = True
        #ImageData.readFields()#TODO
        ImageData.num_peaks_x = int(ImageFields.ImFields.xpeaksIn.text())
        ImageData.num_peaks_y = int(ImageFields.ImFields.ypeaksIn.text())
        ImageData.n_holes = int(MaskFields.MaskWidget.numHoles.text())

        ImageData.hole_diameter = float(MaskFields.MaskWidget.diamIn.text())
        ImageData.hole_separation = float(MaskFields.MaskWidget.sepIn.text())
        ImageData.mask_to_screen = float(MaskFields.MaskWidget.Mask2ScrnIn.text())
        ImageData.pixpermm = float(MaskFields.MaskWidget.Calibration.text())
        min_x = int(ImageFields.ImFields.xminIn.text())
        min_y = int(ImageFields.ImFields.yminIn.text())
        max_x = int(ImageFields.ImFields.xmaxIn.text())
        max_y = int(ImageFields.ImFields.ymaxIn.text())

        if ImageData.reduced == False:
            ImageData.ImageReader.on_Reduce_clicked()
        
        # ImageData.d = (hole_separation*pixpermm)/2+hole_diameter*pixpermm
        # for i in range(Fits.n_holes):
        #     for j in range(Fits.n_holes):
        #         locs2.append([i*ImageData.d-ImageData.d*(Fits.n_holes-1)/2,j*ImageData.d-(ImageData.d*(ImageData.n_holes-1))/2])
        # locs2 = np.array(locs2).T
        
        yprojInt,yprojMean,yprojSig,yprojX = Fits.fitter_func(ImageData.y3s,ImageData.x3s, ImageData.num_peaks_y,ImageData.imgData, math.ceil(ImageData.d/2), False, False)
        xprojInt,xprojMean,xprojSig,xprojY = Fits.fitter_func(ImageData.x3s,ImageData.y3s, ImageData.num_peaks_x,ImageData.imgData, math.ceil(ImageData.d/2), True, False)
        Fits.Xprojdf = pd.DataFrame({'Ypos': xprojY, 'Mean': xprojMean, 'Sig': xprojSig,'Int': xprojInt})
        Fits.Yprojdf = pd.DataFrame({'Xpos': yprojX, 'Mean': yprojMean, 'Sig': yprojSig,'Int': yprojInt})

        #Remapping
        Meanx = []
        Intx = []
        Sigx = []
        j = 0
        for i in range(Fits.Xprojdf.shape[0]):
            if i < ImageData.num_peaks_y:
                # print(i)
                Meanx.append(Fits.Xprojdf.Mean[i * ImageData.num_peaks_x])
                Intx.append(Fits.Xprojdf.Int[i *   ImageData.num_peaks_x])
                Sigx.append(Fits.Xprojdf.Sig[i *   ImageData.num_peaks_x])
            else:
                if i%ImageData.num_peaks_y == 0:
                    j=j+1
                Meanx.append(Fits.Xprojdf.Mean[i%ImageData.num_peaks_y*ImageData.num_peaks_x + j])
                Intx.append(  Fits.Xprojdf.Int[i%ImageData.num_peaks_y*ImageData.num_peaks_x + j])
                Sigx.append(  Fits.Xprojdf.Sig[i%ImageData.num_peaks_y*ImageData.num_peaks_x + j])
        Meanx = np.array(Meanx)
        Intx = np.array(Intx)
        Sigx = np.array(Sigx)
        # Fits.locsdf = pd.DataFrame({'X':locs2[0]+ImageData.x_offset, 'Y':locs2[1]+ImageData.y_offset})
        templocs = ImageData.locsdf[(ImageData.locsdf.X > min_x-ImageData.d/2) & (ImageData.locsdf.X < max_x+ImageData.d/2)]
        templocs = templocs[(ImageData.locsdf.Y > min_y-ImageData.d/2) & (ImageData.locsdf.Y < max_y+ImageData.d/2)]
        print(templocs)
        print(f'templocs.X.to_numpy(),{templocs.X.to_numpy().shape[0]}')
        print(f'Meanx,{Meanx.shape[0]}')
        print(f'Sigx,{Sigx.shape[0]}')
        print(f'Intx,{Intx.shape[0]}')
        print(f'templocs.Y.to_numpy(),{templocs.Y.to_numpy().shape[0]}')
        print(f'yprojMean, {len(yprojMean)}')
        print(f'yprojSig,{len(yprojSig)}')
        print(f'yprojInt {len(yprojInt)}')
        Fits.projectionsdf = pd.DataFrame({'HoleX': templocs.X.to_numpy(),
                                           'MeanX': Meanx,
                                           'SigX':  Sigx,
                                           'IntX':  Intx,
                                           'HoleY': templocs.Y.to_numpy(), 
                                           'MeanY': yprojMean, 
                                           'SigY':  yprojSig,
                                           'IntY':  yprojInt})
        ImageData.ImageReader.slY.setValue(4)
        return
    
    def gaussian(x, height, center, sigma, offset):
        return height/(sigma * np.sqrt(2*np.pi))*np.exp(-(x - center)**2/(2*sigma**2)) + offset
    
    def eight_gaussians(x, h1, c1, w1, 
		h2, c2, w2, 
		h3, c3, w3,
		h4, c4, w4,
		h5, c5, w5,
		h6, c6, w6,
        h7, c7, w7,
		h8, c8, w8,
		offset):
        return (Fits.gaussian(x, h1, c1, w1, offset) +
            Fits.gaussian(x, h2, c2, w2, offset) +
            Fits.gaussian(x, h3, c3, w3, offset) + 
            Fits.gaussian(x, h4, c4, w4, offset) + 
            Fits.gaussian(x, h5, c5, w5, offset) + 
            Fits.gaussian(x, h6, c6, w6, offset) + 
            Fits.gaussian(x, h7, c7, w7, offset) +
            Fits.gaussian(x, h8, c8, w8, offset) +
            offset)

    def seven_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		h5, c5, w5,
    		h6, c6, w6,
            h7, c7, w7,
    		offset):
        return  (Fits.gaussian(x, h1, c1, w1, offset) +
                Fits.gaussian(x, h2, c2, w2, offset) +
                Fits.gaussian(x, h3, c3, w3, offset) + 
                Fits.gaussian(x, h4, c4, w4, offset) + 
                Fits.gaussian(x, h5, c5, w5, offset) + 
                Fits.gaussian(x, h6, c6, w6, offset) + 
                Fits.gaussian(x, h7, c7, w7, offset) +
                offset)

    def six_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		h5, c5, w5,
    		h6, c6, w6,
    		offset):
        return (Fits.gaussian(x, h1, c1, w1, offset) +
            Fits.gaussian(x, h3, c3, w3, offset) + 
            Fits.gaussian(x, h4, c4, w4, offset) + 
            Fits.gaussian(x, h2, c2, w2, offset) +
            Fits.gaussian(x, h5, c5, w5, offset) + 
            Fits.gaussian(x, h6, c6, w6, offset) + 
            offset)

    def five_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		h5, c5, w5,
    		offset):
        return (Fits.gaussian(x, h1, c1, w1, offset) +
            Fits.gaussian(x, h2, c2, w2, offset) +
            Fits.gaussian(x, h3, c3, w3, offset) + 
            Fits.gaussian(x, h4, c4, w4, offset) + 
            Fits.gaussian(x, h5, c5, w5, offset) + 
            offset)

    def four_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		offset):
        return (Fits.gaussian(x, h1, c1, w1, offset) +
            Fits.gaussian(x, h2, c2, w2, offset) +
            Fits.gaussian(x, h3, c3, w3, offset) + 
            Fits.gaussian(x, h4, c4, w4, offset) + 
            offset)
    
    def three_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		offset):
        return (Fits.gaussian(x, h1, c1, w1, offset) +
            Fits.gaussian(x, h2, c2, w2, offset) +
            Fits.gaussian(x, h3, c3, w3, offset) + 
            offset) 

    def two_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		offset):
        return (Fits.gaussian(x, h1, c1, w1, offset) +
            Fits.gaussian(x, h2, c2, w2, offset) + 
            offset)
    
    def fitter_func(arr,arr2, num_peaks,img,pixs, isX, plot):
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
                data = np.array(ImageData.imgData[i-pixs:i+pixs,:])
                temp = np.arange(data.shape[1])
                flim = np.arange(temp.min()-1,temp.max(),1)
                for j in range(temp.shape[0]):
                    temp[j] = sum(data[0:,j])
            else:
                data = np.array(ImageData.imgData[:,i-pixs:i+pixs])
                temp = np.arange(data.shape[0])
                flim = np.arange(temp.min()-1, temp.max(),1)
                for j in range(temp.shape[0]):
                    temp[j] = sum(data[j])
            if plot ==True:
                ppen = pg.mkPen(color=(0, 0, 150),width = 2)
                fpen = pg.mkPen(color=(255, 0, 0),width = 2)
                if isX == True:
                    ImageData.ImageReader.plot2.plot(flim,temp, pen = ppen) 
                else:    
                    ImageData.ImageReader.plot3.plot(temp,flim, pen = ppen) 
                # fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(14,10))
                # plt.title(i)
                # ax[0].plot(flim,temp)
                # im = ax[1].imshow(data)
                # divider = make_axes_locatable(ax[1])
                # cax = divider.append_axes('right', size='5%',pad = 0.05)
                # fig.colorbar(im,cax=cax, orientation='vertical')

            if num_peaks == 1:
                errfunc1 = lambda p, x, y: (Fits.gaussian(x, *p) - y)**2
                guess = [18000, arr2[0], 5,ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc1, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, Fits.gaussian(flim, *optim), pen = fpen, label='fit of Gaussian')
                    else:    
                        ImageData.ImageReader.plot3.plot(Fits.gaussian(flim, *optim),flim, pen = fpen, label='fit of Gaussian')
                    # ax[0].plot(flim, gaussian(flim, *optim),c='red', label='fit of Gaussian')
            elif num_peaks == 2:
                errfunc2 = lambda p, x, y: (Fits.two_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 5, 18000, arr2[1], 5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc2, guess[:], args=(flim, temp))
                # ax[0].plot(flim, two_gaussians(flim, *optim),c='red', label='fit of 2 Gaussians')
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, Fits.two_gaussians(flim, *optim),pen = fpen, label='fit of 2 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(Fits.two_gaussians(flim, *optim),flim,pen = fpen, label='fit of 2 Gaussians')
            elif num_peaks == 3:
                errfunc3 = lambda p, x, y: (Fits.three_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 5, 18000, arr2[1], 5,55000, arr2[2],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc3, guess[:], args=(flim, temp))
                # ax[0].plot(flim, three_gaussians(flim, *optim),c='red', label='fit of 3 Gaussians')
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, Fits.three_gaussians(flim, *optim),pen = fpen, label='fit of 3 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(Fits.three_gaussians(flim, *optim),flim,pen = fpen, label='fit of 3 Gaussians')
            elif num_peaks == 4:
                errfunc4 = lambda p, x, y: (Fits.four_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc4, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, Fits.four_gaussians(flim, *optim),pen = fpen, label='fit of 4 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(Fits.four_gaussians(flim, *optim),flim,pen = fpen, label='fit of 4 Gaussians')
                # ax[0].plot(flim, four_gaussians(flim, *optim),c='red', label='fit of 4 Gaussians') 
            elif num_peaks == 5:
                errfunc5 = lambda p, x, y: (Fits.five_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc5, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, Fits.five_gaussians(flim, *optim),pen = fpen, label='fit of 5 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(Fits.five_gaussians(flim, *optim),flim,pen = fpen, label='fit of 5 Gaussians')
                # ax[0].plot(flim, five_gaussians(flim, *optim),c='red', label='fit of 5 Gaussians') 
            elif num_peaks == 6:
                errfunc6 = lambda p, x, y: (Fits.six_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc6, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, Fits.six_gaussians(flim, *optim),pen = fpen, label='fit of 6 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(Fits.six_gaussians(flim, *optim),flim,pen = fpen, label='fit of 6 Gaussians')
                # ax[0].plot(flim, six_gaussians(flim, *optim),c='red', label='fit of 6 Gaussians') 
            elif num_peaks == 7:
                errfunc7 = lambda p, x, y: (Fits.seven_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5,120000,arr2[6],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc7, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, Fits.seven_gaussians(flim, *optim),pen = fpen, label='fit of 7 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(Fits.seven_gaussians(flim, *optim),flim,pen = fpen, label='fit of 7 Gaussians')
                # ax[0].plot(flim, seven_gaussians(flim, *optim),c='red', label='fit of 7 Gaussians')
            elif num_peaks == 8:
                errfunc8 = lambda p, x, y: (Fits.eight_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5,120000,arr2[6],5,18000, arr2[7],0.25, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc8, guess[:], args=(flim, temp))
                # ax[0].plot(flim, eight_gaussians(flim, *optim),c='red', label='fit of 8 Gaussians')
                if plot ==True:
                    if isX == True:
                        Fits.plot2.plot(flim, Fits.eight_gaussians(flim, *optim),pen = fpen, label='fit of 8 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(Fits.eight_gaussians(flim, *optim),flim,pen = fpen, label='fit of 8 Gaussians')
            n = 0
            for k in range(num_peaks):
                projsInt.append(optim[n])
                projsMean.append(optim[n+1])
                projsSig.append(optim[n+2])
                projsPos.append(i)
                n = n+3
        return projsInt,projsMean,projsSig,projsPos



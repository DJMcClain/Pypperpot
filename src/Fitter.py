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
import ResultFields

fitbool = False
def gaussian(x, height, center, sigma, offset):
    return height/(sigma * np.sqrt(2*np.pi))*np.exp(-(x - center)**2/(2*sigma**2)) + offset

def changeFitplots(value, isX):
      
    if isX:
        ImageData.ImageReader.plot2.clear()
        try:        
            xprojInt,xprojMean,xprojSig,xprojY = MultiFits.fitter_func(ImageData.x3s[value:value+1],ImageData.y3s, ImageData.num_peaks_x,ImageData.imgData, math.ceil(ImageData.d/ImageData.winfrac), True, True)
        except:
            print(f'no peak {value}')
    else:
        ImageData.ImageReader.plot3.clear()
        try:
            yprojInt,yprojMean,yprojSig,yprojX = MultiFits.fitter_func(ImageData.y3s[value:value+1],ImageData.x3s, ImageData.num_peaks_y,ImageData.imgData, math.ceil(ImageData.d/ImageData.winfrac), False, True)
        except:
            print(f'no peak {value}')
            
    # ImageData.ImageReader.plot1.clear()

    bpen = pg.mkPen(color=(0, 0, 255))
    # ImageData.ImageReader.plot1.clear()
    ImageData.p4dots1 =  pg.ScatterPlotItem(x=ImageData.locsdf.X, y=ImageData.locsdf.Y, pen = 'c', symbol = 'o')
    MultiFits.p4linev1 = pg.PlotCurveItem(x=[ImageData.x3s[value]-math.ceil(ImageData.d/ImageData.winfrac), ImageData.x3s[value]-math.ceil(ImageData.d/ImageData.winfrac)], y=[ImageData.ImageReader.min_y-ImageData.d,ImageData.ImageReader.max_y+ImageData.d], pen =  bpen)
    MultiFits.p4linev2 = pg.PlotCurveItem(x=[ImageData.x3s[value]+math.ceil(ImageData.d/ImageData.winfrac), ImageData.x3s[value]+math.ceil(ImageData.d/ImageData.winfrac)], y=[ImageData.ImageReader.min_y-ImageData.d,ImageData.ImageReader.max_y+ImageData.d], pen =  bpen)
    MultiFits.p4lineh1 = pg.PlotCurveItem(x=[ImageData.ImageReader.min_x-ImageData.d,ImageData.ImageReader.max_x+ImageData.d], y=[ImageData.y3s[value]-math.ceil(ImageData.d/ImageData.winfrac), ImageData.y3s[value]-math.ceil(ImageData.d/ImageData.winfrac)], pen = bpen)
    MultiFits.p4lineh2 = pg.PlotCurveItem(x=[ImageData.ImageReader.min_x-ImageData.d,ImageData.ImageReader.max_x+ImageData.d], y=[ImageData.y3s[value]+math.ceil(ImageData.d/ImageData.winfrac), ImageData.y3s[value]+math.ceil(ImageData.d/ImageData.winfrac)], pen = bpen)
    # self.img = pg.ImageItem(image = ImageData.imgData)
    # self.p1view = ImageData.ImageReader.plot1.getView()
    ImageData.ImageReader.p4view.clear()
    ImageData.ImageReader.p4view.addItem(ImageData.ImageReader.img2)
    ImageData.ImageReader.p4view.addItem(ImageData.p4dots1)
    ImageData.ImageReader.p4view.addItem(MultiFits.p4lineh1)
    ImageData.ImageReader.p4view.addItem(MultiFits.p4linev1)
    ImageData.ImageReader.p4view.addItem(MultiFits.p4lineh2)
    ImageData.ImageReader.p4view.addItem(MultiFits.p4linev2)
    # self.edges = feature.canny(self.binary_image, sigma=value)
    # ImageData.ImageReader.plot4.setImage(self.edges)
    # ImageData.ImageReader.plot4.show()
    
class MultiFits(QMainWindow):
 
    def on_MultiFit_clicked():

        MultiFits.fitbool = True
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
        # for i in range(MultiFits.n_holes):
        #     for j in range(MultiFits.n_holes):
        #         locs2.append([i*ImageData.d-ImageData.d*(MultiFits.n_holes-1)/2,j*ImageData.d-(ImageData.d*(ImageData.n_holes-1))/2])
        # locs2 = np.array(locs2).T
        
        yprojInt,yprojMean,yprojSig,yprojX = MultiFits.fitter_func(ImageData.y3s,ImageData.x3s, ImageData.num_peaks_y,ImageData.imgData, math.ceil(ImageData.d/ImageData.winfrac), False, False)
        xprojInt,xprojMean,xprojSig,xprojY = MultiFits.fitter_func(ImageData.x3s,ImageData.y3s, ImageData.num_peaks_x,ImageData.imgData, math.ceil(ImageData.d/ImageData.winfrac), True, False)
        MultiFits.Xprojdf = pd.DataFrame({'Ypos': xprojY, 'Mean': xprojMean, 'Sig': xprojSig,'Int': xprojInt})
        MultiFits.Yprojdf = pd.DataFrame({'Xpos': yprojX, 'Mean': yprojMean, 'Sig': yprojSig,'Int': yprojInt})

        #Remapping
        Meanx = []
        Intx = []
        Sigx = []
        j = 0
        for i in range(MultiFits.Xprojdf.shape[0]):
            if i < ImageData.num_peaks_y:
                # print(i)
                Meanx.append(MultiFits.Xprojdf.Mean[i * ImageData.num_peaks_x])
                Intx.append(MultiFits.Xprojdf.Int[i *   ImageData.num_peaks_x])
                Sigx.append(MultiFits.Xprojdf.Sig[i *   ImageData.num_peaks_x])
            else:
                if i%ImageData.num_peaks_y == 0:
                    j=j+1
                Meanx.append(MultiFits.Xprojdf.Mean[i%ImageData.num_peaks_y*ImageData.num_peaks_x + j])
                Intx.append(  MultiFits.Xprojdf.Int[i%ImageData.num_peaks_y*ImageData.num_peaks_x + j])
                Sigx.append(  MultiFits.Xprojdf.Sig[i%ImageData.num_peaks_y*ImageData.num_peaks_x + j])
        Meanx = np.array(Meanx)
        Intx = np.array(Intx)
        Sigx = np.array(Sigx)
        # MultiFits.locsdf = pd.DataFrame({'X':locs2[0]+ImageData.x_offset, 'Y':locs2[1]+ImageData.y_offset})
        templocs = ImageData.locsdf[(ImageData.locsdf.X > min_x-ImageData.d/ImageData.winfrac) & (ImageData.locsdf.X < max_x+ImageData.d/ImageData.winfrac)]
        templocs = templocs[(ImageData.locsdf.Y > min_y-ImageData.d/ImageData.winfrac) & (ImageData.locsdf.Y < max_y+ImageData.d/ImageData.winfrac)]
        print(templocs)
        print(f'templocs.X.to_numpy(),{templocs.X.to_numpy().shape[0]}')
        print(f'Meanx,{Meanx.shape[0]}')
        print(f'Sigx,{Sigx.shape[0]}')
        print(f'Intx,{Intx.shape[0]}')
        print(f'templocs.Y.to_numpy(),{templocs.Y.to_numpy().shape[0]}')
        print(f'yprojMean, {len(yprojMean)}')
        print(f'yprojSig,{len(yprojSig)}')
        print(f'yprojInt {len(yprojInt)}')
        MultiFits.projectionsdf = pd.DataFrame({'HoleX': templocs.X.to_numpy(),
                                           'MeanX': Meanx,
                                           'SigX':  Sigx,
                                           'IntX':  Intx,
                                           'HoleY': templocs.Y.to_numpy(), 
                                           'MeanY': yprojMean, 
                                           'SigY':  yprojSig,
                                           'IntY':  yprojInt})
        ImageData.ImageReader.slY.setValue(4)
        ImageData.ImageReader.slX.setValue(4)
        return
      
    def eight_gaussians(x, h1, c1, w1, 
		h2, c2, w2, 
		h3, c3, w3,
		h4, c4, w4,
		h5, c5, w5,
		h6, c6, w6,
        h7, c7, w7,
		h8, c8, w8,
		offset):
        return (MultiFits.gaussian(x, h1, c1, w1, offset) +
            MultiFits.gaussian(x, h2, c2, w2, offset) +
            MultiFits.gaussian(x, h3, c3, w3, offset) + 
            MultiFits.gaussian(x, h4, c4, w4, offset) + 
            MultiFits.gaussian(x, h5, c5, w5, offset) + 
            MultiFits.gaussian(x, h6, c6, w6, offset) + 
            MultiFits.gaussian(x, h7, c7, w7, offset) +
            MultiFits.gaussian(x, h8, c8, w8, offset) +
            offset)

    def seven_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		h5, c5, w5,
    		h6, c6, w6,
            h7, c7, w7,
    		offset):
        return  (MultiFits.gaussian(x, h1, c1, w1, offset) +
                MultiFits.gaussian(x, h2, c2, w2, offset) +
                MultiFits.gaussian(x, h3, c3, w3, offset) + 
                MultiFits.gaussian(x, h4, c4, w4, offset) + 
                MultiFits.gaussian(x, h5, c5, w5, offset) + 
                MultiFits.gaussian(x, h6, c6, w6, offset) + 
                MultiFits.gaussian(x, h7, c7, w7, offset) +
                offset)

    def six_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		h5, c5, w5,
    		h6, c6, w6,
    		offset):
        return (MultiFits.gaussian(x, h1, c1, w1, offset) +
            MultiFits.gaussian(x, h3, c3, w3, offset) + 
            MultiFits.gaussian(x, h4, c4, w4, offset) + 
            MultiFits.gaussian(x, h2, c2, w2, offset) +
            MultiFits.gaussian(x, h5, c5, w5, offset) + 
            MultiFits.gaussian(x, h6, c6, w6, offset) + 
            offset)

    def five_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		h5, c5, w5,
    		offset):
        return (MultiFits.gaussian(x, h1, c1, w1, offset) +
            MultiFits.gaussian(x, h2, c2, w2, offset) +
            MultiFits.gaussian(x, h3, c3, w3, offset) + 
            MultiFits.gaussian(x, h4, c4, w4, offset) + 
            MultiFits.gaussian(x, h5, c5, w5, offset) + 
            offset)

    def four_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		h4, c4, w4,
    		offset):
        return (MultiFits.gaussian(x, h1, c1, w1, offset) +
            MultiFits.gaussian(x, h2, c2, w2, offset) +
            MultiFits.gaussian(x, h3, c3, w3, offset) + 
            MultiFits.gaussian(x, h4, c4, w4, offset) + 
            offset)
    
    def three_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		h3, c3, w3,
    		offset):
        return (MultiFits.gaussian(x, h1, c1, w1, offset) +
            MultiFits.gaussian(x, h2, c2, w2, offset) +
            MultiFits.gaussian(x, h3, c3, w3, offset) + 
            offset) 

    def two_gaussians(x, h1, c1, w1, 
    		h2, c2, w2, 
    		offset):
        return (MultiFits.gaussian(x, h1, c1, w1, offset) +
            MultiFits.gaussian(x, h2, c2, w2, offset) + 
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
                errfunc1 = lambda p, x, y: (MultiFits.gaussian(x, *p) - y)**2
                guess = [18000, arr2[0], 5,ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc1, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, MultiFits.gaussian(flim, *optim), pen = fpen, label='fit of Gaussian')
                    else:    
                        ImageData.ImageReader.plot3.plot(MultiFits.gaussian(flim, *optim),flim, pen = fpen, label='fit of Gaussian')
                    # ax[0].plot(flim, gaussian(flim, *optim),c='red', label='fit of Gaussian')
            elif num_peaks == 2:
                errfunc2 = lambda p, x, y: (MultiFits.two_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 5, 18000, arr2[1], 5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc2, guess[:], args=(flim, temp))
                # ax[0].plot(flim, two_gaussians(flim, *optim),c='red', label='fit of 2 Gaussians')
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, MultiFits.two_gaussians(flim, *optim),pen = fpen, label='fit of 2 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(MultiFits.two_gaussians(flim, *optim),flim,pen = fpen, label='fit of 2 Gaussians')
            elif num_peaks == 3:
                errfunc3 = lambda p, x, y: (MultiFits.three_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 5, 18000, arr2[1], 5,55000, arr2[2],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc3, guess[:], args=(flim, temp))
                # ax[0].plot(flim, three_gaussians(flim, *optim),c='red', label='fit of 3 Gaussians')
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, MultiFits.three_gaussians(flim, *optim),pen = fpen, label='fit of 3 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(MultiFits.three_gaussians(flim, *optim),flim,pen = fpen, label='fit of 3 Gaussians')
            elif num_peaks == 4:
                errfunc4 = lambda p, x, y: (MultiFits.four_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc4, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, MultiFits.four_gaussians(flim, *optim),pen = fpen, label='fit of 4 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(MultiFits.four_gaussians(flim, *optim),flim,pen = fpen, label='fit of 4 Gaussians')
                # ax[0].plot(flim, four_gaussians(flim, *optim),c='red', label='fit of 4 Gaussians') 
            elif num_peaks == 5:
                errfunc5 = lambda p, x, y: (MultiFits.five_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc5, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, MultiFits.five_gaussians(flim, *optim),pen = fpen, label='fit of 5 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(MultiFits.five_gaussians(flim, *optim),flim,pen = fpen, label='fit of 5 Gaussians')
                # ax[0].plot(flim, five_gaussians(flim, *optim),c='red', label='fit of 5 Gaussians') 
            elif num_peaks == 6:
                errfunc6 = lambda p, x, y: (MultiFits.six_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc6, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, MultiFits.six_gaussians(flim, *optim),pen = fpen, label='fit of 6 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(MultiFits.six_gaussians(flim, *optim),flim,pen = fpen, label='fit of 6 Gaussians')
                # ax[0].plot(flim, six_gaussians(flim, *optim),c='red', label='fit of 6 Gaussians') 
            elif num_peaks == 7:
                errfunc7 = lambda p, x, y: (MultiFits.seven_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5,120000,arr2[6],5, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc7, guess[:], args=(flim, temp))
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, MultiFits.seven_gaussians(flim, *optim),pen = fpen, label='fit of 7 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(MultiFits.seven_gaussians(flim, *optim),flim,pen = fpen, label='fit of 7 Gaussians')
                # ax[0].plot(flim, seven_gaussians(flim, *optim),c='red', label='fit of 7 Gaussians')
            elif num_peaks == 8:
                errfunc8 = lambda p, x, y: (MultiFits.eight_gaussians(x, *p) - y)**2
                guess = [18000, arr2[0], 0.25, 18000, arr2[1], 5,55000, arr2[2],5,100000,arr2[3],5,
                         120000,arr2[4],5,150000,arr2[5],5,120000,arr2[6],5,18000, arr2[7],0.25, ImageData.ImageReader.threshold]
                optim, success = optimize.leastsq(errfunc8, guess[:], args=(flim, temp))
                # ax[0].plot(flim, eight_gaussians(flim, *optim),c='red', label='fit of 8 Gaussians')
                if plot ==True:
                    if isX == True:
                        ImageData.ImageReader.plot2.plot(flim, MultiFits.eight_gaussians(flim, *optim),pen = fpen, label='fit of 8 Gaussians')
                    else:    
                        ImageData.ImageReader.plot3.plot(MultiFits.eight_gaussians(flim, *optim),flim,pen = fpen, label='fit of 8 Gaussians')
            n = 0
            for k in range(num_peaks):
                projsInt.append(optim[n])
                projsMean.append(optim[n+1])
                projsSig.append(optim[n+2])
                projsPos.append(i)
                n = n+3
        return projsInt,projsMean,projsSig,projsPos

class PeakByPeakFits():
  
    def on_pbpFit_clicked():

        fitbool = True
        #ImageData.readFields()#TODO
        ImageData.num_peaks_x = int(ImageFields.ImFields.xpeaksIn.text())
        ImageData.num_peaks_y = int(ImageFields.ImFields.ypeaksIn.text())
        ImageData.n_holes = int(MaskFields.MaskWidget.numHoles.text())

        ImageData.hole_diameter = float(MaskFields.MaskWidget.diamIn.text())
        ImageData.hole_separation = float(MaskFields.MaskWidget.sepIn.text())
        ImageData.mask_to_screen = float(MaskFields.MaskWidget.Mask2ScrnIn.text())
        ImageData.pixpermm = float(MaskFields.MaskWidget.Calibration.text())
        try:
            puncert = float(MaskFields.MaskWidget.puncert.text()) #pix/mm
        except:
            puncert = 0.05
        try:
            sigL = float(MaskFields.MaskWidget.sigL.text())#mm
        except:
            sigL = 0.05
        try:
            hole_err = float(MaskFields.MaskWidget.hole_err.text())#mm
            PeakByPeakFits.hole_errmm = hole_err #mm
            hole_err = ImageData.hole_separation * ImageData.pixpermm * np.sqrt((puncert/ImageData.pixpermm)**2+(hole_err/ImageData.hole_separation)**2)#px
        except:
            hole_err = 0.005#mm
            PeakByPeakFits.hole_errmm = hole_err #mm
            hole_err = ImageData.hole_separation * ImageData.pixpermm * np.sqrt((puncert/ImageData.pixpermm)**2+(hole_err/ImageData.hole_separation)**2)#px

        min_x = int(ImageFields.ImFields.xminIn.text())
        min_y = int(ImageFields.ImFields.yminIn.text())
        max_x = int(ImageFields.ImFields.xmaxIn.text())
        max_y = int(ImageFields.ImFields.ymaxIn.text())

        if ImageData.reduced == False:
            ImageData.ImageReader.on_Reduce_clicked()
        
        # ImageData.d = (hole_separation*pixpermm)/2+hole_diameter*pixpermm
        # for i in range(MultiFits.n_holes):
        #     for j in range(MultiFits.n_holes):
        #         locs2.append([i*ImageData.d-ImageData.d*(MultiFits.n_holes-1)/2,j*ImageData.d-(ImageData.d*(ImageData.n_holes-1))/2])
        # locs2 = np.array(locs2).T
        #FitterFunc(x2s, y2s, spot2, holes, d, image, pixpermm, mask_to_screen, windowfrac, sigL, hole_err = 0)
        spot2, holes = PeakByPeakFits.Mapping()

        intX, PeakByPeakFits.sterxs, hole_x, PeakByPeakFits.xps, PeakByPeakFits.xs, PeakByPeakFits.xperr, stdx, intY, PeakByPeakFits.sterys, hole_y, PeakByPeakFits.yps, PeakByPeakFits.yperr, PeakByPeakFits.ys, stdy, mu4xs, mu4ys = PeakByPeakFits.FitterFunc(ImageData.x3s,ImageData.y3s, spot2, holes, sigL, hole_err,puncert)
        PeakByPeakFits.emitX, PeakByPeakFits.emitY, PeakByPeakFits.emitXerr, PeakByPeakFits.emitYerr = PeakByPeakFits.EmittanceFunction(intX, PeakByPeakFits.sterxs, hole_x, PeakByPeakFits.xps, PeakByPeakFits.xs, PeakByPeakFits.xperr, stdx,mu4xs, intY, PeakByPeakFits.sterys, hole_y, PeakByPeakFits.yps, PeakByPeakFits.yperr, PeakByPeakFits.ys, stdy, mu4ys, sigL, hole_err, puncert)
        image = ImageData.imgData
        x_offset = (image.shape[1])/2
        y_offset = (image.shape[0])/2
        pixpermm = ImageData.pixpermm
        PeakByPeakFits.resultsdfx = pd.DataFrame({'X(mm)':(PeakByPeakFits.xs-x_offset)/pixpermm ,  "X'(mrad)":PeakByPeakFits.xps,  "Xerr":PeakByPeakFits.sterxs/pixpermm, "X'err":PeakByPeakFits.xperr,"IntX":intX})
        PeakByPeakFits.resultsdfy = pd.DataFrame({'Y(mm)':(PeakByPeakFits.ys-y_offset)/pixpermm ,  "Y'(mrad)":PeakByPeakFits.yps,  "Yerr":PeakByPeakFits.sterys/pixpermm, "Y'err":PeakByPeakFits.yperr,"IntY":intY})            
        
       # MultiFits.Xprojdf = pd.DataFrame({'Ypos': hole_y, 'Mean': PeakByPeakFits.xs, 'Sig': stdx,'Int': intX})
       # MultiFits.Yprojdf = pd.DataFrame({'Xpos': hole_x, 'Mean': PeakByPeakFits.ys, 'Sig': stdy,'Int': intY})

        # #Remapping
        # Meanx = []
        # Intx = []
        # Sigx = []
        # j = 0
        # for i in range(MultiFits.Xprojdf.shape[0]):
        #     if i < ImageData.num_peaks_y:
        #         # print(i)
        #         Meanx.append(MultiFits.Xprojdf.Mean[i * ImageData.num_peaks_x])
        #         Intx.append(MultiFits.Xprojdf.Int[i *   ImageData.num_peaks_x])
        #         Sigx.append(MultiFits.Xprojdf.Sig[i *   ImageData.num_peaks_x])
        #     else:
        #         if i%ImageData.num_peaks_y == 0:
        #             j=j+1
        #         Meanx.append(MultiFits.Xprojdf.Mean[i%ImageData.num_peaks_y*ImageData.num_peaks_x + j])
        #         Intx.append(  MultiFits.Xprojdf.Int[i%ImageData.num_peaks_y*ImageData.num_peaks_x + j])
        #         Sigx.append(  MultiFits.Xprojdf.Sig[i%ImageData.num_peaks_y*ImageData.num_peaks_x + j])
        # Meanx = np.array(Meanx)
        # Intx = np.array(Intx)
        # Sigx = np.array(Sigx)

        # templocs = ImageData.locsdf[(ImageData.locsdf.X > min_x-ImageData.d/2) & (ImageData.locsdf.X < max_x+ImageData.d/2)]
        # templocs = templocs[(ImageData.locsdf.Y > min_y-ImageData.d/2) & (ImageData.locsdf.Y < max_y+ImageData.d/2)]
        # print(templocs)
        # print(f'templocs.X.to_numpy(),{templocs.X.to_numpy().shape[0]}')
        # print(f'Meanx,{Meanx.shape[0]}')
        # print(f'Sigx,{Sigx.shape[0]}')
        # print(f'Intx,{Intx.shape[0]}')
        # print(f'templocs.Y.to_numpy(),{templocs.Y.to_numpy().shape[0]}')
        # print(f'yprojMean, {len(ys)}')
        # print(f'yprojSig,{len(stdy)}')
        # print(f'yprojInt {len(intY)}')
        # MultiFits.projectionsdf = pd.DataFrame({'HoleX': templocs.X.to_numpy(),
        #                                    'MeanX': Meanx,
        #                                    'SigX':  Sigx,
        #                                    'IntX':  Intx,
        #                                    'HoleY': templocs.Y.to_numpy(), 
        #                                    'MeanY': ys, 
        #                                    'SigY':  stdy,
        #                                    'IntY':  intY})
        #output resultsdf as header, projectionsdf as file body
        # ImageData.ImageReader.slY.setValue(4)
        # ImageData.ImageReader.slX.setValue(4)
        # print(ImageData.resultsdf())
        return
  
    def on_AutoFit_clicked():
        xemit = []
        xerr = []
        yemit = []
        yerr = []
        avgemit = []
        avgerr = []
        threshold = []
        for i in range(10):
            ImageData.ImageReader.changeThreshold(i+1)
            if ImageData.reduced == False:
                ImageData.ImageReader.on_Reduce_clicked()
                PeakByPeakFits.on_pbpFit_clicked()
                xemit.append(float(ResultFields.ResFields.xemit.text()))
                xerr.append(float(ResultFields.ResFields.xemiterr.text()))
                yemit.append(float(ResultFields.ResFields.yemit.text()))
                yerr.append(float(ResultFields.ResFields.yemiterr.text()))
                avg = (float(ResultFields.ResFields.xemit.text()) + float(ResultFields.ResFields.yemit.text()))/2
                delavg = 1/2 * np.sqrt(float(ResultFields.ResFields.xemiterr.text())**2+float(ResultFields.ResFields.yemiterr.text())**2)
                avgemit.append(avg)
                avgerr.append(delavg)
                threshold.append(i+1)
            else:
                break
        print('-------------------------------------------------------------------------------------------------------')
        print('Threshold \t Avg Emit \t Avg err \t Xemit \t\t xerr \t\t Yemit \t\t yerr')
        for i in range(len((threshold))):
            print(f'\t{threshold[i]} \t {avgemit[i]:.3f} \t\t {avgerr[i]:.3f} \t\t {xemit[i]:.3f} \t\t {xerr[i]:.3f} \t\t {yemit[i]:.3f} \t\t {yerr[i]:.3f}')
        print('-------------------------------------------------------------------------------------------------------')
    def get_ordered_list(points, x, y):
        new_points = []
        new_points = sorted(points,key = lambda p: (p[0] - x)**2 + (p[1] - y)**2)
        return new_points
    
    def Binning(arr):
        spotx = arr.T[0]-ImageData.x_offset
        spoty = arr.T[1]-ImageData.y_offset
        print(spotx)
        binx = np.copy(spotx)
        biny = np.copy(spoty)
        # print(np.unique(spotx).shape[0])
        # print(np.mean(spotx[abs(spotx)==min(abs(spotx))]))
        centerx = np.mean(spotx[abs(spotx)==min(abs(spotx))])
        centery = np.mean(spoty[abs(spoty)==min(abs(spoty))])
    
        # if n_holes%2 !=0: #if our hole number is odd
        i = 0
        j = 0
        posSpotx = spotx[spotx >= 0]
        posSpoty = spoty[spoty >= 0]
        negSpotx = spotx[spotx < 0]
        negSpoty = spoty[spoty < 0]
        binx[spotx == centerx] = 0 
        biny[spoty == centery] = 0 
        try:
            tempx = min(posSpotx[posSpotx > centerx])#1
            
            for spot in np.unique(posSpotx):
                i+=1
                binx[spotx == tempx] = i
            #     print(tempx)
                try:
                    tempx =  min(posSpotx[posSpotx > tempx])

                except:
                    break
        except:
            print("No positive X vals")        
        try:
            tempy = min(posSpoty[posSpoty > centery])#1
            for spot in np.unique(posSpoty):
                j+=1
                biny[spoty == tempy] = j
            #     print(tempy)
                try:
                    tempy =  min(posSpoty[posSpoty > tempy])
                except:
                    break
        except:
            print ('no Positive Y values')  
        try:     
            tempx = max(negSpotx[negSpotx < centerx])#1
             
            i = 0
            
            for spot in np.unique(negSpotx):
                i-=1
                binx[spotx == tempx] = i
            #     print(tempx)
                try:
                    tempx = max(negSpotx[negSpotx < tempx])
                except:
                    break
        except:
            print("No negative X values")
        try:
            j = 0
            tempy = max(negSpoty[negSpoty < centery])#1 
            for spot in np.unique(negSpoty):
                j-=1
                biny[spoty == tempy] = j
            #     print(tempy)
                try:
                    tempy = max(negSpoty[negSpoty < tempy])
                except:
                    break
        except:
            print('no negative Y values')
            # min(spotx[spotx > 0 ])
        # print(binx)
        outarr = np.array([binx,biny]).T
        return(outarr)
    def Mapping():
        locs3 = [ImageData.locsdf['X'], ImageData.locsdf['Y']]
        spots = [] 
        holes = [] 
        for x in ImageData.x3s:
            for y in ImageData.y3s:
                spots.append([x,y])
        for x in np.unique(locs3[0]):
            for y in np.unique(locs3[1]):
                holes.append([x,y])
        spots = np.array(spots)
        holes = np.array(holes)
        ordered_spots = PeakByPeakFits.get_ordered_list(spots, ImageData.x_offset,  ImageData.y_offset)#order by distance away from center
        ordered_holes = PeakByPeakFits.get_ordered_list(holes,  ImageData.x_offset,  ImageData.y_offset)
        ordered_spots = np.array(ordered_spots)
        ordered_holes = np.array(ordered_holes)
        spot2 = ordered_spots#[:4]
        hole2 = ordered_holes#[:4]
        print("spot2")
        print(spot2)
        #attempt to overcome many hole tripping
        spot3 = PeakByPeakFits.Binning(spot2)
        hole3 = PeakByPeakFits.Binning(hole2)
        hole4 = np.copy(spot3)
        spot4 = np.copy(spot3)
        print("spot4")
        print(spot4)
        # print("hole4")
        # print(hole4)
        # print(hole2[(hole3.T[0]==0) & (hole3.T[1]==0)])
        for i in np.unique(spot3).astype(int):
            for j in np.unique(spot3).astype(int):
                # print(i,j)
                try:
                    spot4[(spot3.T[0]==i) & (spot3.T[1]==j)] = spot2[(spot3.T[0]==i) & (spot3.T[1]==j)]
                    hole4[(spot3.T[0]==i) & (spot3.T[1]==j)] = hole2[(hole3.T[0]==i) & (hole3.T[1]==j)]
                except:
                    print("missing holes")
        # holes = []
        # for spot in spot2:
        #     holedistx = np.abs(hole2.T[0] - spot[0])
        #     holedisty = np.abs(hole2.T[1] - spot[1])
        #     holetempx = hole2[np.where(holedistx == np.min(holedistx))]#this is tripping up in many hole cases
        #     holetempy = hole2[np.where(holedisty == np.min(holedisty))]
        #     for x in holetempx:
        #         for y in holetempy:
        #             if x[0] == y[0]:
        #                 if x[1] == y[1]:
        #                     holetemp = x
        #                     break
        #                 else:
        #                     continue
        #             else:
        #                 continue
        #     holes.append(holetemp)    
        # holes = np.array(holes)
        # for spot in ordered_spots:#[4:]:
        #     if spot.T[0] in spot2.T[0]:
        #         tempx = holes.T[0][np.where(spot2.T[0] == spot.T[0])]
        #     elif (spot.T[0] > np.max(spot2.T[0])):
        #           tempx = np.min(ordered_holes.T[0][ordered_holes.T[0]>np.max(holes.T[0])])
        #     else:
        #           tempx = np.max(ordered_holes.T[0][ordered_holes.T[0]<np.min(holes.T[0])])

        #     if spot.T[1] in spot2.T[1]:
        #         tempy = holes.T[1][np.where(spot2.T[1] == spot.T[1])]
        #     elif (spot.T[1] > np.max(spot2.T[1])):
        #           tempy = np.min(ordered_holes.T[1][ordered_holes.T[1]>np.max(holes.T[1])])
        #     else:
        #           tempy = np.max(ordered_holes.T[1][ordered_holes.T[1]<np.min(holes.T[1])])

        #     tempx = np.unique(tempx) 
        #     tempy = np.unique(tempy)
        #     newhole = np.array([tempx[0],tempy[0]])
        #     holes = np.append(holes, [newhole], axis=0)
        #     spot2 = np.append(spot2,[spot], axis=0)

        return spot4.astype(int), hole4 #old spot2 and holes

    def hist_stats(bins, freq):
        total = np.sum(freq)
        mean = np.sum(bins*freq)/total
        vari = np.sum(freq*(bins - mean)**2)/(total - 1)
        stdv = np.sqrt(vari)
        ster = stdv / np.sqrt(total)
        return total, mean, stdv, ster

    def FitterFunc(x2s, y2s, spot2, holes,  sigL, hole_err, puncert):
        d = ImageData.d 
        image = ImageData.imgData
        pixpermm = ImageData.pixpermm
        mask_to_screen = ImageData.mask_to_screen

        pixs = math.ceil(ImageData.d/ImageData.winfrac)
        xs = []
        ys = []
        hole_x = []
        hole_y = []
        xps = []
        yps = []
        xperr = []
        yperr = []
        intX = []
        intY = []
        sterxs = []
        sterys = []
        stdy = []
        stdx = []
        mu4xs = []
        mu4ys = []
        xfail = False
        yfail = False

        for k in range(spot2.shape[0]):
            spot = spot2[k]
            hole = holes[k]
            x = spot[0]
            y = spot[1]
            holex = hole[0]
            holey = hole[1]
            data = np.array(image[y-pixs:y+ pixs,x-pixs:x+pixs])
    #         print(f'({x},{y})')
            tempx = np.arange(data.shape[1])
            tempy = np.arange(data.shape[0])
            flimx = np.arange(tempx.min()+0.0, tempx.max()+1.00,1)
            flimy = np.arange(tempy.min()+0.0, tempy.max()+1.00,1)
            for i in range(tempy.shape[0]):
                tempy[i] = sum(data[:,i])
            for j in range(tempx.shape[0]):
                tempx[j] = sum(data[j])

            totx, meanx, stdvx, sterx = PeakByPeakFits.hist_stats(flimx, tempx)
            toty, meany, stdvy, stery = PeakByPeakFits.hist_stats(flimy, tempy)
        #        print(f'({x},{y})')
    #         print(f'sum: x = {totx:.2f}, y = {toty:.2f}')
    #         print(f'mean: x = {meanx+x-tempx.shape[0]/2:.2f} +/- {sterx:.2f}, y = {meany+y-tempy.shape[0]/2:.2f} +/- {stery:.2f}')
    #         print(f'hole position: x = {holex}, y = {holey}')
    #         print(f'stdv: x = {stdvx:.2f}, y = {stdvy:.2f}')
            try:
                # print('Resituate')
                x = round(meanx+x-tempx.shape[0]/2)#mean within the window moved to the spot position with a zeroed image center
                y = round(meany+y-tempy.shape[0]/2)
                data = np.array(image[y-pixs:y+ pixs,x-pixs:x+pixs])
                tempx = np.arange(data.shape[1])
                tempy = np.arange(data.shape[0])
                flimx = np.arange(tempx.min()+0.00, tempx.max()+1.00,1)
                flimy = np.arange(tempy.min()+0.00, tempy.max()+1.00,1)
                for i in range(tempy.shape[0]):
                    tempy[i] = sum(data[:,i])
                for j in range(tempx.shape[0]):
                    tempx[j] = sum(data[j])
                totx, meanx, stdvx, sterx = PeakByPeakFits.hist_stats(flimx, tempx)
                toty, meany, stdvy, stery = PeakByPeakFits.hist_stats(flimy, tempy)
    #             print(f'sum: x = {totx:.2f}, y = {toty:.2f}')
    #             print(f'mean: x = {meanx+x-tempx.shape[0]/2:.2f} +/- {sterx:.2f}, y = {meany+y-tempy.shape[0]/2:.2f} +/- {stery:.2f}')
    #             print(f'hole position: x = {holex}, y = {holey}')
    #             print(f'stdv: x = {stdvx:.2f}, y = {stdvy:.2f}')
            except:
                print('No resituation, Nan present') #there was not a peak here


            if totx >=150:#make this threshold accessible?

                guessx = [totx,meanx, stdvx,0]
                guessy = [toty,meany, stdvy,0]

                try:
                    optimx, pcov = curve_fit(gaussian, flimx,tempx, p0=guessx)

                except:
                    try:
                        guessx = [totx,meanx, stdvx*2,0]
                        optimx, pcov = curve_fit(gaussian, flimx,tempx, p0=guessx)

                    except:
                        xfail = True
                try:
                    optimy, pcov = curve_fit(gaussian, flimy,tempy, p0=guessy)

                except:
                    try:
                        guessy = [toty,meany, stdvy*3,0]
                        optimy, pcov = curve_fit(gaussian, flimy,tempy, p0=guessy)
                    except:
                        yfail = True


                if xfail == False:
                    totx = optimx[0]
                    meanx = optimx[1]
                    stdvx = optimx[2]
                else:
                    xfail == False
                if yfail == False:
                    toty = optimy[0]
                    meany = optimy[1]
                    stdvy = optimy[2]
                else:
                    yfail == False
                sterx = stdvx / np.sqrt(totx)
                stery = stdvy / np.sqrt(toty)
                mu4y = scipy.stats.moment(tempy, moment=4)
                mu4x = scipy.stats.moment(tempx, moment=4)

            mu4xs.append(mu4x)
            mu4ys.append(mu4y)
            intX.append(totx)
            intY.append(toty)
            sterxs.append(sterx)
            xs.append(meanx - tempx.shape[0]/2 + x)
            stdx.append(stdvx)
            hole_x.append(holex)
            sterys.append(stery)
            ys.append(meany - tempy.shape[0]/2 + y)
            stdy.append(stdvy)
            hole_y.append(holey)
            xps.append((meanx+x-tempx.shape[0]/2-holex)/(pixpermm * mask_to_screen)*1000)#meanx+x-tempx.shape[0]/2
            xperr.append(np.abs(((meanx+x-tempx.shape[0]/2-holex)/(pixpermm * mask_to_screen)*1000)*np.sqrt((stdvx**2+hole_err**2)/((meanx+x-tempx.shape[0]/2-holex)**2)+(puncert/pixpermm)**2+(sigL/mask_to_screen)**2)))
            yps.append((meany+y-tempy.shape[0]/2-holey)/(pixpermm * mask_to_screen)*1000)
            yperr.append(np.abs(((meany+y-tempy.shape[0]/2-holey)/(pixpermm * mask_to_screen)*1000)*np.sqrt((stdvy**2+hole_err**2)/((meany+y-tempy.shape[0]/2-holey)**2)+(puncert/pixpermm)**2+(sigL/mask_to_screen)**2)))

        #         data = np.array(image[x-pixs:x+pixs,:])
        mu4xs = np.array(mu4xs)
        mu4ys = np.array(mu4ys)
        intX = np.array(intX)
        intY = np.array(intY)
        sterxs = np.array(sterxs)
        xs = np.array(xs)
        stdx = np.array(stdx)
        hole_x = np.array(hole_x)
        sterys = np.array(sterys)
        ys = np.array(ys)
        stdy = np.array(stdy)
        hole_y = np.array(hole_y)
        xps = np.array(xps)
        yps = np.array(yps)
        xperr = np.array(xperr)
        yperr = np.array(yperr)

        if np.isnan(xs).any():
            intX = intX[~np.isnan(xs)]
            sterxs = sterxs[~np.isnan(xs)]
            stdx = stdx[~np.isnan(xs)]
            hole_x = hole_x[~np.isnan(xs)]
            xps = xps[~np.isnan(xs)]
            xperr = xperr[~np.isnan(xs)]
            mu4xs = mu4xs[~np.isnan(xs)]
            xs = xs[~np.isnan(xs)]
        if np.isnan(stdx).any():
            intX = intX[~np.isnan(stdx)]
            sterxs = sterxs[~np.isnan(stdx)]
            hole_x = hole_x[~np.isnan(stdx)]
            xps = xps[~np.isnan(stdx)]
            xs = xs[~np.isnan(stdx)]
            xperr = xperr[~np.isnan(stdx)]
            mu4xs = mu4xs[~np.isnan(stdx)]
            stdx = stdx[~np.isnan(stdx)]
        if np.isnan(intX).any():
            sterxs = sterxs[~np.isnan(intX)]
            hole_x = hole_x[~np.isnan(intX)]
            xps = xps[~np.isnan(intX)]
            xperr = xperr[~np.isnan(intX)]
            xs = xs[~np.isnan(intX)]
            stdx = stdx[~np.isnan(intX)]
            mu4xs = mu4xs[~np.isnan(intX)]
            intX = intX[~np.isnan(intX)]  
        if np.isnan(sterxs).any():
            hole_x = hole_x[~np.isnan(sterxs)]
            xps = xps[~np.isnan(sterxs)]
            xperr = xperr[~np.isnan(sterxs)]
            xs = xs[~np.isnan(sterxs)]
            stdx = stdx[~np.isnan(sterxs)]
            intX = intX[~np.isnan(sterxs)]
            mu4xs = mu4xs[~np.isnan(sterxs)]
            sterxs = sterxs[~np.isnan(sterxs)]
        if np.isnan(hole_x).any():
            xps = xps[~np.isnan(hole_x)]
            xperr = xperr[~np.isnan(hole_x)]
            xs = xs[~np.isnan(hole_x)]
            stdx = stdx[~np.isnan(hole_x)]
            intX = intX[~np.isnan(hole_x)]
            sterxs = sterxs[~np.isnan(hole_x)]
            mu4xs = mu4xs[~np.isnan(hole_x)]
            hole_x = hole_x[~np.isnan(hole_x)]
        if np.isnan(xps).any():
            xperr = xperr[~np.isnan(xps)]
            xs = xs[~np.isnan(xps)]
            stdx = stdx[~np.isnan(xps)]
            intX = intX[~np.isnan(xps)]
            sterxs = sterxs[~np.isnan(xps)]
            hole_x = hole_x[~np.isnan(xps)]
            mu4xs = mu4xs[~np.isnan(xps)]
            xps = xps[~np.isnan(xps)]
        if np.isnan(xperr).any():
            xs = xs[~np.isnan(xperr)]
            stdx = stdx[~np.isnan(xperr)]
            intX = intX[~np.isnan(xperr)]
            sterxs = sterxs[~np.isnan(xperr)]
            hole_x = hole_x[~np.isnan(xperr)]
            xps = xps[~np.isnan(xperr)]
            mu4xs = mu4xs[~np.isnan(xperr)]
            xperr = xperr[~np.isnan(xperr)]

        if np.isnan(ys).any():
            intY = intY[~np.isnan(ys)]
            sterys = sterys[~np.isnan(ys)]
            stdy = stdy[~np.isnan(ys)]
            hole_y = hole_y[~np.isnan(ys)]
            yps = yps[~np.isnan(ys)]
            yperr = yperr[~np.isnan(ys)]
            mu4ys = mu4ys[~np.isnan(ys)]
            ys = ys[~np.isnan(ys)]
        if np.isnan(stdy).any():
            intY = intY[~np.isnan(stdy)]
            sterys = sterys[~np.isnan(stdy)]
            hole_y = hole_y[~np.isnan(stdy)]
            yps = yps[~np.isnan(stdy)]
            yperr = yperr[~np.isnan(stdy)]
            ys = ys[~np.isnan(stdy)]
            mu4ys = mu4ys[~np.isnan(stdy)]
            stdy = stdy[~np.isnan(stdy)]
        if np.isnan(intY).any():  
            sterys = sterys[~np.isnan(intY)]
            hole_y = hole_y[~np.isnan(intY)]
            yps = yps[~np.isnan(intY)]
            yperr = yperr[~np.isnan(intY)]
            ys = ys[~np.isnan(intY)]
            stdy = stdy[~np.isnan(intY)]
            mu4ys = mu4ys[~np.isnan(intY)]
            intY = intY[~np.isnan(intY)]       
        if np.isnan(sterys).any():
            hole_y = hole_y[~np.isnan(sterys)]
            yps = yps[~np.isnan(sterys)]
            yperr = yperr[~np.isnan(sterys)]
            ys = ys[~np.isnan(sterys)]
            stdy = stdy[~np.isnan(sterys)]
            intY = intY[~np.isnan(sterys)] 
            mu4ys = mu4ys[~np.isnan(sterys)]
            sterys = sterys[~np.isnan(sterys)]
        if np.isnan(hole_y).any():
            yps = yps[~np.isnan(hole_y)]
            yperr = yperr[~np.isnan(hole_y)]
            ys = ys[~np.isnan(hole_y)]
            stdy = stdy[~np.isnan(hole_y)]
            intY = intY[~np.isnan(hole_y)] 
            sterys = sterys[~np.isnan(hole_y)]
            mu4ys = mu4ys[~np.isnan(hole_y)]
            hole_y = hole_y[~np.isnan(hole_y)]
        if np.isnan(yps).any():
            yperr = yperr[~np.isnan(yps)]
            ys = ys[~np.isnan(yps)]
            stdy = stdy[~np.isnan(yps)]
            intY = intY[~np.isnan(yps)] 
            sterys = sterys[~np.isnan(yps)]
            hole_y = hole_y[~np.isnan(yps)]
            mu4ys = mu4ys[~np.isnan(yps)]
            yps = yps[~np.isnan(yps)]
        if np.isnan(yperr).any():
            ys = ys[~np.isnan(yperr)]
            stdy = stdy[~np.isnan(yperr)]
            intY = intY[~np.isnan(yperr)] 
            sterys = sterys[~np.isnan(yperr)]
            hole_y = hole_y[~np.isnan(yperr)]
            yps = yps[~np.isnan(yperr)]
            mu4ys = mu4ys[~np.isnan(yperr)]
            yperr = yperr[~np.isnan(yperr)]

        if np.isnan(mu4ys).any():
            print('bad mu4 y')
        if np.isnan(mu4xs).any():
            print('bad mu4 x')

        return intX, sterxs, hole_x, xps, xs, xperr, stdx, intY, sterys, hole_y, yps, yperr, ys, stdy, mu4xs, mu4ys

    def EmittanceFunction(intX, sterxs, hole_x, xps, xs, xperr, stdx, mu4xs, intY, sterys, hole_y, yps, yperr, ys, stdy, mu4ys, sigL, hole_err, puncert):
        pixpermm = ImageData.pixpermm
        mask_to_screen = ImageData.mask_to_screen
        image = ImageData.imgData

        x_offset = (image.shape[1])/2
        y_offset = (image.shape[0])/2#px

        meanXtot2 = 1/np.sum(intX)*np.sum(hole_x*intX)
        meanXp2 = 1/np.sum(intX)*np.sum(xps*intX)
        meanYtot2 = 1/np.sum(intY)*np.sum(hole_y*intY)
        meanYp2 = 1/np.sum(intY)*np.sum(yps*intY)

        exp_x2 =  np.sum(intX * (hole_x - meanXtot2)**2/pixpermm**2)/np.sum((intX))#mm
        exp_xp2 = np.sum(intX * ((np.arctan(sterxs/(mask_to_screen*pixpermm))*1000)**2+(xps - meanXp2)**2))/np.sum((intX))#mrad #no longer std
        exp_xxp = ((np.sum(intX*hole_x*xps)-(np.sum(intX)*meanXp2*meanXtot2))/(pixpermm))/np.sum((intX))#mmmrad
        emitX = np.sqrt(exp_x2 * exp_xp2 - exp_xxp**2)/np.pi

        alphX = -exp_xxp / emitX
        betX = exp_x2 / emitX
        gamX = exp_xp2 / emitX 
        # print(f"alpha = {alphX}")
        # print(f"beta = {betX}")
        ellipsexs = np.arange((min(xs)-x_offset)/pixpermm-2, (max(xs)-x_offset)/pixpermm+2, 0.05)
        ellipse1x = (np.sqrt(betX * emitX * np.pi - ellipsexs**2)-alphX*ellipsexs)/betX
        ellipse2x = (-np.sqrt(betX * emitX * np.pi - ellipsexs**2)-alphX*ellipsexs)/betX
        emitXerr = PeakByPeakFits.EmittanceUncertaintyFunc(intX, sterxs, hole_x, xps, xs, xperr, stdx, meanXtot2, meanXp2, exp_x2, exp_xp2, exp_xxp, mask_to_screen, sigL, mu4xs, emitX, pixpermm, x_offset, hole_err, puncert)
        exp_y2 =  np.sum(intY * (hole_y - meanYtot2)**2/pixpermm**2)/np.sum((intY))#mm
        exp_yp2 = np.sum((intY *(np.arctan(sterys/(mask_to_screen*pixpermm))*1000)**2+intY*(yps - meanYp2)**2))/np.sum((intY))#mrad
        exp_yyp = ((np.sum(intY*hole_y*yps)-(np.sum(intY)*meanYp2*meanYtot2))/(pixpermm))/np.sum((intY))#mmmrad
        emitY = np.sqrt(exp_y2 * exp_yp2 - exp_yyp**2)/np.pi
        alphY = -exp_yyp / emitY
        betY = exp_y2 / emitY
        gamY = exp_yp2 / emitY
        ellipseys = np.arange((min(ys)-y_offset)/pixpermm-2, (max(ys)-y_offset)/pixpermm+2, 0.05)
        ellipse1y = (np.sqrt(betY * emitY * np.pi - ellipseys**2)-alphY*ellipseys)/betY
        ellipse2y = (-np.sqrt(betY * emitY * np.pi - ellipseys**2)-alphY*ellipseys)/betY
        emitYerr = PeakByPeakFits.EmittanceUncertaintyFunc(intY, sterys, hole_y, yps, ys, yperr, stdy, meanYtot2, meanYp2, exp_y2, exp_yp2, exp_yyp, mask_to_screen, sigL, mu4ys, emitY,pixpermm, y_offset, hole_err, puncert)
        ImageData.ImageReader.plot2.clear()
        ImageData.ImageReader.plot3.clear()
        ImageData.ImageReader.plot2.addLine(x=None, y=0, pen=pg.mkPen('k', width=1))
        ImageData.ImageReader.plot2.addLine(x=0, y=None, pen=pg.mkPen('k', width=1))
        ImageData.ImageReader.plot3.addLine(x=None, y=0, pen=pg.mkPen('k', width=1))
        ImageData.ImageReader.plot3.addLine(x=0, y=None, pen=pg.mkPen('k', width=1))
        # plt.figure(figsize=(9,6))
        # plt.errorbar((xs-x_offset)/pixpermm,xps, xerr = sterxs/pixpermm, yerr = xperr, fmt = 'o',label = f'Horizontal Phase Space: $\epsilon_x$ = {emitX:.3f} +/- {emitXerr:.3f} $\pi$*mm*mrad', capsize = 3, markeredgewidth=1)
        # plt.errorbar((ys-y_offset)/pixpermm,yps, xerr = sterys/pixpermm, yerr = yperr,fmt ='.',label = f'Vertical Phase Space: $\epsilon_y$ = {emitY:.3f} +/- {emitYerr:.3f} $\pi$*mm*mrad', capsize = 3, markeredgewidth=1)
        # plt.plot(ellipsexs,ellipse1x, c='tab:blue')
        # plt.plot(ellipsexs,ellipse2x, c='tab:blue')
        # plt.plot(ellipseys,ellipse1y, c='tab:orange')
        # plt.plot(ellipseys,ellipse2y, c='tab:orange')
        # plt.xlabel('X (mm)')
        # plt.ylabel("X' (mrad)")
        # plt.axhline(0,c='k')
        # plt.axvline(0,c='k')
        # plt.legend()
        # plt.show()
        # plto = pg.PlotItem()
        xvalsItem = pg.ScatterPlotItem(x=((xs-x_offset)-(meanXtot2-x_offset))/pixpermm, y=xps, pen=pg.mkPen("#1f77b4", width=1),brush = pg.mkBrush("#1f77b4"),name = f'Horizontal Phase Space: $\epsilon_x$ = {emitX:.3f} +/- {emitXerr:.3f} $\pi$*mm*mrad')
        yvalsItem = pg.ScatterPlotItem(x=((ys-y_offset)-(meanYtot2-y_offset))/pixpermm, y=yps, pen=pg.mkPen("#ff7f0e", width=1),brush = pg.mkBrush("#ff7f0e"),name=f'Vertical Phase Space: $\epsilon_y$ = {emitY:.3f} +/- {emitYerr:.3f} $\pi$*mm*mrad')
        xerrsItem = pg.ErrorBarItem(x=((xs-x_offset)-(meanXtot2-x_offset))/pixpermm, y=xps, height =2* xperr, width=2*sterxs/pixpermm, beam = 0.1,pen = pg.mkPen("#1f77b4", width=1))
        yerrsItem = pg.ErrorBarItem(x=((ys-y_offset)-(meanYtot2-y_offset))/pixpermm, y=yps, height =2* yperr,width = 2*sterys/pixpermm, beam = 0.1, pen = pg.mkPen("#ff7f0e", width=1))
        ImageData.ImageReader.plot2.plot(ellipsexs,ellipse1x, pen=pg.mkPen("#1f77b4", width=5), alpha = 0.8)
        ImageData.ImageReader.plot2.plot(ellipsexs,ellipse2x, pen=pg.mkPen("#1f77b4", width=5), alpha = 0.8)
        ImageData.ImageReader.plot2.plot(ellipseys,ellipse1y, pen=pg.mkPen("#ff7f0e", width=5), alpha = 0.8)
        ImageData.ImageReader.plot2.plot(ellipseys,ellipse2y, pen=pg.mkPen("#ff7f0e", width=5), alpha = 0.8)
        ImageData.ImageReader.plot2.addItem(xerrsItem)
        ImageData.ImageReader.plot2.addItem(yerrsItem)
        ImageData.ImageReader.plot2.addItem(xvalsItem)
        ImageData.ImageReader.plot2.addItem(yvalsItem)
        # legend = pg.LegendItem(pen = pg.mkPen('k'))
        # legend.setParentItem(plto)
        # legend.addItem(xvalsItem, name = f'Horizontal Phase Space: $\epsilon_x$ = {emitX:.3f} +/- {emitXerr:.3f} $\pi$*mm*mrad')
        # legend.addItem(yvalsItem, name=f'Vertical Phase Space: $\epsilon_y$ = {emitY:.3f} +/- {emitYerr:.3f} $\pi$*mm*mrad')
        # ImageData.ImageReader.plot2.addItem(legend)
        #ImageData.ImageReader.plot2.addLegend(legend)
        xemitstr = f'{emitX:.3f}'
        yemitstr = f'{emitY:.3f}'
        xemiterrstr = f'{emitXerr:.3f}'
        yemiterrstr = f'{emitYerr:.3f}'
        xalphstr = f'{alphX:.3f}'
        xbetastr = f'{betX:.3f}'
        xgammstr = f'{gamX:.3f}'
        yalphstr = f'{alphY:.3f}'
        ybetastr = f'{betY:.3f}'
        ygammstr = f'{gamY:.3f}'
        ResultFields.ResFields.xemit.setText(xemitstr)
        ResultFields.ResFields.yemit.setText(yemitstr)
        ResultFields.ResFields.xemiterr.setText(xemiterrstr)
        ResultFields.ResFields.yemiterr.setText(yemiterrstr)
        ResultFields.ResFields.xalph.setText(xalphstr)
        ResultFields.ResFields.yalph.setText(yalphstr)
        ResultFields.ResFields.xbeta.setText(xbetastr)
        ResultFields.ResFields.ybeta.setText(ybetastr)
        ResultFields.ResFields.xgamm.setText(xgammstr)
        ResultFields.ResFields.ygamm.setText(ygammstr)

        return emitX, emitY, emitXerr, emitYerr

    def EmittanceUncertaintyFunc(intX, sterxs, hole_x, xps, xs, xperr, stdx, meanXtot, meanXp, exp_x2, exp_xp2, exp_xxp, L, sigL, mu4x, eps_x,pixpermm, offset, hole_err, puncert):
        #sig_<x^2>
        meanXtot = meanXtot-offset
        hole_x = hole_x - offset+0.00000000001#odd masks have holes at the center, this is to prevent nans in sigxbar
        xs = xs - offset
        sighij = hole_err#(1.2 Known sig_hij) px
        sigxbar = meanXtot * np.sqrt(1/np.sum(intX)+np.sum((hole_x)**2 * intX**2 * (1/intX + sighij**2/(hole_x)**2)) / np.sum((hole_x) * intX)**2)#pix good (3)
        
        sigNp2 = np.sum(intX) * pixpermm**2 * np.sqrt(1/np.sum(intX)+4*puncert**2/pixpermm**2) #(6)(pix/mm)^2
        sigSig1 = np.sqrt(np.sum((intX * (hole_x - meanXtot)**2)**2 * (1/intX + 4/(hole_x - meanXtot)**2*(sighij**2+meanXtot**2*(1/np.sum(intX)+np.sum(hole_x**2*intX**2*(1/intX+(sighij/hole_x)**2))/np.sum(hole_x*intX)**2)))))#(5)
        sigexp_x2 = exp_x2 * np.sqrt((sigNp2/ (np.sum(intX)*pixpermm**2))**2 + (sigSig1 / (np.sum(intX * (hole_x - meanXtot)**2)))**2)#(7)mm good nan here
        
        #sig_<x'^2> # this looks ok?
        xij0 = xs-hole_x#(10) pix
        sigij0 = np.sqrt(sighij**2+stdx**2 - 2*scipy.stats.pearsonr(xs,hole_x)[0] * sighij * stdx)#(11) pix, hole err mixed with spots stdv
        # print(f'hole-spot correlation: {2*scipy.stats.pearsonr(xs,hole_x)[0]*stdx}')
        hole_errs = np.arange(0,0.1,0.001)
        # print(f'stdx: {stdx}')
        #Find minimum in uncertainty
        # for i in hole_errs:
        #     # temphole_err = i
        #     tempsighij = ImageData.hole_separation * ImageData.pixpermm * np.sqrt((puncert/ImageData.pixpermm)**2+(i/ImageData.hole_separation)**2)
        #     plt.plot(i, np.mean(tempsighij), 'o',c = 'b', label = 'no mixing')
        #     plt.plot(i,np.mean(np.sqrt(tempsighij**2+(2*stdx)**2 - 2*scipy.stats.pearsonr(xs,hole_x)[0] * tempsighij * (2*stdx))), 'o', c='r', label = 'mixing')
        #     # plt.plot(i, stdx)
        # plt.xlabel("Hole Uncertainty (mm)")
        # plt.ylabel("Mean Standard Deviation (pix)")
        # # plt.legend()
        # plt.show()
        sigsig22 = (mu4x - (intX-3)/(intX-1)*stdx**4)/intX#(13) pix^4, pre squared to avoid errors
        sigL2p2 = 2 * L**2 * pixpermm**2 * np.sqrt((puncert/pixpermm)**2+(sigL/L)**2)#(14) pix^2
        sig1stxp2 = 1000**2 * (stdx**2+xij0**2)/(L**2*pixpermm**2)*np.sqrt((sigsig22+(2*xij0*sigij0)**2)/(stdx**2+xij0**2)**2+sigL2p2**2/(L**4*pixpermm**4))#(15) looks good mrad
        # print(f'sig1stxp2: {sig1stxp2}')
        sigxpbari = xps * np.sqrt((sigij0/xij0)**2 + sigL**2/L**2+puncert**2/pixpermm**2)#(17) mrad 
        # sigxpbari = xperr # Mixing term if commented, no mixing term if commented
        # print(f'differences = {sigxpbari - xperr}')#sigxpbari takes into account correlation between hole and spot position
        sigxpbar = meanXp * np.sqrt(1/np.sum(intX) + np.sum(intX**2*xps**2*(1/intX + sigxpbari**2/xps**2))/(np.sum(intX*xps)**2))#(19)mrad
        sig22 = 2 * abs(xps-meanXp)*np.sqrt(sigxpbar**2+sigxpbari**2)#(20)mrad
        sig2ndxp2 = intX * (xps-meanXp)**2 *np.sqrt(1/intX + (sig22/((xps-meanXp)**2))**2)#(21)mrad
        # print(f'sig2ndxp2 = {sig2ndxp2}')
        sigexp_xp2 = exp_xp2 * np.sqrt(1/np.sum(intX)+ (np.sum(sig1stxp2**2+sig2ndxp2**2)/(np.sum(1000**2*(stdx**2+xij0**2)/(L**2*pixpermm**2)+intX*(xps-meanXp)**2))**2))#(22)mrad^2

        #sig_<xx'>^2
        sigxxp = meanXtot * meanXp / pixpermm * np.sqrt(sigxbar**2/meanXtot**2 + sigxpbar**2/ meanXp**2+(puncert/pixpermm)**2)#(23) added conversion to mm mrad
        signxhxp = intX * hole_x * xps * np.sqrt(1/intX + (sighij/hole_x)**2+(sigxpbari/xps)**2)#(25)
        sig2ndxxp = np.sum(intX * hole_x*xps)/ (pixpermm * np.sum(intX)) *np.sqrt(1/np.sum(intX) + (puncert/pixpermm)**2+np.sum(signxhxp**2)/(np.sum(intX*hole_x*xps))**2) #(26)
        print(f'meanXtot: {meanXtot} +/- {sigxbar}')
        print(f'meanXp: {meanXp} +/- {sigxpbar}')
        # sigxhxp = hole_x *xps * np.sqrt(xperr**2/xps**2 + hole_err**2/hole_x**2 + 2 * scipy.stats.pearsonr(xps,hole_x)[0] * hole_err * xperr / (hole_x * xps))
        # sigSig3 = intX *hole_x * xps / pixpermm * np.sqrt(1 / intX + sigxhxp**2  / (hole_x * xps)**2)
        # sigSig3N = np.sum(intX * hole_x * xps/pixpermm)/np.sum(intX) * np.sqrt(1/np.sum(intX) + np.sum(sigSig3)/np.sum(intX * hole_x * xps/pixpermm)**2)
        # sigexp_xxp = np.sqrt(sigxxp**2 + sigSig3N**2)
        sigexp_xxp = np.sqrt(sigxxp**2 + sig2ndxxp**2)#(26)
        sigexp_xxp2 = np.abs(2 * exp_xxp * sigexp_xxp)#(27)
        sig_xxp2s = exp_x2 * exp_xp2 * np.sqrt(sigexp_x2**2/exp_x2**2 + sigexp_xp2**2/exp_xp2**2)
        sig_eps2 = np.sqrt(sig_xxp2s**2+sigexp_xxp2**2)#(29)
        sig_eps = np.abs(1/(2 *eps_x)*sig_eps2)#(30)
        #sig_emm
        print(f'exp_x2:{exp_x2} +/- {sigexp_x2}')
        print(f'exp_xp2: {exp_xp2} +/- {sigexp_xp2}')
        print(f'exp_xxp: {exp_xxp} +/- {sigexp_xxp}')
        print(f'eps: {eps_x} +/- {sig_eps}')
        # print(sigexp_x2)
#         sig_eps = 1/(2*eps_x) *np.sqrt(sig_xxp2s**2 + sigexp_xxp2**2)
        return sig_eps
    def on_SaveData_clicked(self):
        saveFileName = QFileDialog.getSaveFileName(caption = "Save Data",filter ="*.csv")
        file = open(saveFileName[0], 'w')
        writer =csv.writer(file)
        writer.writerow(['Mask Details'])
        writer.writerow(["-----------------------------------------------------------------------------"])
        writer.writerow([f' 1D Hole Number: {ImageData.n_holes}', f' Hole Diameter: {ImageData.hole_diameter} mm',f' Hole Separation: {ImageData.hole_separation} mm',f' Mask-to-Screen Distance: {ImageData.mask_to_screen} mm',f' Pixel-to-Millimeter Calibration: {ImageData.pixpermm}p/mm'])
        writer.writerow(['Fit Settings'])
        writer.writerow(["-----------------------------------------------------------------------------"])
        writer.writerow([f' X Peaks: {ImageData.num_peaks_x}', f' Y Peaks: {ImageData.num_peaks_y}',f' Min X: {ImageFields.ImFields.xminIn.text()}',f' Min Y: {ImageFields.ImFields.yminIn.text()}',f' Max X: {ImageFields.ImFields.xmaxIn.text()}', f' Max Y: {ImageFields.ImFields.ymaxIn.text()}'])
        writer.writerow([f'Fit Results'])
        writer.writerow(["-----------------------------------------------------------------------------"])
        writer.writerow([f'X Emittance = {ResultFields.ResFields.xemit.text()} +/- {ResultFields.ResFields.xemiterr.text()} pi mm mrad', f' alpha_x = {ResultFields.ResFields.xalph.text()}', f' beta_x = {ResultFields.ResFields.xbeta.text()}', f' gamma_x = {ResultFields.ResFields.xgamm.text()}'])
        writer.writerow([f'Y Emittance = {ResultFields.ResFields.yemit.text()} +/- {ResultFields.ResFields.yemiterr.text()} pi mm mrad', f' alpha_y = {ResultFields.ResFields.yalph.text()}', f' beta_y = {ResultFields.ResFields.ybeta.text()}', f' gamma_y = {ResultFields.ResFields.ygamm.text()}'])
        writer.writerow(["-----------------------------------------------------------------------------"])
        writer.writerow([f'Fit Data X'])
        writer.writerow(["-----------------------------------------------------------------------------"])
        PeakByPeakFits.resultsdfx.to_csv(file,mode='a')
        writer.writerow([f'Fit Data Y'])
        writer.writerow(["-----------------------------------------------------------------------------"])        
        PeakByPeakFits.resultsdfy.to_csv(file,mode='a')
        file.close()
        return
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
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
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
from numba import jit
#TODO TEMPORARY
from datetime import datetime

path = os.getcwd()
# path = 'D:/Workspace/Images/'
# Define constants
amu2eV = 9.3149410242e8#eV
mass = 1.67262192e-27  # kg
kg2eV = 5.6095886e35 #ev/c^2
c = 299792458*1e3 #mm/s
charge = 1.6e-19  # C

class SimDatWidget(QMainWindow):
    def __init__(self):
        super(SimDatWidget, self).__init__()
        self.central_widget = QWidget()
        self.layoutV1 = QVBoxLayout() # data Params column
        self.layoutH1 = QHBoxLayout() # Particle Info title
        self.layoutH4 = QHBoxLayout() # particle number
        self.layoutH5 = QHBoxLayout() # mass number
        self.layoutH6 = QHBoxLayout() # kinetic energy
        self.layoutH7 = QHBoxLayout() # beam radius
        self.layoutH8 = QHBoxLayout() # beam divergence
        self.layoutH12 = QHBoxLayout() # Alignment offset
        self.layoutH9 = QHBoxLayout() # generate/save
        self.layoutH10 = QHBoxLayout()# load 
        self.layoutH11 = QHBoxLayout()# Emittance Report 
        self.setCentralWidget(self.central_widget)

        SimDatWidget.simNumPartIn = num_part_read()
        SimDatWidget.simMassNumIn = mass_num_read()
        SimDatWidget.simKinEIn = kin_en_read()
        SimDatWidget.simKinEstdvIn = kin_en_stdv_read()
        SimDatWidget.simRadIn = simRad_read()
        SimDatWidget.simDivIn = simDiv_read()
        SimDatWidget.simxAlign = simXalign_read()
        SimDatWidget.simyAlign = simYalign_read()
        SimDatWidget.xemit = QLabel('')#x_peak_read()
        SimDatWidget.yemit = QLabel('')#y_peak_read()

        loadDataPrompt = QPushButton('Load Source Data')
        genDataPrompt = QPushButton('Generate Source Data')
        saveDataPrompt = QPushButton('Save Source Data')
        loadDataPrompt.clicked.connect(self.on_LoadDat_clicked)
        genDataPrompt.clicked.connect(self.on_GenDat_clicked)
        saveDataPrompt.clicked.connect(self.on_SaveDat_clicked)

        
        # saveImagePrompt = QPushButton('*Save Image*')
        # loadImagePrompt.clicked.connect(ImageData.ImageReader.on_LoadIm_clicked)
        # saveImagePrompt.clicked.connect(ImageData.ImageReader.on_SaveIm_clicked)

        self.central_widget.setLayout(self.layoutV1)
        self.layoutV1.addLayout(self.layoutH1)
        self.pinfo = QLabel("<b>Particle Info<\b>")
        self.pinfo.setTextFormat(Qt.RichText)
        self.pinfo.setAlignment(Qt.AlignHCenter)
        self.layoutH1.addWidget(self.pinfo)
        # self.layoutH1.addWidget(loadImagePrompt)
        self.layoutV1.addLayout(self.layoutH4)
        self.layoutH4.addWidget(QLabel('Number of Particles'))
        self.layoutH4.addWidget(SimDatWidget.simNumPartIn)

        self.layoutV1.addLayout(self.layoutH5)
        self.layoutH5.addWidget(QLabel('Particle Mass Number (amu)'))
        self.layoutH5.addWidget(SimDatWidget.simMassNumIn)

        self.layoutV1.addLayout(self.layoutH6)
        self.layoutH6.addWidget(QLabel('Average Kinetic Energy(eV)'))
        self.layoutH6.addWidget(SimDatWidget.simKinEIn)
        self.layoutH6.addWidget(QLabel('+/-'))
        self.layoutH6.addWidget(SimDatWidget.simKinEstdvIn)

        self.layoutV1.addLayout(self.layoutH7)
        self.layoutH7.addWidget(QLabel('Beam Radius (mm)'))
        self.layoutH7.addWidget(SimDatWidget.simRadIn )

        self.layoutV1.addLayout(self.layoutH8)
        self.layoutH8.addWidget(QLabel('Beam Divergence (degrees)'))
        self.layoutH8.addWidget(SimDatWidget.simDivIn ) 

        self.layoutV1.addLayout(self.layoutH12)
        self.layoutH12.addWidget(QLabel('X Offset (mm)'))
        self.layoutH12.addWidget(SimDatWidget.simxAlign ) 
        self.layoutH12.addWidget(QLabel('Y Offset (mm)'))
        self.layoutH12.addWidget(SimDatWidget.simyAlign ) 

        self.layoutV1.addLayout(self.layoutH9)
        self.layoutH9.addWidget(genDataPrompt)
        self.layoutH9.addWidget(saveDataPrompt)
        self.layoutV1.addLayout(self.layoutH10)
        self.layoutH10.addWidget(loadDataPrompt)
        self.layoutV1.addLayout(self.layoutH11)
        self.layoutH11.addWidget(QLabel("X Emittance = "))
        self.layoutH11.addWidget(SimDatWidget.xemit)
        self.layoutH11.addWidget(QLabel("Y Emittance = "))
        self.layoutH11.addWidget(SimDatWidget.yemit)
        # self.layoutH9.addWidget(self.Calibration) 
        # self.layoutH9.addWidget(QLabel('+/-'))
        # self.layoutH9.addWidget(self.puncert) 

    def on_GenDat_clicked(self):
        now = datetime.now()
        print("GENERATE ", now.strftime("%H:%M:%S"))
        try:
            x_align = float(SimDatWidget.simxAlign.text())
        except:
            x_align = 0
        try:
            y_align =float(SimDatWidget.simyAlign.text())
        except:
            y_align = 0
        try:
            num_particles = abs(int(float(SimDatWidget.simNumPartIn.text())))
            mass_num = abs(float(SimDatWidget.simMassNumIn.text()))
            kinetic_energy =abs(float(SimDatWidget.simKinEIn.text()))
            beamRad =abs(float(SimDatWidget.simRadIn.text()))
            theta2_max =abs(float(SimDatWidget.simDivIn.text()))
        except:
            msgBox = QMessageBox()
            msgBox.setWindowIcon(QIcon("mrsPepper.png"))
            msgBox.setWindowTitle('Particle Data Error')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText('Key Particle Information Failed to load')
            msgBox.exec_()
        try:
            kinetic_energy_stdv =abs(float(SimDatWidget.simKinEstdvIn.text()))
        except:
            kinetic_energy_stdv = 0
        # current_time = now.strftime("%H:%M:%S")

        theta1 = np.random.normal(0, 2*np.pi, num_particles)#position
        r = np.sqrt(np.absolute(np.random.normal(0,beamRad**2/2, num_particles)))#position mm
        now = datetime.now()
        print("5% ", now.strftime("%H:%M:%S"))
        x1 = (r * np.cos(theta1) + x_align)
        y1 = (r * np.sin(theta1) + y_align)
        theta2 = np.arccos(1-2*np.random.uniform(0, 0.5*(1-np.cos(theta2_max*np.pi/180)), num_particles))#np.random.normal(0,6.5e-5, num_particles)
        now = datetime.now()
        print("15% ", now.strftime("%H:%M:%S"))
        phi = 2*np.pi*np.random.uniform(size=num_particles)#np.arccos(np.random.uniform(-1,1,num_particles))
        pos_z = np.random.uniform(-0.5,0.5,num_particles)#mm
        E = np.random.normal(kinetic_energy, kinetic_energy_stdv, num_particles)
        v = np.sqrt(2 * E / (mass_num * amu2eV))*c#mm/s
        now = datetime.now()
        print("20% ", now.strftime("%H:%M:%S"))
        sign = np.random.uniform(size=1)*2-1
        xp = (np.arctan(np.cos(phi)*np.tan(theta2)))*1000+x1*sign*np.random.normal(50*theta2_max,theta2_max*1,num_particles)#mrad np.sin(theta2)*
        yp = (np.arctan(np.sin(phi)*np.tan(theta2)))*1000+y1*sign*np.random.normal(50*theta2_max,theta2_max*1,num_particles)#mrad 
        zp = np.cos(theta2)#mm/s?
        now = datetime.now()
        print("40% ", now.strftime("%H:%M:%S"))
        vx = v * np.tan(xp /1000)*zp
        vy = v * np.tan(yp /1000)*zp
        #40%
        vz = v * zp 
        SimDatWidget.tempdf = pd.DataFrame({'X':x1,'Y':y1,'Z':pos_z,'Vx':vx,'Vy':vy,'Vz':vz, 'Xp':xp,'Yp': yp})
        # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        now = datetime.now()
        print("45% ", now.strftime("%H:%M:%S"))
        SimDatWidget.xxp = np.histogram2d(SimDatWidget.tempdf.X, SimDatWidget.tempdf.Xp,bins = int(np.min([SimDatWidget.tempdf.shape[0]/100,500])))
        now = datetime.now()
        print("60% ", now.strftime("%H:%M:%S"))
        SimDatWidget.yyp = np.histogram2d(SimDatWidget.tempdf.Y, SimDatWidget.tempdf.Yp,bins =  int(np.min([SimDatWidget.tempdf.shape[0]/100,500])))
        now = datetime.now()
        print("70% ", now.strftime("%H:%M:%S"))
        SimDatWidget.xy = np.histogram2d(SimDatWidget.tempdf.X, SimDatWidget.tempdf.Y,bins =  int(np.min([SimDatWidget.tempdf.shape[0]/100,500])))
        now = datetime.now()
        print("85% ", now.strftime("%H:%M:%S"))
        SimDatWidget.xpyp = np.histogram2d(SimDatWidget.tempdf.Xp, SimDatWidget.tempdf.Yp,bins =  int(np.min([SimDatWidget.tempdf.shape[0]/100,500])))

        xxp_img = pg.ImageItem(SimDatWidget.xxp[0], invertY=False)#
        xxp_img.setRect([min(SimDatWidget.xxp[1]),min(SimDatWidget.xxp[2]),(max(SimDatWidget.xxp[1])-min(SimDatWidget.xxp[1])),(max(SimDatWidget.xxp[2])-min(SimDatWidget.xxp[2]))])
        xxp_img.setColorMap('viridis')#

        yyp_img = pg.ImageItem(SimDatWidget.yyp[0], invertY=False)
        yyp_img.setRect([min(SimDatWidget.yyp[1]),min(SimDatWidget.yyp[2]),(max(SimDatWidget.yyp[1])-min(SimDatWidget.yyp[1])),(max(SimDatWidget.yyp[2])-min(SimDatWidget.yyp[2]))])
        yyp_img.setColorMap('viridis')

        xy_img = pg.ImageItem(SimDatWidget.xy[0], invertY=False)
        xy_img.setRect([min(SimDatWidget.xy[1]),min(SimDatWidget.xy[2]),(max(SimDatWidget.xy[1])-min(SimDatWidget.xy[1])),(max(SimDatWidget.xy[2])-min(SimDatWidget.xy[2]))])
        xy_img.setColorMap('viridis')

        xpyp_img = pg.ImageItem(SimDatWidget.xpyp[0], invertY=False)
        xpyp_img.setRect([min(SimDatWidget.xpyp[1]),min(SimDatWidget.xpyp[2]),(max(SimDatWidget.xpyp[1])-min(SimDatWidget.xpyp[1])),(max(SimDatWidget.xpyp[2])-min(SimDatWidget.xpyp[2]))])
        xpyp_img.setColorMap('viridis')

        SimagesWidget.ax_xxp.clear()
        SimagesWidget.ax_xxp.addItem(xxp_img)

        SimagesWidget.ax_xy.clear()
        SimagesWidget.ax_xy.addItem(xy_img)

        SimagesWidget.ax_yyp.clear()
        SimagesWidget.ax_yyp.addItem(yyp_img)

        SimagesWidget.ax_xpyp.clear()
        SimagesWidget.ax_xpyp.addItem(xpyp_img)

        self.show()
        now = datetime.now()
        print("95% ", now.strftime("%H:%M:%S"))
        SimDatWidget.xemit.setText(f'{np.sqrt(np.mean(SimDatWidget.tempdf.X**2)*np.mean(SimDatWidget.tempdf.Xp**2)-np.mean(SimDatWidget.tempdf.X*SimDatWidget.tempdf.Xp)**2)/np.pi:.3f}')
        SimDatWidget.yemit.setText(f'{np.sqrt(np.mean(SimDatWidget.tempdf.Y**2)*np.mean(SimDatWidget.tempdf.Yp**2)-np.mean(SimDatWidget.tempdf.Y*SimDatWidget.tempdf.Yp)**2)/np.pi:.3f}')
        # del tempdf
        gc.collect()
        now = datetime.now()
        print("100% ", now.strftime("%H:%M:%S"))

    def on_LoadDat_clicked(self):
        loadDatName = QFileDialog.getOpenFileName(caption="Load Particle Data", directory=path, filter="*.pkl")
        SimDatWidget.tempdf = pd.read_pickle(loadDatName[0])
        now = datetime.now()
        print("5% ", now.strftime("%H:%M:%S"))
        SimDatWidget.xxp = np.histogram2d(SimDatWidget.tempdf.X, SimDatWidget.tempdf.Xp,bins = int(np.min([SimDatWidget.tempdf.shape[0]/100,500])))
        now = datetime.now()
        print("30% ", now.strftime("%H:%M:%S"))
        SimDatWidget.yyp = np.histogram2d(SimDatWidget.tempdf.Y, SimDatWidget.tempdf.Yp,bins =  int(np.min([SimDatWidget.tempdf.shape[0]/100,500])))
        now = datetime.now()
        print("65% ", now.strftime("%H:%M:%S"))
        SimDatWidget.xy = np.histogram2d(SimDatWidget.tempdf.X, SimDatWidget.tempdf.Y,bins =  int(np.min([SimDatWidget.tempdf.shape[0]/100,500])))
        now = datetime.now()
        print("95% ", now.strftime("%H:%M:%S"))
        SimDatWidget.xpyp = np.histogram2d(SimDatWidget.tempdf.Xp, SimDatWidget.tempdf.Yp,bins =  int(np.min([SimDatWidget.tempdf.shape[0]/100,500])))

        xxp_img = pg.ImageItem(SimDatWidget.xxp[0], invertY=False)#
        xxp_img.setRect([min(SimDatWidget.xxp[1]),min(SimDatWidget.xxp[2]),(max(SimDatWidget.xxp[1])-min(SimDatWidget.xxp[1])),(max(SimDatWidget.xxp[2])-min(SimDatWidget.xxp[2]))])
        xxp_img.setColorMap('viridis')#

        yyp_img = pg.ImageItem(SimDatWidget.yyp[0], invertY=False)
        yyp_img.setRect([min(SimDatWidget.yyp[1]),min(SimDatWidget.yyp[2]),(max(SimDatWidget.yyp[1])-min(SimDatWidget.yyp[1])),(max(SimDatWidget.yyp[2])-min(SimDatWidget.yyp[2]))])
        yyp_img.setColorMap('viridis')

        xy_img = pg.ImageItem(SimDatWidget.xy[0], invertY=False)
        xy_img.setRect([min(SimDatWidget.xy[1]),min(SimDatWidget.xy[2]),(max(SimDatWidget.xy[1])-min(SimDatWidget.xy[1])),(max(SimDatWidget.xy[2])-min(SimDatWidget.xy[2]))])
        xy_img.setColorMap('viridis')

        xpyp_img = pg.ImageItem(SimDatWidget.xpyp[0], invertY=False)
        xpyp_img.setRect([min(SimDatWidget.xpyp[1]),min(SimDatWidget.xpyp[2]),(max(SimDatWidget.xpyp[1])-min(SimDatWidget.xpyp[1])),(max(SimDatWidget.xpyp[2])-min(SimDatWidget.xpyp[2]))])
        xpyp_img.setColorMap('viridis')

        SimagesWidget.ax_xxp.clear()
        SimagesWidget.ax_xxp.addItem(xxp_img)

        SimagesWidget.ax_xy.clear()
        SimagesWidget.ax_xy.addItem(xy_img)

        SimagesWidget.ax_yyp.clear()
        SimagesWidget.ax_yyp.addItem(yyp_img)

        SimagesWidget.ax_xpyp.clear()
        SimagesWidget.ax_xpyp.addItem(xpyp_img)
        now = datetime.now()
        print("95% ", now.strftime("%H:%M:%S"))
        SimDatWidget.xemit.setText(f'{np.sqrt(np.mean(SimDatWidget.tempdf.X**2)*np.mean(SimDatWidget.tempdf.Xp**2)-np.mean(SimDatWidget.tempdf.X*SimDatWidget.tempdf.Xp)**2)/np.pi:.3f}')
        SimDatWidget.yemit.setText(f'{np.sqrt(np.mean(SimDatWidget.tempdf.Y**2)*np.mean(SimDatWidget.tempdf.Yp**2)-np.mean(SimDatWidget.tempdf.Y*SimDatWidget.tempdf.Yp)**2)/np.pi:.3f}')
    def on_SaveDat_clicked(self):
        saveDatName = QFileDialog.getSaveFileName(caption="Save Mask", filter="*.pkl")
        # print(saveDatName)
        SimDatWidget.tempdf.to_pickle(saveDatName[0])
    def on_CalcTraj_clicked(self):
        print("Number Crunching")
        hole_diameter = float(Mainapp.MainWindow.MskFields2.diamIn.text())#127e-3 #mm
        hole_separation = float(Mainapp.MainWindow.MskFields2.sepIn.text())#0.50 #mm
        mask_to_screen = float(Mainapp.MainWindow.MskFields2.Mask2ScrnIn.text())#6.35#12.7 #mm
        pixpermm = float(Mainapp.MainWindow.MskFields2.Calibration.text())#15#12.5 um pores on MCP
        n_holes = int(Mainapp.MainWindow.MskFields2.numHoles.text()) #squared
        box_start = 0 #mm
        try:
            box_end = float(Mainapp.MainWindow.maskWidth.maskwidth.text()) #mm
        except:
            print("Invalid mask width, defaulting to 0.1 mm")
            box_end = 0.1
        target_z = box_end + mask_to_screen
        d = (hole_separation)*pixpermm+hole_diameter*pixpermm
        #d = (hole_separation)/2*pixpermm+hole_diameter*pixpermm # this is the old line of code and I don't know why I defined this off half the hole separation
        locs2 = []
        for i in range(n_holes):
            for j in range(n_holes):
                locs2.append([i*d-d*(n_holes-1)/2,j*d-(d*(n_holes-1))/2])#pix
        locs2 = np.array(locs2).T
        @jit(nopython=True)
        def motion(xi,yi,zi,vxi,vyi,vzi,t):
                zf = []
                xf = []
                yf = []
                for j in range(zi.shape[0]):#iterate over particle number
                    #print(j)
                    xt = xi[j]
                    yt = yi[j]
                    zt = zi[j]
                    passed = False
                    for i in range(t.shape[0]):#walk through time
                    #print('Still kicking')
                        zt = zi[j] + vzi[j]*t[i]
                        #print(zt)
                        if ((zt >= box_start) and (zt <= box_end)):#check that we are within the mask
                            xt = xi[j] + vxi[j]*t[i]
                            yt = yi[j] + vyi[j]*t[i]
#                             print(f'Hey Charlie! {xt}')
                            for loc in range(locs2.shape[1]):#/pixpermm:#mm
                                x = locs2[0][loc]/pixpermm
                                y = locs2[1][loc]/pixpermm
#                                 for x in (locs2[0])/pixpermm:
                                if ((xt - x) * (xt - x) +(yt - y) * (yt - y) <= (hole_diameter/2 * hole_diameter/2)): #check that we pass the mask
                                    passed = True
#                                         print(f'yay Charlie{xt}')
                                    break #break out of loc loop
                                else:
#                                   print(f'ono Charlie! {xt}')
                                    continue
                        elif(zt>box_end):
                            break
                        else:
                            continue           
                        break#break out of time
                            #is xt,yt in x_box,y_box?
                    if passed == True:
                        for i in range(t.shape[0]):
                            zt = zi[j] + vzi[j]*t[i]
                            if zt >= target_z:
                                xt = xi[j] + vxi[j]*t[i]
                                yt = yi[j] + vyi[j]*t[i]
                                zf.append(zt)
                                xf.append(xt)
                                yf.append(yt)
                                break
                    elif passed == False:
                        xt = 9999
                        yt = 9999
                        zt = 9999
                        zf.append(zt)
                        xf.append(xt)
                        yf.append(yt)
                    continue
                zf = np.array(zf)
                xf = np.array(xf)
                yf = np.array(yf)
                return xf,yf,zf
        time = 1e-6
        dt = 1e-10
        times = np.arange(0,time,dt)
        num_pix = pixpermm*25.4*1.5
        now = datetime.now()
        print("PreMotion % ", now.strftime("%H:%M:%S"))
        fin_x,fin_y, fin_z = motion(SimDatWidget.tempdf.X.to_numpy(),SimDatWidget.tempdf.Y.to_numpy(),SimDatWidget.tempdf.Z.to_numpy(),SimDatWidget.tempdf.Vx.to_numpy(),SimDatWidget.tempdf.Vy.to_numpy(),SimDatWidget.tempdf.Vz.to_numpy(),times[:-1])
        now = datetime.now()
        print("PostMotion % ", now.strftime("%H:%M:%S"))
        offset = num_pix/2
        fin_x = fin_x[fin_y != 9999] * pixpermm +offset#px
        fin_y = fin_y[fin_y != 9999] * pixpermm +offset#px
        pixels = np.arange(0,num_pix+1, 1)
        SimDatWidget.hist, xedges, yedges = np.histogram2d(fin_x, fin_y, bins=[pixels,pixels])
 #add noise and generate image
        try:
            noise1 = int(MaskFields.MaskSimDat.noiseLevel.text()) #squared
        except:
            print("Issue reading in Noise Level, defaulting to 0")
            noise1 = 0
        try:
            noise2 = int(MaskFields.MaskSimDat.noiseUncert.text()) #squared
        except:
            print("Issue reading in Noise Uncertainty, defaulting to 0")
            noise2 = 0
        print(np.max(SimDatWidget.hist))
        print(SimDatWidget.hist)
        m = (255-noise1)/np.max(SimDatWidget.hist)
        noise = np.random.normal(noise1, noise2, (SimDatWidget.hist.shape[0],SimDatWidget.hist.shape[0]))
        print(noise)
        #print(noise.shape[0])
        print(np.max(SimDatWidget.hist))
        SimDatWidget.newhist = SimDatWidget.hist*m + noise
        plt.figure(figsize = (0.98,0.98), dpi =  25.4 * pixpermm)#(2.0455,2.0455)
        print( SimDatWidget.newhist.shape[0])
        print(SimDatWidget.newhist)
        print(np.max(SimDatWidget.newhist[0]))
        plt.imshow( SimDatWidget.newhist, cmap = plt.cm.gray)
        #plt.colorbar()
        plt.axis('off')
        #filenum2 = filenum # toggle for new emit data
    #     plt.savefig(imgAlbum+folder+f'simupepper-{filenum2}-h.png', bbox_inches='tight', pad_inches = 0)
        plt.show()
    def on_SaveTraj_clicked():
            saveTrajName = QFileDialog.getSaveFileName(caption="Save Trajectories", filter="*.csv")
            #TODO add ability to save to PNG
            np.savetxt(saveTrajName[0], SimDatWidget.newhist.T, delimiter=",")
class num_part_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
        self.setPlaceholderText("#")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

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

class mass_num_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        self.setMaxLength(6)
        self.setPlaceholderText("amu")
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
class kin_en_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        # self.setMaxLength(6)
        self.setPlaceholderText("eV")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

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

class simRad_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        # self.setMaxLength(6)
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

class simDiv_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        # self.setMaxLength(2)
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
        # print("num changed...")
        # print(s)
        return
    
    def text_edited(self, s):
        # print("num edited...")
        # print(s)
        return

class simXalign_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        # self.setMaxLength(8)
        self.setPlaceholderText("0")
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

class simYalign_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        # self.setMaxLength(8)
        self.setPlaceholderText("0")
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
    
class kin_en_stdv_read(QLineEdit):
    def __init__(self):
        QLineEdit.__init__(self)
        # self.setMaxLength(6)
        self.setPlaceholderText("eV")
        self.returnPressed.connect(self.return_pressed)
        self.selectionChanged.connect(self.selection_changed)
        self.textChanged.connect(self.text_changed)
        self.textEdited.connect(self.text_edited)

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
class SimagesWidget(QMainWindow):
    def __init__(self):
        super(SimagesWidget, self).__init__()
        self.central_widget = QWidget()
        self.layoutV1 = QVBoxLayout() # Central Layout
        self.layoutV2 = QVBoxLayout() # X-X'
        self.layoutV3 = QVBoxLayout() # X-Y
        self.layoutV4 = QVBoxLayout() # Y-Y'
        self.layoutV5 = QVBoxLayout() # X'-Y'
        self.layoutH1 = QHBoxLayout() # row 1
        self.layoutH2 = QHBoxLayout() # row two
        self.setCentralWidget(self.central_widget)

        SimagesWidget.ax_xxp = pg.PlotWidget(plotItem=pg.PlotItem())
        SimagesWidget.ax_xy =   pg.PlotWidget(plotItem=pg.PlotItem())
        SimagesWidget.ax_yyp =  pg.PlotWidget(plotItem=pg.PlotItem())
        SimagesWidget.ax_xpyp = pg.PlotWidget(plotItem=pg.PlotItem())
        xxplabel = QLabel("X-X'")
        yyplabel = QLabel("Y-Y'")
        xylabel  = QLabel("X-Y")
        xpyplabel = QLabel("X'-Y'")
        xxplabel.setAlignment(Qt.AlignCenter)
        yyplabel.setAlignment(Qt.AlignCenter)
        xylabel.setAlignment(Qt.AlignCenter)
        xpyplabel.setAlignment(Qt.AlignCenter)
        self.central_widget.setLayout(self.layoutV1)
        self.layoutV1.addLayout(self.layoutH1)
        self.layoutH1.addLayout(self.layoutV2)
        self.layoutV2.addWidget(xxplabel)
        self.layoutV2.addWidget(SimagesWidget.ax_xxp)
    
        self.layoutH1.addLayout(self.layoutV3)
        self.layoutV3.addWidget(xylabel)
        self.layoutV3.addWidget(SimagesWidget.ax_xy)

        self.layoutV1.addLayout(self.layoutH2)
        self.layoutH2.addLayout(self.layoutV4)
        self.layoutV4.addWidget(yyplabel)
        self.layoutV4.addWidget(SimagesWidget.ax_yyp)

        self.layoutH2.addLayout(self.layoutV5)
        self.layoutV5.addWidget(xpyplabel)
        self.layoutV5.addWidget(SimagesWidget.ax_xpyp)
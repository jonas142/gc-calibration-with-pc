from cProfile import label
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from os.path import exists
from math import sqrt
from datetime import datetime

class GCTracePlotter:
    """Used for plotting the activity Trace of the Grid Cells"""
    def __init__(self, env_model, from_data):
        plt.ion()

        if env_model == "large_box":
            self.size = 7.5
        elif env_model == "T-shape":
            self.size = 10
        else:
            self.size = 3
        print("size: {}".format(self.size))
        self.directory = "data/plots/"

        self.x, self.y, self.applied_color = [], [], []
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        plt.xlim(-self.size,self.size)
        plt.ylim(-self.size,self.size)

        if from_data and exists(self.directory + "plotting_xdata_gc.npy"):
            self.x = np.load(self.directory + "plotting_xdata_gc.npy").tolist()
            self.y = np.load(self.directory + "plotting_ydata_gc.npy").tolist()
            self.applied_color =  np.load(self.directory + "plotting_colordata_gc.npy").tolist()
            print(len(self.x))
            self.ax.scatter(self.x, self.y,s=0.5, c=self.applied_color)
        # plt.show()

    def pause(self, interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

    def drawMotion(self, xy_coordinate, gc_spiking):
        self.x.append(xy_coordinate[0])
        self.y.append(xy_coordinate[1])
        color = 'gray'
        if gc_spiking[0] > 0.15:
            color = 'red'
        self.applied_color.append(color)
        self.ax.scatter(xy_coordinate[0], xy_coordinate[1], s = 1,color = color)
#        self.pause(0.01) # enable for real time plot

    def save_plot(self):

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        filename = self.directory + "gridCells_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".png"

        print("Saving figure...")
        self.fig.savefig(filename)
        print("Saving data...")
        np.save(self.directory + "plotting_xdata_gc.npy", self.x)
        np.save(self.directory + "plotting_ydata_gc.npy", self.y)
        np.save(self.directory + "plotting_colordata_gc.npy", self.applied_color)

        
        self.destroy()

    def compare(self):
        self.x = np.load(self.directory + "t-shape_calibrated/" + "plotting_xdata_gc.npy").tolist()
        self.y = np.load(self.directory + "t-shape_calibrated/" + "plotting_ydata_gc.npy").tolist()
        self.applied_color =  np.load(self.directory + "t-shape_calibrated/" + "plotting_colordata_gc.npy").tolist()
        self.x2 = np.load(self.directory + "t-shape_no_noise/" + "plotting_xdata_gc.npy").tolist()
        self.y2 = np.load(self.directory + "t-shape_no_noise/" + "plotting_ydata_gc.npy").tolist()
        self.applied_color2 =  np.load(self.directory + "t-shape_no_noise/" + "plotting_colordata_gc.npy").tolist()
        for i, color in enumerate(self.applied_color):
            if color == "red":
                self.ax.scatter(self.x[i], self.y[i], s=0.5, c = color)
        for i, color in enumerate(self.applied_color2):
            if color == "red":
                self.ax.scatter(self.x2[i], self.y2[i], s=0.5, c = "blue")
        filename = self.directory + "gridCells_comparison_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".png"
        self.fig.savefig(filename)
        plt.show()


    def destroy(self):
        plt.close()

class GCActivityPlotter:
    """Used for plotting the activity Peaks of the Grid Cells"""
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        plt.xlim(0, 39)
        plt.ylim(0, 39)
        plt.show(block = False)

    def drawPeaks(self, gc_spiking):
        s = np.reshape(gc_spiking, (40, 40))
        plt.imshow(s, origin="lower")
        self.pause(0.1)

    def pause(self, interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

class GCCalibrationPlotter:
    def __init__(self):
        spiking = [1.730880014399245, 1.7300485382153583, 1.729767162137046, 1.7307290363717058, 1.733788065638382, 1.7399744125421188, 1.7505291838150387, 1.766969050379076, 1.7911961380734724, 1.8256685858422916, 1.8736486124445726, 1.9395489088436881, 2.0293912499685014, 2.151393779296806, 2.316689927118408, 2.5401733806537576, 2.8414649334753426, 3.2459859848649466, 3.786141641951321, 4.502591733716102, 5.445523822070006, 6.675671378843316, 8.264515500392516, 10.29268718616338, 12.845106149628563, 16.001143202564037, 19.818564009027384, 24.31190722232516, 29.42960949106716, 35.038416102381206, 40.92526205932043, 46.821968572850835, 52.44692683677535, 57.54702982243178, 61.92096758376875, 65.41371903112, 67.8875525471208, 69.19733476206733, 69.22283292796368, 67.95461436205188, 65.50055593897461, 62.00392344759341, 57.60710060859434, 52.47323410540085, 46.81314578970279, 40.88843496439263, 34.9858808163972, 29.374453261753633, 24.264333310812653, 19.784144595785857, 15.98102606879587, 12.837372637822572, 10.293955132680004, 8.271201320394448, 6.684734265979248, 5.454761424272558, 4.510642833250094, 3.7923406612140145, 3.25017044582276, 2.8437929857287, 2.540975714113109, 2.316362660571591, 2.150329390687653, 2.02793817776243, 1.9379925584701474, 1.8722057739636657, 1.824488431275451, 1.7903663654823592, 1.7665234785508612, 1.7504562522168927, 1.7402271305022323, 1.7342933351409466, 1.7313986686659715, 1.730507565104963, 1.7307694787432508, 1.7315037144460903, 1.7321911995901713, 1.7324700706839056, 1.7321281720285513, 1.7310904413469512, 1.7293996463246581, 1.72719250413211, 1.7246719349687458, 1.7220769553200905, 1.7196548185921265, 1.7176345369581478, 1.7162053662998495, 1.7154999755032554, 1.7155829461932353, 1.716445957153907, 1.7180080996188325, 1.720122763962523, 1.7225895543794816, 1.7251722609118092, 1.727622245733351, 1.7297041537928144, 1.731225228889529, 1.732063981930127, 1.7321976756560018, 1.731724516038577]

        x = list(range(len(spiking)))
        # plt.plot(x, spiking, color='r', label="spiking0")
        plt.plot(x, spiking, color='g', label="inject0")
        # plt.legend()
        filename = "data/plots/" + "calibration_data_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".png"
        # plt.savefig(filename)
        plt.show()

class GCWeigthPlotter:
    def __init__(self):
        i = 322
        weigths = np.load("data/gc_model/pc_weigths.npy")
        acd_active = np.load("data/pc_model/acd_active.npy")
        print(acd_active[i])
        x = list(range(len(weigths[0][0])))
        plt.imshow(np.reshape(weigths[0][i], (40,40)))
        plt.show()
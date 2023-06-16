from datetime import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    """ This class deals with plotting """
    def __init__(self, env_model):
        plt.ion()

        if env_model == "large_box":
            self.size = 7.5
        else:
            self.size = 3

        self.x, self.y = [], []
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.colors = ['gray','red']
        # self.ax.scatter(self.x,self.y, color = self.colors)
        plt.xlim(-self.size,self.size)
        plt.ylim(-self.size,self.size)
        plt.show(block=False)

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

    def computeSpikeLocations(self, spiking_values):
        tmp = []
        for i in range(len(spiking_values)):
            if spiking_values[i] > 0.75: 
                tmp.append(self.colors[1])
            else:
                tmp.append(self.colors[0])
        return tmp

    def appendNewSpikingCoordinate(self, xy_coordinate):
        self.x.append(xy_coordinate[0])
        self.y.append(xy_coordinate[1])

    def animate(self, spiking_values, xy_coordinate = [], createdPlaceCell=False):
        color_array = self.computeSpikeLocations(spiking_values)
        if createdPlaceCell:
            self.appendNewSpikingCoordinate(xy_coordinate)

        self.ax.cla()
        self.ax.scatter(xy_coordinate[0], xy_coordinate[1], s = 2, color = 'blue')

        if len(self.x) > 0:
            self.ax.scatter(self.x, self.y, s = 2, color = color_array)
            for i in range(len(spiking_values)):
                plt.annotate("{:.4f}".format(spiking_values[i]), (self.x[i], self.y[i]))
        plt.xlim(-self.size,self.size)
        plt.ylim(-self.size,self.size)
        self.pause(0.1)

    def save_plot(self):
        directory = "data/plots/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filename = directory + "placeCells_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".png"

        print("Saving figure...")
        plt.savefig(filename)
        
        self.destroy()

    def destroy(self):
        plt.close()

class PCInputPlotter:
    def __init__(self):
        self.size = 7.5
        self.x, self.y = [], []
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.colors = ['gray','red']
        # self.ax.scatter(self.x,self.y, color = self.colors)
        plt.xlim(-self.size,self.size)
        plt.ylim(-self.size,self.size)

        self.env_coordinates = np.load("data/pc_model/env_coordinates.npy")
        self.acd_active = np.load("data/pc_model/acd_active.npy")
        self.x = np.transpose(self.env_coordinates)[0]
        self.y = np.transpose(self.env_coordinates)[1]
        self.colors = np.where(self.acd_active, 'red', 'gray')

        self.ax.scatter(self.x, self.y, c=self.colors)

        filename = "data/plots/" + "acd_distribution_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".png"
        plt.savefig(filename)
        plt.show


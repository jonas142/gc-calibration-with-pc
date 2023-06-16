# from plotting.plotThesis import *
import os

import numpy as np

from .parametersFdbckLoop import training
from .helper import *

class PlaceCell:
    """Class to keep track of an individual Place Cell"""
    def __init__(self, gc_connections, bvc_connections, acd_connections, acd_active):
        self.gc_connections = gc_connections  # Connection matrix to grid cells of all modules; has form (n^2 x M)
        
        bvc_nan = np.isnan(bvc_connections)
        if (True in bvc_nan) or (np.sum(bvc_connections) == 0):
            self.bvc_active = False
        else:
            self.bvc_active = True
        
        self.bvc_connections = bvc_connections # Connection array to all bvc cells
        if self.bvc_active:
            self.bvc_excitatory_inhibitory  = np.where(bvc_connections > 0.3, 1, -1) # defines whether connections are excitatory or inhibitory 
            self.max_bvc_spike_potential = np.sum(np.multiply(self.bvc_excitatory_inhibitory, bvc_connections))

        self.acd_active = acd_active
        self.acd_connections = np.array(acd_connections) # connection array to all acd cells
        # print(self.acd_connections)

        if self.acd_active and self.bvc_active:
            self.acd2pcweights = np.where(self.acd_connections > 68, 1, 0)
            # print(self.acd2pcweights)
            self.max_acd_spike_potential = np.sum(np.multiply(self.acd2pcweights, self.acd_connections))
            if self.max_acd_spike_potential == 0:
                self.acd_active = False
                if self.bvc_active: 
                        self.bvc_weight = 0.5
                        self.gc_weight = 0.5
                else:
                    self.gc_weight = 1
            else:
                self.acd_weight = 1/2
                self.bvc_weight = 1/2
            # get weights and adjust to cue position
        elif acd_active:
            self.acd2pcweights = np.where(self.acd_connections > 68, 1, 0)
            self.max_acd_spike_potential = np.sum(np.multiply(self.acd2pcweights, self.acd_connections))
            if self.max_acd_spike_potential == 0:
                self.acd_active = False
                self.gc_weight = 1
            else: 
                self.acd_weight = 0.5
                self.gc_weight = 0.5

        elif self.bvc_active: 
            self.bvc_weight = 0.5
            self.gc_weight = 0.5
        else:
            self.gc_weight = 1

        self.env_coordinates = None  # Save x and y coordinate at moment of creation

        self.plotted_found = [False, False]  # Was used for debug plotting, of linear lookahead

    def compute_firing(self, s_vectors, bvc_activity, acd_activity):
        """Computes firing value based on current grid cell spiking, bvc spiking and acd spiking"""
        # add refactory period = theta cycle
        if self.bvc_active:
            if True in np.isnan(bvc_activity):
                bvc_firing = 0
            else:
                bvc_filtered = np.multiply(self.bvc_excitatory_inhibitory, bvc_activity)
                # TODO Decide which is better or further improvements
                # bvc_firing = min(np.sum(bvc_filtered) / self.max_bvc_spike_potential, 1) # normalize spike / 1 is highest value
                if self.max_bvc_spike_potential:
                    bvc_firing = np.sum(bvc_filtered) / self.max_bvc_spike_potential
                    if bvc_firing > 1:
                        bvc_firing = 2 - bvc_firing
                else:
                    bvc_firing = 0


        # print("bvcfiring: {}".format(bvc_firing))

        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # determine where connection exist to grid cells
        filtered = np.multiply(gc_connections, s_vectors)  # filter current grid cell spiking, by connections.
        # modules_firing = np.sum(filtered, axis=1) / np.sum(s_vectors, axis=1)  # for each module determine pc firing
        modules_firing = np.sum(filtered) / np.sum(s_vectors) # Only when using one grid cell module
        gc_firing = np.average(modules_firing)  # compute overall pc firing by averaging over modules
        if self.acd_active and self.bvc_active:
            acd_activity_filtered = np.where(acd_activity > 68, acd_activity, 0)
            acd_firing = np.divide(np.sum(np.multiply(self.acd2pcweights, acd_activity_filtered)), self.max_acd_spike_potential)
            # print("acd_firing {}".format(acd_firing))

            firing = self.bvc_weight * bvc_firing + self.acd_weight * acd_firing
                
        elif self.acd_active:
            acd_activity_filtered = np.where(acd_activity > 68, acd_activity, 0)
            acd_firing = np.sum(np.divide(np.multiply(acd_activity_filtered, self.acd2pcweights), self.max_acd_spike_potential))
            # acd_firing = calculate_ACD_firing(self.acd2pcweights, acd_activity, self.max_acd_spike_potential)
            firing = self.gc_weight * gc_firing + self.acd_weight * acd_firing
        elif self.bvc_active:
            firing = self.gc_weight * gc_firing + self.bvc_weight * bvc_firing
            if np.amax(acd_activity) > 68:
                firing -= 0.3
        else:
            firing = self.gc_weight * gc_firing
            if np.amax(acd_activity) > 68:
                firing -= 0.3
            if np.sum(bvc_activity) > 0.5:
                firing -= 0.3
        return max(firing, 0)

    def compute_firing_gc_only(self, s_vectors):
        """Computes firing value based on current grid cell spiking"""
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # determine where connection exist to grid cells
        filtered = np.multiply(gc_connections, s_vectors)  # filter current grid cell spiking, by connections
        modules_firing = np.sum(filtered, axis=1) / np.sum(s_vectors, axis=1)  # for each module determine pc firing
        firing = np.average(modules_firing)  # compute overall pc firing by averaging over modules
        return firing

def compute_weights(s_vectors):

    # weights = np.where(s_vectors > 0.1, 1, 0)
    weights = np.array(s_vectors)  # decided to not change anything here, but do it when computing firing

    return weights


class PlaceCellNetwork:
    """A PlaceCellNetwork holds information about all Place Cells"""
    def __init__(self, continuous_fdbk, from_data=False):
        self.continuous_fdbk = continuous_fdbk
        self.place_cells = []  # array of place cells
        if from_data:
            # Load place cells if wanted
            gc_connections = np.load("data/pc_model/gc_connections.npy")
            bvc_connections = np.load("data/pc_model/bvc_connections.npy")
            acd_connections = np.load("data/pc_model/acd_connections.npy")
            acd_active = np.load("data/pc_model/acd_active.npy")
            env_coordinates = np.load("data/pc_model/env_coordinates.npy")

            for idx, gc_connection in enumerate(gc_connections):
                pc = PlaceCell(gc_connection, bvc_connections[idx], acd_connections[idx], acd_active[idx])
                pc.env_coordinates = env_coordinates[idx]
                self.place_cells.append(pc)

    def create_new_pc(self, gc_modules, bvc_activity, acd_activity, acd_active):
        # Consolidate grid cell spiking vectors to matrix of size n^2 x M
        s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
        for m, gc in enumerate(gc_modules):
            s_vectors[m] = gc.s
        weights = compute_weights(s_vectors)
        pc = PlaceCell(weights, bvc_activity, acd_activity, acd_active)
        self.place_cells.append(pc)
        return


    def track_movement(self, gc_modules, bvc_activity, acd_activity, acd_active, reward_first_found=False):
        """Keeps track of current grid cell firing"""
        acd_activity = np.array(acd_activity)
        firing_values = self.compute_firing_values(gc_modules, bvc_activity, acd_activity)

        created_new_pc = False
        if (len(firing_values) == 0 or np.max(firing_values) < 0.7 or reward_first_found) and (training or not self.continuous_fdbk):
            # No place cell shows significant excitement
            # If the random number is above a threshold a new pc is created
            self.create_new_pc(gc_modules, bvc_activity, acd_activity, acd_active)
            firing_values.append(1)
            # creation_str = "Created new place cell with idx: " + str(len(self.place_cells) - 1) \
            #                + " | Highest alternative spiking value: " + str(np.max(firing_values))
            # print(creation_str)
            # if reward_first_found:
            #     print("Found the goal and created place cell")
            created_new_pc = True

        return firing_values, created_new_pc

    def compute_firing_values(self, gc_modules, bvc_activity, acd_activity):

        s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
        # Consolidate grid cell spiking vectors that we want to consider
        for m, gc in enumerate(gc_modules):
            s_vectors[m] = gc.s

        firing_values = []
        for i, pc in enumerate(self.place_cells):
            # NOTE Use compute_firing_gc_only to get original firing values
            # firing = pc.compute_firing_gc_only(s_vectors)
            firing = pc.compute_firing(s_vectors, bvc_activity, acd_activity)  # overall firing
            firing_values.append(firing)
        return firing_values

    def save_pc_network(self, filename=""):
        gc_connections = []
        env_coordinates = []
        bvc_connections = []
        acd_connections = []
        acd_active = []
        for pc in self.place_cells:
            gc_connections.append(pc.gc_connections)
            bvc_connections.append(pc.bvc_connections)
            env_coordinates.append(pc.env_coordinates)
            acd_connections.append(pc.acd_connections)
            acd_active.append(pc.acd_active)

        directory = "data/pc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        np.save("data/pc_model/gc_connections" + filename + ".npy", gc_connections)
        np.save("data/pc_model/bvc_connections" + filename + ".npy", bvc_connections)
        np.save("data/pc_model/acd_connections" + filename + ".npy", acd_connections)
        np.save("data/pc_model/acd_active" + filename + ".npy", acd_active)
        np.save("data/pc_model/env_coordinates" + filename + ".npy", env_coordinates)

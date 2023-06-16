import numpy as np
import random
from tqdm import tqdm
import time
import math

from system.bio_model.gridCellModel import GridCellNetwork
from system.bio_model.placecellModel import PlaceCellNetwork
from system.bio_model.parametersFdbckLoop import training
from system.pybulletEnv import PybulletEnvironment
from system.bvc_model.parametersHDC import weight_av_stim,n_hdc 
from system.bvc_model.parametersNoise import *
from system.bvc_model.hdc_template import generateHDC
import system.bvc_model.helper as helper
import system.bvc_model.parametersBC as parametersBC
import system.bvc_model.BCActivity as BCActivity
import system.bvc_model.HDCActivity as HDCActivity
import system.bvc_model.BCsimulation as BCsimulation
import system.bvc_model.placeEncoding as placeEncoding
import system.bvc_model.hdcNetwork as hdcNetwork

from plotting.pcPlotter import *
from plotting.polarBCplotter import BCplotter
from plotting.polarPlotter import PolarPlotter
from plotting.gcPlotter import GCTracePlotter, GCActivityPlotter, GCCalibrationPlotter

from hdcCalibrConnectivity import ACDScale

dt = 1e-2 # in seconds

# initialize Grid Cell Model
M = 1  # 6 for default, number of modules
n = 40  # 40 for default, size of sheet -> nr of neurons is squared
gmin = 2.4  # 0.2 for default, maximum arena size, 0.5 -> ~10m | 0.05 -> ~105m
gmax = 2.4  # 2.4 for default, determines resolution, dont pick to high (>2.4 at speed = 0.5m/s)
# note that if gc modules are created from data n and M are overwritten
from_data = True # False for default, set to True if you want to load cognitive map data to the model
load_weights = True # For testing realignment
# Choose calibration model for GC network (single point or continuous feedback model) by setting it to "True"
# If no calibration model is chosen (both = False), the GC network will not be recalibrated
continuous_fdbk = True 

# To test the calibration model we can add noise and observe the impact on the grid cell spiking
use_noisy_xy_speed = False 

pc_network = PlaceCellNetwork(continuous_fdbk, from_data)
# If not found run placeCellConnectivity first
gc_network = GridCellNetwork(n, M, dt, gmin, gmax, from_data, load_weights, continuous_fdbk)

# Choose calibration model for HD network (place encoding or simple feedback model) by setting it to "True"
# If no calibration model is chosen (both = False), the HD is only estimated on path integration
# Attention: If both options are set to True, both calibration models are mixed
simple_fdbk = False
place_enc_fdbk = True

# Set calibration mode for the place encoding feedback model
# FGL "on" True: Stores only the ACD, cue distance and the agent's position at the first glance at the cue/landmark
# to reset HD
# FGL "off" False: Associate ACDs in every newly discovered position, to reset HD when revisiting these positions.
FirstGlanceLRN = True
viewAngle = (np.pi)/2

# Attention: The visualization of the vectors is very slow for large matrix dimensions.
# set place encoding parameters
matrixRealLengthCovered = 16 # 4 # size of space in the environment that is encoded by the position encoder
matrixDim = 3*matrixRealLengthCovered # place encoding granularity



############## for running pybullet ##############

visualize = False  # set this to True if you want a simulation window to open (requires computational power)
env_model = "T-shape"
# robot timestep
dt_robot = 0.05
# minimum neuron model timestep
dt_neuron_min = 0.01
# total episode time in seconds
t_episode = 8000# 375 = original from HDC 180:curved 240:maze

if env_model=="maze": vis_cue_pos = [8.5, -7.7, 0]
elif env_model == "T-shape": vis_cue_pos = [0, 8, 0]
elif env_model == "box": vis_cue_pos = [1.05, 0, 0]
elif env_model == "large_box": vis_cue_pos = [0, 7.5, 0]

env = PybulletEnvironment(visualize, env_model, dt, vis_cue_pos)
realDir = env.euler_angle[2]
agentState =  {'Position': None,'Orientation': None}
##################################################

# initialize Position Encoder
PosEncoder = placeEncoding.PositionalCueDirectionEncoder(vis_cue_pos[0:2], matrixDim, matrixRealLengthCovered, FirstGlanceLRN=FirstGlanceLRN)
ACD_PeakActive = False

timesteps_neuron = math.ceil(dt_robot/dt_neuron_min)
dt_neuron = dt_robot / timesteps_neuron

print("neuron timesteps: {}, dt={}".format(timesteps_neuron, dt_neuron))

##################   Plotting ####################
pcPlot = False # Set to True to plot PlaceCell firing
bvcPlot = False # Set to True to plot BVcell firing
hdcPlot = True # Set to True to plot HDcell firing
gcPlot = False # Set to True to plot GridCell firing
gcAPlot = True # Set to True to plot GridCellActivity firing

nextplot = time.time()
plotfps = 0.025
if hdcPlot:
    hdcPlotter = PolarPlotter(n_hdc, 0.0, False, PosEncoder)
if bvcPlot:
    bcPlotter = BCplotter()
if gcPlot:
    gcPlotter = GCTracePlotter(env_model, from_data)
if gcAPlot:
    gcActivityPlotter = GCActivityPlotter()

# variables needed for plotting
eBCSummed = 0
bvcSummed = 0
eBCRates = []
bvcRates = []
eachRateDiff = []
xPositions = []
yPositions = []
sampleX = []
sampleY = []
sampleT = []
bcTimes = []

if pcPlot:
    plotter = Plotter(env_model)
    if from_data: # If we load Place Cells from data, the coordinates of the place cells have to be added at the start.
        for i in range(len(pc_network.place_cells)):
            plotter.appendNewSpikingCoordinate(pc_network.place_cells[i].env_coordinates)

##################################################

# init HDC
print("-> generate HDC") # takes a few minutes
hdc = generateHDC(InitHD=realDir, place_enc_fdbk=place_enc_fdbk, simpleFdbk=simple_fdbk)
print("-> HDC generated")

# rad to deg factor
r2d = (360 / (2*np.pi))

avs = []
errs = []
errs_signed = []
robotTimes = []
thetas = []
netTimes = []
errs_noisy_signed = []
noisyDir = 0.0
decodeTimes = []
plotTimes = []

t_before = time.time()
t_ctr = 0

firing_values = []
feedback = None

for t in tqdm(np.arange(0.0, t_episode, dt_robot)):

    def getStimL(ahv):
        if ahv < 0.0:
            return 0.0
        else:
            return ahv * weight_av_stim
    def getStimR(ahv):
        if ahv > 0.0:
            return 0.0
        else:
            return - ahv * weight_av_stim
    def getNoisyTheta(theta, t):
        noisy_theta = theta
        # gaussian noise
        if noisy_av_rel_sd != 0.0:
            noisy_theta = random.gauss(noisy_theta, noisy_av_rel_sd * theta)
        if noisy_av_abs_sd != 0.0:
            noisy_theta = random.gauss(noisy_theta, noisy_av_abs_sd * dt_robot * (1/r2d))
        # noise spikes
        if noisy_av_spike_frequency != 0.0:
            # simplified, should actually use poisson distribution
            probability = noisy_av_spike_frequency * dt_robot
            if random.random() < probability:
                deviation = random.gauss(noisy_av_spike_magnitude * dt_robot * (1/r2d), noisy_av_spike_sd * dt_robot * (1/r2d))
                # print(deviation)
                if random.random() < 0.5:
                    noisy_theta = noisy_theta + deviation
                else:
                    noisy_theta = noisy_theta - deviation
        # noise oscillation
        if noisy_av_osc_magnitude != 0.0:
            noisy_theta += noisy_av_osc_magnitude * dt_robot * (1/r2d) * np.sin(noisy_av_osc_phase + noisy_av_osc_frequency * t)
        return noisy_theta
    def getNoisyXYSpeed(xy_speed):
        # Noise to test stabilization capability
        if random.random() < 0.75:
            noisy_xspeed = xy_speed[0] + abs(random.gauss(0, xy_speed[0] * 0.05))
            noisy_yspeed = xy_speed[1] + abs(random.gauss(0, xy_speed[1] * 0.05))
        else:
            noisy_xspeed, noisy_yspeed = xy_speed
        return [noisy_xspeed, noisy_yspeed]

    #robot simulation step
    beforeStep = time.time()
    theta = env.compute_movement(gc_network, pc_network)
    afterStep = time.time()
    robotTimes.append((afterStep - beforeStep))
    thetas.append(theta)
    agentState['Position'] = env.xy_coordinates[-1]
    agentState['Orientation'] = env.orientation_angle[-1]

    # current is calculated from angular velocity
    angVelocity = theta * (1.0/dt_robot)
    # add noise
    noisy_theta = getNoisyTheta(theta, t)
    av_net = noisy_theta * (1.0/dt_robot) if use_noisy_av else angVelocity
    avs.append(angVelocity)
    stimL = getStimL(av_net)
    stimR = getStimR(av_net)
    # print(av_net, stimL, stimR)

    ############## BVC and HDC networks ##################

    # Calculate the visual cue's ego- & allocentric direction from agent's position & orientation
    ACDir = helper.calcACDir(agentState['Position'], vis_cue_pos[0:2])
    ECDir = helper.calcECDir(agentState['Orientation'], ACDir)

    # simulate hdc network
    beforeStep = time.time()
    hdc.setStimulus('hdc_shift_left', lambda _ : stimL)
    hdc.setStimulus('hdc_shift_right', lambda _ : stimR)

    ## Set stimulus to ACD-layer & ECD-layer
    # Check if the visual cue is in the agent's field of vision
    # NOTE For simplification reasons work with 360° field of view for cue
    cueInSight = True #helper.cueInSight(viewAngle, ECDir)

    # Check if the agent is in range for learning or restoring the cue's allocentric cue direction
    cueInRange = PosEncoder.checkRange(agentState['Position'])

    if (cueInRange == True) and (cueInSight == True) :

        # Set ECD stimuli to ECD cells
        hdcNetwork.setPeak(hdc, 'ecd_ring', ECDir)

        # save positions in which the agent perceived the cue (only testing)
        #cue_view_pos.append(agentState['Position'])
        #cue_view_pos.append(t)

        if (ACD_PeakActive == True):
            # decode the ACD encoded by ACD cells
            # ACD_PeakActive takes care that ACD is only derived from a fully emerged activity peak
            decodedACDDir = helper.decodeRingActivity(list(hdc.getLayer('acd_ring')))
            # Set ACD = False if ACD learned in new position
            # Set ACD = restored ACD for calibration
            ACD = PosEncoder.get_set_ACDatPos(agentState['Position'], decodedACDDir)

            if (ACD != False):
                # Set ACD stimuli to ACD cells
                hdcNetwork.setPeak(hdc, 'acd_ring', ACD,scale=(1-ACDScale))
        ACD_PeakActive = True
    else:
        # Set stimuli = 0 when cue is out of sight
        hdc.setStimulus('ecd_ring', lambda i: 0)
        hdc.setStimulus('acd_ring', lambda i: 0)
        ACD_PeakActive = False

    hdc.step(dt_neuron, numsteps=timesteps_neuron)
    afterStep = time.time()
    netTimes.append((afterStep - beforeStep) / timesteps_neuron)

    rates_hdc       = list(hdc.getLayer('hdc_attractor'))
    rates_sl        = list(hdc.getLayer('hdc_shift_left'))
    rates_sr        = list(hdc.getLayer('hdc_shift_right'))
    rates_ecd       = list(hdc.getLayer('ecd_ring'))
    rates_conj      = list(hdc.getLayer('Conj'))
    rates_acd       = list(hdc.getLayer('acd_ring'))
    rates_conj_2    = list(hdc.getLayer('Conj2'))
    rates_hdc2      = list(hdc.getLayer('hdc_ring_2'))

    # decode direction, calculate errors
    beforeStep = time.time()

    decodedDir = helper.decodeRingActivity(rates_hdc)
    decodedECDDir   = helper.decodeRingActivity(rates_ecd)
    decodedACDDir   = helper.decodeRingActivity(rates_acd)

    realDir = (realDir + theta) % (2 * np.pi)
    noisyDir = (noisyDir + noisy_theta) % (2 * np.pi)

    err_noisy_signed_rad = helper.angleDist(realDir, noisyDir)
    errs_noisy_signed.append(r2d * err_noisy_signed_rad)
    err_signed_rad = helper.angleDist(realDir, decodedDir)
    errs_signed.append(r2d * err_signed_rad)
    errs.append(abs(r2d * err_signed_rad))
    
    afterStep = time.time()
    decodeTimes.append(afterStep - beforeStep)

    #bc model
    beforeBCStep = time.time()
    raysThatHit = env.getRays()
    polar_angles = np.linspace(0, 2*math.pi - parametersBC.polarAngularResolution, 51)   # - parametersBC.polarAngularResolution

    rayDistances = np.array(raysThatHit)
    rayAngles = np.where(rayDistances == -1, rayDistances, polar_angles)

    noEntriesRadial = np.where(rayDistances == -1)
    noEntriesAngular = np.where(rayAngles == -1)

    # get boundary points
    thetaBndryPts = np.delete(rayAngles, noEntriesAngular)
    if env_model == "maze" or "curved" or "eastEntry":
        rBndryPts = np.delete(rayDistances, noEntriesRadial) * parametersBC.scalingFactorK # scaling factor to match training environment 16/rayLen from pyBullet_environment
    else:
        rBndryPts = np.delete(rayDistances, noEntriesRadial) * 16

    egocentricBCActivity = BCActivity.boundaryCellActivitySimulation(thetaBndryPts, rBndryPts)
    ratesHDC_Amir = np.array(rates_hdc)
    rescaler = max(rates_hdc)
    # Not sure whether that clipping is biologically plausible but without it to many tr Layers are activated
    ratesHDC_Amir = np.where(ratesHDC_Amir / rescaler >= 0.8, ratesHDC_Amir / rescaler, 0)
    ratesHDCSimple = HDCActivity.headingCellsActivityTraining(decodedDir) # gaussian for each neurons activity
    transLayers, bvcActivity = BCsimulation.calculateActivities(egocentricBCActivity, ratesHDCSimple)

    afterBCStep = time.time()
    bcTimes.append(afterBCStep-beforeBCStep)

    ######################################################

    ################ GC and PC networks ##################

    xy_speed =  getNoisyXYSpeed(env.xy_speeds[-1]) if use_noisy_xy_speed else env.xy_speeds[-1]

    # grid cell network track movement

    gc_network.track_movement(xy_speed, firing_values) #TODO add feedbackloop if existing

    # place cell network track gc firing
    # NOTE option 1: pass bvcactivity alogn with gcmodules
    # TODO do something with the feedback and the PC2GC...
    firing_values, created_new_pc = pc_network.track_movement(gc_network.gc_modules, bvcActivity, rates_acd, ACD_PeakActive)
    if created_new_pc:
        pc_network.place_cells[-1].env_coordinates = np.array(env.xy_coordinates[-1])
        # print("ACDPeakActive: {}".format(ACD_PeakActive))
        #TODO add new pc2gc weights to grid cells
        if continuous_fdbk:
            gc_network.add_PC2GC_weight_vectors()

    print("average pc_firing: {}".format(np.average(firing_values)))
    ######################################################

    ##################### plotting #######################
    ## for difference and plotting values
    eBCSummed += np.sum(egocentricBCActivity)
    bvcSummed += np.sum(bvcActivity)
    diff = np.sum(bvcActivity) - np.sum(egocentricBCActivity)
    eachRateDiff.append(diff)

    xPositions.append(env.xy_coordinates[0])
    yPositions.append(env.xy_coordinates[1])
    eBCRates.append(np.sum(egocentricBCActivity))
    bvcRates.append(np.sum(bvcActivity))
    if t % 2 == 0:
        print(rates_acd)
        nextplot += 1.0 / plotfps
        beforeStep = time.time()
        if hdcPlot:
            hdcPlotter.plot(rates_hdc, rates_sl, rates_sr, rates_ecd, rates_conj, rates_acd, rates_conj_2, rates_hdc2, PosEncoder, stimL, stimR, realDir, decodedDir)

        if bvcPlot:
            bcPlotter.bcPlotting(egocentricBCActivity, bvcActivity, transLayers[:, 5], transLayers[:, 15], transLayers[:, 14], decodedDir)
        
        if gcPlot:
            gcPlotter.drawMotion(env.xy_coordinates[-1], gc_network.gc_modules[0].s)
        if gcAPlot:
            gcActivityPlotter.drawPeaks(gc_network.gc_modules[0].s)

        if pcPlot:
            plotter.animate(firing_values, env.xy_coordinates[-1], created_new_pc)
        afterStep = time.time()
        plotTimes.append((afterStep - beforeStep))
    elif created_new_pc and pcPlot:
        plotter.animate(firing_values, env.xy_coordinates[-1], True)

    afterStep = time.time()
    t_ctr += 1
    ######################################################

gc_network.save_calibration_data()
pc_network.save_pc_network()  # provide filename="_navigation" to avoid overwriting the exploration phase
if training:
    gc_network.save_PC2GC_weights()
if pcPlot:
    plotter.save_plot()
if gcPlot:
    gcPlotter.save_plot()

env.end_simulation()

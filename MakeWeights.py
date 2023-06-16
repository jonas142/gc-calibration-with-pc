import time
import os

import system.bvc_model.BCActivity as BCActivity
import system.bvc_model.HDCActivity as HDCActivity
import math
import numpy as np
import system.bvc_model.parametersBC as p
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

#initiate weight tensors which will later contain synaptic weights to generate network activity

ego2TransformationWts = np.zeros(( p.nrBVC, p.nrBVC, p.nrTransformationLayers,))  # 816x816x20 3D-Array
transformation2BVCWts = np.eye(p.nrBVC, 20)  # 816x816 matrix = identity matrix
heading2TransformationWts = np.zeros((p.nrBVC, p.nrHDC, p.nrTransformationLayers))  # 816x100x20 3D-Array

timePerStep = []
timeForThousand = []
allBefore = time.time()

for count in range(p.nrSteps):
    if count % 100 == 0:  # keep track of training
        print(count)
    # first create random starting point of boundary
    angle_i = 2 * math.pi * random.random()  # random polar coordinates
    distance_i = p.maxR * random.random()
    x_i = distance_i * math.cos(angle_i)  # convert to cartesian x,y in allocentric reference frame
    y_i = distance_i * math.sin(angle_i)

    # end point of boundary segment
    angle_i_end = 2 * math.pi * random.random()
    x_i_end = angle_i_end * math.cos(angle_i_end) + x_i
    y_i_end = angle_i_end * math.sin(angle_i_end) + y_i

    boundary_pointAllocentric = np.array([x_i, y_i, x_i_end, y_i_end])


    BVCrate = BCActivity.boundaryCellActivityTraining(boundary_pointAllocentric)

    transformationLayer = math.floor(p.nrTransformationLayers * random.random() + 1)
    heading = p.transformationAngles[transformationLayer - 1]   # 0 indexing
    # create egocentric point of view from the allocentric point of view by rotating the edges
    rxi = x_i * math.cos(heading) + y_i * math.sin(heading)
    ryi = -x_i * math.sin(heading) + y_i * math.cos(heading)
    rxf = x_i_end * math.cos(heading) + y_i_end * math.sin(heading)
    ryf = -x_i_end * math.sin(heading) + y_i_end * math.cos(heading)
    boundary_pointEgocentric = np.array([rxi, ryi, rxf, ryf])

    egocentricRate = BCActivity.boundaryCellActivityTraining(boundary_pointEgocentric)
    ego2TransformationWts[:, :, transformationLayer - 1] = ego2TransformationWts[:, :, transformationLayer - 1] + np.outer(egocentricRate, np.transpose(BVCrate))

for index in range(p.nrTransformationLayers):
    heading = p.transformationAngles[index]
    HDCrate = HDCActivity.headingCellsActivityTraining(heading)
    HDCrate = np.where(HDCrate < 0.01, 0, HDCrate)
    # HDCrate = sparse(HDCrate)     for better computational costs
    heading2TransformationWts[:, :, index] = np.outer(np.ones(816), HDCrate)

#rescaling
divtmp = np.zeros((816, 816))
for index in range(20):
    ego2TransformationWts[:, :, index] = ego2TransformationWts[:, :, index] / np.amax(ego2TransformationWts[:, :, index])
    divtmp = np.outer(np.sum(ego2TransformationWts[:, :, index], 1), np.ones(816))
    ego2TransformationWts[:, :, index] = np.divide(ego2TransformationWts[:, :, index], divtmp)
allAfter = time.time()
print(str(allAfter-allBefore))

directory = "data/bvc_model/"
if not os.path.exists(directory):
    os.makedirs(directory)
    
np.save("data/bvc_model/ego2TransformationWts", ego2TransformationWts)
np.save("data/bvc_model/heading2TransformationWts", heading2TransformationWts)
np.save("data/bvc_model/transformation2BVCWts", transformation2BVCWts)

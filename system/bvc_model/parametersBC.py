import math
import numpy as np

hRes = 0.5
maxR = 16
maxX = 12.5
maxY = 6.25
minX = -12.5
minY = -12.5
polarDistRes = 1
resolution = 0.2 
polarAngularResolution = (2 * math.pi) / 51  # 51 BCs distributed around the circle
hSig = 0.5
nrHDC = 100  # Number of HD neurons
nrSteps = 10000 # training steps
hdActSig = 0.1885
angularDispersion = 0.2236

radialRes = 1
maxRadius = 16
nrBCsRadius = round(maxRadius / radialRes)

transformationRes = math.pi / 10  # for 20 Layers
nrTransformationLayers = int((2 * math.pi) // transformationRes)

transformationAngles = np.linspace(0, 2*math.pi, 20)

nrBVCRadius = round(maxR / polarDistRes)
nrBVCAngle = ((2 * math.pi - 0.01) // polarAngularResolution) + 1
nrBVC = int(nrBVCRadius * nrBVCAngle)

######Simulation#########
#set sensor length from simulation the longer the narrower space appears in the same environment
rayLength = 5
scalingFactorK = maxRadius / rayLength

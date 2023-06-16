import numpy as np

ego2trans = np.load("data/bvc_model/ego2TransformationWts.npy")
heading2trans = np.load("data/bvc_model/heading2TransformationWts.npy")
trans2BVC = np.load("data/bvc_model/transformation2BVCWts.npy")
# clipping small weights makes activities sharper
#ego2trans = np.where(ego2trans >= np.max(ego2trans * 0.3), ego2trans, 0)

# rescaling as in BB-Model
ego2trans = ego2trans * 50
trans2BVC = trans2BVC * 35
heading2trans = heading2trans * 15
def calculateActivities(egocentricActivity, heading):
    '''
    calculates activity of all transformation layers and the BVC layer by multiplying with respective weight tensors
    :param egocentricActivity: egocentric activity which was previously calculated in @BCActivity
    :param heading: HDC networks activity
    :return: activity of all TR layers and BVC layer
    '''
    ego = np.reshape(egocentricActivity, 816)
    transformationLayers = np.einsum('i,ijk -> jk', ego, ego2trans)

    maxTRLayers = np.amax(transformationLayers)
    if maxTRLayers > 0:
        transformationLayers = transformationLayers / maxTRLayers
    headingIntermediate = np.einsum('i,jik -> jk ', heading, heading2trans)
    headingScaler = headingIntermediate[0, :]
    scaledTransformationLayers = np.ones((816, 20))
    for i in range(20):
        scaledTransformationLayers[:, i] = transformationLayers[:, i] * headingScaler[i]
    bvcActivity = np.sum(scaledTransformationLayers, 1)
    maxBVC = np.amax(bvcActivity)
    if maxBVC > 0:
        bvcActivity = bvcActivity/maxBVC

    return transformationLayers, bvcActivity

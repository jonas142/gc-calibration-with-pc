import numpy as np

def get_And_Adjust_ACD2PC(ACD2PC, initialInput):
    """ Adjusts ACD2PC weights to recognize PC  specific pattern """
    indices = np.where(initialInput == np.amax(initialInput))
    rotationFactor = 100 - indices[0][0]
    adjustedACD2PC = np.append(ACD2PC[rotationFactor:], ACD2PC[:rotationFactor])
    maxSpike = np.sum(np.multiply(adjustedACD2PC, np.divide(initialInput, 100)))
    print(maxSpike)
    return adjustedACD2PC, maxSpike

def calculate_ACD_firing(ACD2PC, momentaryInput, maxSpikeValue):
    """ Computes the PC activity in relation to the ACD spiking """
    # Input has to be divided by 100 as otherwise overflow during training can occur
    momentaryInput = np.divide(momentaryInput, 100)
    return min(np.power(np.sum(np.multiply(ACD2PC, momentaryInput)) / maxSpikeValue, 7), 1)

import math
import random

#global variables
totalErrors = 0

def sigmoid(x):
    print(f"Test: {(1 + math.exp(-x))}")
    return 1 / (1 + math.exp(-x))

def magnitude(pos):
    magnitude = 0

    for dim in pos:
        magnitude += dim**2

    magnitude = magnitude**0.5

    return magnitude

#Not Used Right Now
# def dictonaryToArray(dict):
#     array = []    
#     counter = 0

#     for key in dict:
#         array.append(key)
#         array[counter].append(dict[key])
#         counter += 1
    
#     return array

#function that is called when theres an error and prints to the user what the error was and the line it occured
def throwError(errorCode, line):
    global totalErrors
    errorDictonary = {1:"Trying to make more synapses than neurons avaiable for synapse creation.", 2:"There was an attempted operation on a type that shouldnt be used in that operation.", 3:"Output would produce a negative value region", 4:"Required Conditions for the function call has not been met.", 5:"Mismatch in size between inputs size and the number input neurons", 6:"Mismatch in size between expected outputs size and the actual outputs size"}
    print(f"There was an Error on Line {line}\nError: {errorDictonary[errorCode]}")
    totalErrors += 1

#assumes the list is already sorted
def insertSorted(lst, value):
    if len(lst) == 0:
        lst.append(value)
        return

    topBound = len(lst)
    bottomBound = 0

    while bottomBound < topBound:
        index = (topBound + bottomBound) // 2

        if type(value).__name__ == 'tuple':
            if lst[index][0] == value[0]:
                lst.insert(index, value)
                return
            elif lst[index][0] < value[0]:
                bottomBound = index + 1
            else:
                topBound = index
        else:
            if lst[index] == value:
                lst.insert(index, value)
                return
            elif lst[index] < value:
                bottomBound = index + 1
            else:
                topBound = index

    lst.insert(bottomBound, value)

#function to calculate the total region a set of dimensions takes this could be area, volume, ect.
def calcTotalSpace(dimensionsTuple):
    region = 1

    for i in range(len(dimensionsTuple)):
        if type(dimensionsTuple[i]).__name__ == 'int':
            region *= dimensionsTuple[i]
        else:
            throwError(2, 61)
    
    return region

#function to calculate difference in dimensions between regions
def dimensionalDifference(region1, region2):
    if len(region1) > len(region2):
        dimensionRestriction = (0,) * len(region2) + (1,) * (len(region1) - len(region2))
    elif len(region1) < len(region2):
        throwError(3, 76)
        dimensionalRestriction = (0,) * len(region1)
    else:
        dimensionRestriction = (0,) * len(region1)
    
    return dimensionRestriction

#function to project the dimensions of one region to the dimension of another region
def project(region1, region2):
    regionProjection = ()

    if len(region1) > len(region2):
        #error might occur here if the regions have dimensions 2 or under. dont feel like fixing or checking
        regionProjection = region2 + (1,) * (len(region1) - len(region2))
    elif len(region1) < len(region2):
        dimensionRestriction = region2[:len(region1)]
    else:
        regionProjection = region2
    
    return regionProjection

#function to subract tuples that represent reigons or dimensions the function also has the ability to subract an entire reigon restricted to a set(s) of axis
def subtractReigons(region1, region2):
    newRegion = 0
    if type(region1).__name__ == 'int' and type(region2).__name__ == 'int':
        newRegion = region1 - region2
        return newRegion
    else:
        region1 = calcTotalSpace(region1)
        region2 = calcTotalSpace(region2)

        newRegion = region1 - region2
        return newRegion

#function that subtracts the dimensions only of two regions
def subtractRegionDimensions(region1, region2):
    newRegion = []
    regionConstrictor = dimensionalDifference(region1, region2)

    for i in range(len(region1)):
        newRegion.append(region1[i] - regionConstrictor[i])
    
    return tuple(newRegion)

def sumTuple(tuple):
    sum = 0

    for i in range(len(tuple)):
        sum += tuple[i]
    
    return sum

        
class Brain:
    def __init__(self, brainDim, inputDim, outputDim):
        #variables for brain data
        self.brainDimensions = brainDim
        self.inputDimension = inputDim
        self.outputDimension = outputDim
        self.maxNumNeurons = calcTotalSpace(subtractRegionDimensions(subtractRegionDimensions(self.brainDimensions, self.inputDimension), self.outputDimension)) #I thought it would be funny
        self.neuronDimension = None
        self.inputNeurons = {}
        self.outputNeurons = {}
        self.synapses = {}
        self.neurons = {}
        self.allNeurons = {}

        self.generateInputNeurons()
        self.generateOutputNeurons()

    def generateNeurons(self, numNeurons):
        coordinate = ()
        neuron = None
        if len(self.neurons) == (self.brainDimensions[0] * self.brainDimensions[1] * (self.brainDimensions[2] - 2)):
            print("At Max capacity of Neurons!")
            return
        
        totalNumNeurons = len(self.neurons) + numNeurons

        while len(self.neurons) < totalNumNeurons and len(self.neurons) < self.maxNumNeurons:
            #YES I KNOW THIS HAS HORRIBLE TIME COMPLEXITY ILL DO SOMETHING BETTER LATER!
            coordinate = (random.randint(0, self.brainDimensions[0] - 1), random.randint(0, self.brainDimensions[1] - 1), random.randint(1, self.brainDimensions[2] - 2))

            if not(coordinate in self.neurons):
                self.neurons[coordinate] = Neuron(coordinate)
        
        self.allNeurons.update(self.neurons)

    def generateInputNeurons(self):
        pos = None

        for z in range(self.inputDimension[2]):
            for y in range(self.inputDimension[1]):
                for x in range(self.inputDimension[0]):
                    pos = (x, y, z)
                    self.inputNeurons[pos] = InputNeuron(pos)
        
        self.allNeurons.update(self.inputNeurons)

    def generateOutputNeurons(self):
        pos = None

        for y in range(self.outputDimension[1]):
            for x in range(self.outputDimension[0]):
                pos = (x, y, 9)
                self.outputNeurons[pos] = OutputNeuron(pos)
        self.allNeurons.update(self.outputNeurons)

    #function that checks wether all the conditions for generating synapses are correct
    def checkSynapseGenConditions(self):
        neuronCheck = self.getNoOutputStatus()

        if all(neuronCheck):
            return True
        else:
            return False

    def generateSynapses(self, preNeuron, numSynapses):
        if self.checkSynapseGenConditions():
            #list that holds all the synapse connections for the preNeuron
            synapses = []

            #variable that holds a list containing all the neurons and their distance from the current neuron
            neuronDistancesAhead = self.findNeuronDistancesFrom(preNeuron.position)[1]

            if len(neuronDistancesAhead) >= numSynapses:
                for i in range(numSynapses):
                    synapseObject = Synapse(preNeuron, neuronDistancesAhead[i])
                    synapses.append(synapseObject)
                    neuronDistancesAhead[i][1].presynapticConnections.append(synapseObject)
            else:
                for i in range(len(neuronDistancesAhead)):
                    synapseObject = Synapse(preNeuron, neuronDistancesAhead[i])
                    synapses.append(synapseObject)
                    neuronDistancesAhead[i][1].presynapticConnections.append(synapseObject)
            
            preNeuron.postsynapticConnections = synapses
            self.synapses[preNeuron] = synapses
        else:
            throwError(4, 201)

    def findNeuronDistancesFrom(self, currentPos):
        neuronsBehind = []
        neuronsAhead = []

        distanceMagnitude = 0
        deltaXPos = 0
        deltaYPos = 0
        deltaZPos = 0

        for pos in self.allNeurons:
            deltaXPos = self.allNeurons[pos].position[0] - currentPos[0]
            deltaYPos = self.allNeurons[pos].position[1] - currentPos[1]
            deltaZPos = self.allNeurons[pos].position[2] - currentPos[2]

            magnitudeDistance = magnitude((deltaXPos, deltaYPos, deltaZPos))

            if deltaZPos <= 0:
                insertSorted(neuronsBehind, (magnitudeDistance, self.allNeurons[pos]))
            else:
                insertSorted(neuronsAhead, (magnitudeDistance, self.allNeurons[pos]))
        
        return (neuronsBehind, neuronsAhead)

    def getStatus(self):
        #boolean variables that check whether each part of the network contains neurons
        containsNeurons = True if len(self.neurons) > 0 else False
        containsInputNeurons = True if len(self.inputNeurons) > 0 else False
        containsOutputNeurons = True if len(self.outputNeurons) > 0 else False

        #print status
        print("Status:")
        print("==============================") #spacer
        print(f"Current Dimensions For Inputs: {self.inputDimension}")
        print(f"Current Number Of Input Neurons: {len(self.inputNeurons)}")
        print("------------------------------") #spacer
        print(f"Current Dimensions For Outputs: {self.outputDimension}")
        print(f"Current Number Of Output Neurons: {len(self.outputNeurons)}")
        print("------------------------------") #spacer
        print(f"Current Number Of Neurons: {len(self.neurons)}")
        print("------------------------------") #spacer
        print(f"Maximum All Neuron Capacity: {calcTotalSpace(self.brainDimensions)}")
        print(f"Current Number Of All Neurons: {len(self.neurons) + len(self.outputNeurons) + len(self.inputNeurons)}")
        print(f"Remaining Empty Positions For Neurons: {self.maxNumNeurons - len(self.neurons)}")
        print("------------------------------") #spacer
        print(f"Total Errors Thrown: {totalErrors}")
        print("==============================") #spacer

    def getNoOutputStatus(self):
        containsNeurons = True if len(self.neurons) > 0 else False
        containsInputNeurons = True if len(self.inputNeurons) > 0 else False
        containsOutputNeurons = True if len(self.outputNeurons) > 0 else False

        return (containsInputNeurons, containsOutputNeurons, containsNeurons)
    
    #function to generate all the synapses for all the neurons
    def buildAllConnections(self, numSynapses):
        #if statement to check that the conditions for generating synapses has been met
        if self.checkSynapseGenConditions():
            for key in self.inputNeurons:
                self.generateSynapses(self.inputNeurons[key], numSynapses[0])
            for key in self.neurons:
                self.generateSynapses(self.neurons[key], numSynapses[1])
        else:
            throwError(4, 192)
    
    #function that sends the output of the presynaptic neuron through the connection to the postsynaptic neuron 
    def processInput(self, neuron):
        if type(neuron).__name__ != 'OutputNeuron':
            neuroTransmitterOutput = round(sigmoid(neuron.neuroTransmitterInput))

            if neuroTransmitterOutput == 1:
                for synapse in neuron.postsynapticConnections:
                    synapse.postSynapticNeuron[1].neuroTransmitterInput += neuroTransmitterOutput * synapse.connectionStrength #corrected by ai, added ".connectionStrength"
                    synapse.hasFired = True
                    self.processInput(synapse.postSynapticNeuron[1])
        else:
            neuron.outputValue = round(sigmoid(neuron.neuroTransmitterInput))
        
    def calcHormoneRelease(self, actualOutputs, expectedOutputs):
        if len(expectedOutputs) == len(actualOutputs):
            actualOutputs = tuple(actualOutputs)

            for i in range(len(actualOutputs)):
                if actualOutputs[i] != expectedOutputs[i]:
                    return -1
            
            return 1
        else:
            throwError(6, 306)
            
    def updateSynapseConnections(self, hormoneLevel):
        for preNeuron in self.synapses:
            for synapse in self.synapses[preNeuron]:
                synapse.updateConnectionStrength(hormoneLevel)
    
    def displayOutputInfo(self, actualOutputs, hormoneLevel, expectedOutputs):
        print("Output Info:")
        print("")
        print("==============================")
        print(f"Outputs that Were Expected: {expectedOutputs}")
        print(f"Outputs that we Got: {actualOutputs}")
        print("==============================")
        print("")
        print("==============================")
        print(f"Hormone Level Outputed: {hormoneLevel}")
        print("==============================")
        print("")
        print("NeuroTransmitter Inputs To the Output Neurons:")
        print("==============================")
        
        counter = 1
        for pos in self.outputNeurons:
            print(f"Input {counter}: {self.outputNeurons[pos].neuroTransmitterInput}")
            print("==============================")

        print("")
        print("Synapses To Output Neurons:")
        print("==============================")
        print("")

        counter = 1
        counter2 = 1
        for pos in self.outputNeurons:
            print("")
            print(f"Output Neuron {counter}")
            print("==============================")
            counter += 1

            for synapse in self.outputNeurons[pos].presynapticConnections:
                print(f"Synapse {counter2}: {synapse}")
                counter2 += 1

            print("==============================")
        
        print("==============================")

    #function that runs the brain with the defined inputs
    def run(self, inputs, expectedOutputs):
        if len(inputs) == len(self.inputNeurons):
            counter = 0
            for pos in self.inputNeurons:
                self.inputNeurons[pos].neuroTransmitterInput = inputs[counter]
                self.processInput(self.inputNeurons[pos])
                counter += 1
        else:
            throwError(5, 306)
        
        #get all the outputs and send it to the calcLoss function
        outputs = []

        for pos in self.outputNeurons:
            outputs.append(self.outputNeurons[pos].outputValue)
        
        hormones = self.calcHormoneRelease(outputs, expectedOutputs)

        #update all the synapses
        self.updateSynapseConnections(hormones)

        #print outputs
        self.displayOutputInfo(outputs, hormones, expectedOutputs)

class Synapse:
    def __init__(self, preNeuron, postNeuron):
        self.preSynapticNeuron = preNeuron
        self.postSynapticNeuron = postNeuron
        self.connectionStrength = random.random() * (1 if random.random() < 0.8 else -1)
        self.hasFired = False

    def updateConnectionStrength(self, hormoneMultiplier):
        if self.hasFired == False: #probably dont need " and ((self.connectionStrength - 0.01) >= -1.0)"
            self.connectionStrength -= random.uniform(0.001, 0.01)
        elif self.hasFired == True: #probably dont need "and ((self.connectionStrength + 0.05) <= 1.0)"
            self.connectionStrength += random.uniform(0.001, 0.03) * hormoneMultiplier
            self.hasFired = False

class InputNeuron:
    def __init__(self, pos):
        self.position = pos
        self.neuroTransmitterInput = None
        self.fired = False
        self.neuroTransmitterOutput = 10
        self.postsynapticConnections = None

class Neuron:
    def __init__(self, pos):
        self.position = pos
        self.neuroTransmitterInput = 0
        self.neuroTransmitterOutput = 10
        self.presynapticConnections = []
        self.postsynapticConnections = None

class OutputNeuron:
    def __init__(self, pos):
        self.position = pos
        self.neuroTransmitterInput = 0
        self.outputValue = 0
        self.presynapticConnections = []

brainSize = (10,10,10)
inputSize = (10,10,1)
outputSize = (10,1)
inputs = (
    0,0,0,0,0,0,0,0,0,0,    # Row 0
    0,0,0,0,0,0,0,0,0,0,    # Row 1
    0,0,0,0,1,1,0,0,0,0,    # Row 2
    0,0,0,0,1,1,0,0,0,0,    # Row 3
    0,0,0,0,1,1,0,0,0,0,    # Row 4
    0,0,0,0,1,1,0,0,0,0,    # Row 5
    0,0,0,0,1,1,0,0,0,0,    # Row 6
    0,0,0,0,1,1,0,0,0,0,    # Row 7
    0,0,0,0,1,1,0,0,0,0,    # Row 8
    0,0,0,1,1,1,1,0,0,0     # Row 9
)
outputs = (0,1,0,0,0,0,0,0,0,0)

neuralBrain = Brain(brainSize, inputSize, outputSize)

neuralBrain.generateNeurons(100)

#function call to build all synapses with the parameter in the format (input neuron num connections, neuron num connections)
neuralBrain.buildAllConnections((1,2))

neuralBrain.getStatus()

for i in range(100):
    neuralBrain.run(inputs, outputs)


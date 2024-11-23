import math
import random
import csv
import os

# Global variables
totalErrors = 0

def sigmoid(x):
    # Clamp x to prevent overflow
    x = max(min(x, 709), -709)
    return 1 / (1 + math.exp(-x))

def magnitude(pos):
    return math.sqrt(sum(dim ** 2 for dim in pos))

# Function that is called when there's an error and prints to the user what the error was and the line it occurred
def throwError(errorCode, line):
    global totalErrors
    errorDictionary = {
        1: "Trying to make more synapses than neurons available for synapse creation.",
        2: "There was an attempted operation on a type that shouldn't be used in that operation.",
        3: "Output would produce a negative value region.",
        4: "Required conditions for the function call have not been met.",
        5: "Mismatch in size between inputs size and the number of input neurons.",
        6: "Mismatch in size between expected outputs size and the actual outputs size."
    }
    print(f"There was an Error on Line {line}\nError: {errorDictionary.get(errorCode, 'Unknown Error')}")
    totalErrors += 1

# Assumes the list is already sorted
def insertSorted(lst, value):
    if len(lst) == 0:
        lst.append(value)
        return

    topBound = len(lst)
    bottomBound = 0

    while bottomBound < topBound:
        index = (topBound + bottomBound) // 2

        if isinstance(value, tuple):
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

# Function to calculate the total region a set of dimensions takes (e.g., area, volume)
def calcTotalSpace(dimensionsTuple):
    region = 1

    for dim in dimensionsTuple:
        if isinstance(dim, int):
            region *= dim
        else:
            throwError(2, 61)
            return 0  # Return 0 to prevent further errors

    return region

# Function to subtract tuples that represent regions or dimensions
def subtractRegionDimensions(region1, region2):
    newRegion = []
    min_len = min(len(region1), len(region2))
    for i in range(min_len):
        newRegion.append(region1[i] - region2[i])
    for i in range(min_len, len(region1)):
        newRegion.append(region1[i])

    return tuple(newRegion)

class Brain:
    def __init__(self, brainDim, inputDim, outputDim):
        # Variables for brain data
        self.brainDimensions = brainDim
        self.inputDimension = inputDim  # (28, 28, 1)
        self.outputDimension = outputDim  # (10, 1) for 10 classes
        self.inputNeurons = {}
        self.outputNeurons = {}
        self.synapses = {}
        self.neurons = {}
        self.allNeurons = {}

        self.generateInputNeurons()
        self.generateOutputNeurons()
        self.maxNumNeurons = calcTotalSpace(self.brainDimensions) - len(self.inputNeurons) - len(self.outputNeurons)

    def generateNeurons(self, numNeurons):
        if len(self.neurons) >= self.maxNumNeurons:
            print("At max capacity of neurons!")
            return

        totalNumNeurons = min(len(self.neurons) + numNeurons, self.maxNumNeurons)

        while len(self.neurons) < totalNumNeurons:
            # Random position avoiding input and output neuron positions
            coordinate = (
                random.randint(0, self.brainDimensions[0] - 1),
                random.randint(0, self.brainDimensions[1] - 1),
                random.randint(1, self.brainDimensions[2] - 2)
            )

            if coordinate not in self.neurons and coordinate not in self.inputNeurons and coordinate not in self.outputNeurons:
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
        # Ensure outputDimension has three dimensions
        if len(self.outputDimension) == 2:
            self.outputDimension = (*self.outputDimension, 1)

        for z in range(self.outputDimension[2]):
            for y in range(self.outputDimension[1]):
                for x in range(self.outputDimension[0]):
                    pos = (x, y, self.brainDimensions[2] - self.outputDimension[2] + z)
                    self.outputNeurons[pos] = OutputNeuron(pos)

        self.allNeurons.update(self.outputNeurons)

    # Function that checks whether all the conditions for generating synapses are correct
    def checkSynapseGenConditions(self):
        neuronCheck = self.getNoOutputStatus()

        if all(neuronCheck):
            return True
        else:
            return False

    def generateSynapses(self, preNeuron, numSynapses):
        if self.checkSynapseGenConditions():
            # List that holds all the synapse connections for the preNeuron
            synapses = []

            # Variable that holds a list containing all the neurons and their distance from the current neuron
            neuronDistancesAhead = self.findNeuronDistancesFrom(preNeuron.position)[1]

            if len(neuronDistancesAhead) >= numSynapses:
                for i in range(numSynapses):
                    synapseObject = Synapse(preNeuron, neuronDistancesAhead[i][1])
                    synapses.append(synapseObject)
                    neuronDistancesAhead[i][1].presynapticConnections.append(synapseObject)
            else:
                for i in range(len(neuronDistancesAhead)):
                    synapseObject = Synapse(preNeuron, neuronDistancesAhead[i][1])
                    synapses.append(synapseObject)
                    neuronDistancesAhead[i][1].presynapticConnections.append(synapseObject)

            preNeuron.postsynapticConnections = synapses
            self.synapses[preNeuron] = synapses
        else:
            throwError(4, 201)

    def findNeuronDistancesFrom(self, currentPos):
        neuronsBehind = []
        neuronsAhead = []

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
        # Boolean variables that check whether each part of the network contains neurons
        containsNeurons = True if len(self.neurons) > 0 else False
        containsInputNeurons = True if len(self.inputNeurons) > 0 else False
        containsOutputNeurons = True if len(self.outputNeurons) > 0 else False

        # Print status
        print("Status:")
        print("==============================")  # Spacer
        print(f"Current Dimensions For Inputs: {self.inputDimension}")
        print(f"Current Number Of Input Neurons: {len(self.inputNeurons)}")
        print("------------------------------")  # Spacer
        print(f"Current Dimensions For Outputs: {self.outputDimension}")
        print(f"Current Number Of Output Neurons: {len(self.outputNeurons)}")
        print("------------------------------")  # Spacer
        print(f"Current Number Of Neurons: {len(self.neurons)}")
        print("------------------------------")  # Spacer
        print(f"Maximum All Neuron Capacity: {calcTotalSpace(self.brainDimensions)}")
        print(f"Current Number Of All Neurons: {len(self.neurons) + len(self.outputNeurons) + len(self.inputNeurons)}")
        print(f"Remaining Empty Positions For Neurons: {self.maxNumNeurons - len(self.neurons)}")
        print("------------------------------")  # Spacer
        print(f"Total Errors Thrown: {totalErrors}")
        print("==============================")  # Spacer

    def getNoOutputStatus(self):
        containsNeurons = True if len(self.neurons) > 0 else False
        containsInputNeurons = True if len(self.inputNeurons) > 0 else False
        containsOutputNeurons = True if len(self.outputNeurons) > 0 else False

        return (containsInputNeurons, containsOutputNeurons, containsNeurons)

    # Function to generate all the synapses for all the neurons
    def buildAllConnections(self, numSynapses):
        # Check that the conditions for generating synapses have been met
        if self.checkSynapseGenConditions():
            for key in self.inputNeurons:
                self.generateSynapses(self.inputNeurons[key], numSynapses[0])
            for key in self.neurons:
                self.generateSynapses(self.neurons[key], numSynapses[1])
        else:
            throwError(4, 192)

    # Function that sends the output of the presynaptic neuron through the connection to the postsynaptic neuron
    def processInput(self, neuron, expectedOutputs, actualOutputs, processed_neurons=None):
        if processed_neurons is None:
            processed_neurons = set()

        neuron_id = neuron.position  # Using position as a unique identifier

        if neuron_id in processed_neurons:
            return  # Neuron has already been processed

        processed_neurons.add(neuron_id)

        if not isinstance(neuron, OutputNeuron):
            neuroTransmitterOutput = sigmoid(neuron.neuroTransmitterInput)

            if neuroTransmitterOutput > 0:  # Only proceed if there is some output
                if neuron.postsynapticConnections:
                    for synapse in neuron.postsynapticConnections:
                        synapse.postSynapticNeuron.neuroTransmitterInput += neuroTransmitterOutput * synapse.connectionStrength
                        # Clamp neuroTransmitterInput to prevent extreme values
                        synapse.postSynapticNeuron.neuroTransmitterInput = max(min(synapse.postSynapticNeuron.neuroTransmitterInput, 10), -10)
                        synapse.hasFired = True
                        self.processInput(synapse.postSynapticNeuron, expectedOutputs, actualOutputs, processed_neurons)
        else:
            neuron.outputValue = sigmoid(neuron.neuroTransmitterInput)
            # Determine if the output is moving towards the expected output
            index = list(sorted(self.outputNeurons.keys())).index(neuron.position)
            expected_output = expectedOutputs[index]
            actual_output = neuron.outputValue
            error = expected_output - actual_output
            # Mark synapses leading to this neuron if they contribute to error
            for synapse in neuron.presynapticConnections:
                if (error > 0 and synapse.connectionStrength < 0) or (error < 0 and synapse.connectionStrength > 0):
                    synapse.contributed_to_error = True

    def calcHormoneRelease(self, actualOutputs, expectedOutputs):
        if len(expectedOutputs) == len(actualOutputs):
            # Calculate the Mean Squared Error (MSE)
            errors = [(expected - actual) for expected, actual in zip(expectedOutputs, actualOutputs)]
            mse = sum(error ** 2 for error in errors) / len(errors)
            
            # Calculate hormone level: negative when there's error
            hormoneLevel = - mse * 2  # Scaling factor can be adjusted
            
            # Clamp hormoneLevel to [-1, 0]
            hormoneLevel = max(min(hormoneLevel, 0), -1)
            
            return hormoneLevel
        else:
            throwError(6, 306)
            return 0  # Return 0 to prevent further errors

    def updateSynapseConnections(self, hormoneLevel):
        for preNeuron in self.synapses:
            for synapse in self.synapses[preNeuron]:
                synapse.updateConnectionStrength(hormoneLevel)
                if synapse.contributed_to_error:
                    synapse.disconnect()
                    # Find a new neuron to connect to
                    new_post_neuron = self.find_new_post_neuron(synapse.preSynapticNeuron)
                    if new_post_neuron:
                        synapse.reconnect_to(new_post_neuron)
                    else:
                        # If no suitable neuron is found, remove the synapse
                        pass
                    synapse.contributed_to_error = False  # Reset the flag

    def find_new_post_neuron(self, preNeuron):
        # Get a list of neurons ahead of the preNeuron
        _, neuronsAhead = self.findNeuronDistancesFrom(preNeuron.position)
        # Filter out neurons that are not output neurons and not already connected
        potential_neurons = [neuron for _, neuron in neuronsAhead if neuron not in preNeuron.postsynapticConnections and not isinstance(neuron, OutputNeuron)]
        if potential_neurons:
            # Select a random neuron from the potential neurons
            return random.choice(potential_neurons)
        else:
            return None

    def displayOutputInfo(self, actualOutputs, hormoneLevel, expectedOutputs):
        print("Output Info:")
        print("")
        print("==============================")
        print(f"Outputs that Were Expected: {expectedOutputs}")
        print(f"Outputs that We Got: {actualOutputs}")
        print("==============================")
        print("")
        print("==============================")
        print(f"Hormone Level Outputted: {hormoneLevel}")
        print("==============================")
        print("")
        print("NeuroTransmitter Inputs To the Output Neurons:")
        print("==============================")

        counter = 1
        for pos in sorted(self.outputNeurons):
            print(f"Input {counter}: {self.outputNeurons[pos].neuroTransmitterInput}")
            print("==============================")
            counter += 1

        print("")

    # Function that runs the brain with the defined inputs
    def run(self, inputs, expectedOutputs):
        # Reset neuroTransmitterInput and outputValue of all neurons
        self.clearNeuroTransmitterInputs()

        if len(inputs) == len(self.inputNeurons):
            sorted_input_positions = sorted(self.inputNeurons.keys())
            for counter, pos in enumerate(sorted_input_positions):
                self.inputNeurons[pos].neuroTransmitterInput = inputs[counter]
            # First pass to compute outputs
            for pos in sorted_input_positions:
                self.processInput(self.inputNeurons[pos], expectedOutputs, [])
            # Get all the outputs
            outputs = []
            sorted_output_positions = sorted(self.outputNeurons.keys())
            for pos in sorted_output_positions:
                outputs.append(self.outputNeurons[pos].outputValue)
            # Second pass to mark synapses that contributed to error
            processed_neurons = set()
            for pos in sorted_input_positions:
                self.processInput(self.inputNeurons[pos], expectedOutputs, outputs, processed_neurons)
            # Calculate hormones
            hormones = self.calcHormoneRelease(outputs, expectedOutputs)
            # Update all the synapses
            self.updateSynapseConnections(hormones)
            # Print outputs
            self.displayOutputInfo(outputs, hormones, expectedOutputs)
        else:
            throwError(5, 306)

    def clearNeuroTransmitterInputs(self):
        for neuron in self.neurons.values():
            neuron.neuroTransmitterInput = 0
        for neuron in self.outputNeurons.values():
            neuron.neuroTransmitterInput = 0
            neuron.outputValue = 0
        for neuron in self.inputNeurons.values():
            neuron.neuroTransmitterInput = 0

    # Method to read data from CSV file
    def load_data_from_csv(self, filename):
        if not os.path.exists(filename):
            print(f"CSV file '{filename}' not found.")
            return []

        data = []
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)  # Skip header
            for row_num, row in enumerate(csvreader, start=2):  # Start at 2 to account for header
                if len(row) != 785:
                    throwError(2, row_num)
                    continue
                label = int(row[0])
                pixels = [float(pixel)/255.0 for pixel in row[1:]]  # Normalize pixel values
                data.append((label, pixels))
        return data

    # Method to convert label to one-hot encoding
    def one_hot_encode(self, label, num_classes=10):
        one_hot = [0.0 for _ in range(num_classes)]
        if 0 <= label < num_classes:
            one_hot[label] = 1.0
        else:
            throwError(2, label)
        return tuple(one_hot)

class Synapse:
    def __init__(self, preNeuron, postNeuron):
        self.preSynapticNeuron = preNeuron
        self.postSynapticNeuron = postNeuron
        # Initialize connectionStrength to small values around zero
        self.connectionStrength = random.uniform(-0.5, 0.5)
        self.hasFired = False
        self.contributed_to_error = False  # New attribute to track error contribution

    def updateConnectionStrength(self, hormoneMultiplier):
        if self.hasFired:
            # Make delta proportional to the absolute value of hormoneMultiplier
            delta = random.uniform(0.01, 0.1) * abs(hormoneMultiplier)
            self.connectionStrength += hormoneMultiplier * delta
        self.hasFired = False

        # Clamp connectionStrength to [-1, 1]
        self.connectionStrength = max(min(self.connectionStrength, 1.0), -1.0)

    def disconnect(self):
        # Remove this synapse from the presynaptic neuron's postsynaptic connections
        if self in self.preSynapticNeuron.postsynapticConnections:
            self.preSynapticNeuron.postsynapticConnections.remove(self)
        # Remove this synapse from the postsynaptic neuron's presynaptic connections
        if self in self.postSynapticNeuron.presynapticConnections:
            self.postSynapticNeuron.presynapticConnections.remove(self)

    def reconnect_to(self, new_post_neuron):
        self.postSynapticNeuron = new_post_neuron
        # Add this synapse to the new postsynaptic neuron's presynaptic connections
        new_post_neuron.presynapticConnections.append(self)
        # Add this synapse back to the presynaptic neuron's postsynaptic connections
        self.preSynapticNeuron.postsynapticConnections.append(self)

    def __str__(self):
        return f"Synapse from {self.preSynapticNeuron.position} to {self.postSynapticNeuron.position} with strength {self.connectionStrength:.4f}"

class InputNeuron:
    def __init__(self, pos):
        self.position = pos
        self.neuroTransmitterInput = 0.0
        self.postsynapticConnections = []

class Neuron:
    def __init__(self, pos):
        self.position = pos
        self.neuroTransmitterInput = 0.0
        self.presynapticConnections = []
        self.postsynapticConnections = []

class OutputNeuron:
    def __init__(self, pos):
        self.position = pos
        self.neuroTransmitterInput = 0.0
        self.outputValue = 0.0
        self.presynapticConnections = []

def main():
    # Define network dimensions
    brainSize = (28, 28, 5)  # Increased depth to accommodate more neurons
    inputSize = (28, 28, 1)  # 28x28 input neurons
    outputSize = (10, 1)     # 10 output neurons for classes 0-9

    # Initialize the brain
    neuralBrain = Brain(brainSize, inputSize, outputSize)

    # Generate internal neurons (adjust number as needed)
    neuralBrain.generateNeurons(500)  # Example: 500 internal neurons

    # Build all synapses with the parameter in the format (input neuron num connections, neuron num connections)
    neuralBrain.buildAllConnections((10, 10))  # Example: 10 synapses per input neuron and internal neuron

    # Display initial status
    neuralBrain.getStatus()

    # Load data from CSV
    csv_filename = 'train.csv'  # Replace with your CSV filename
    data = neuralBrain.load_data_from_csv(csv_filename)
    if not data:
        print("No data loaded. Exiting.")
        return

    # Training parameters
    epochs = 10  # Number of times to iterate over the entire dataset

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1} ===")
        random.shuffle(data)  # Shuffle data each epoch for better training

        for sample_num, (label, pixels) in enumerate(data, start=1):
            expected_output = neuralBrain.one_hot_encode(label)
            neuralBrain.run(pixels, expected_output)

            if sample_num % 100 == 0:
                print(f"Processed {sample_num} samples in Epoch {epoch+1}")

    # Final status after training
    neuralBrain.getStatus()

if __name__ == "__main__":
    main()

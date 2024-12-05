import pygame
import random
import math
import sys

#Start Brain Game

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
    print(f"There was an Error on Line {line}\nError: {errorDictionary[errorCode]}")
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
        self.inputDimension = inputDim
        self.outputDimension = outputDim
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
                        synapse.postSynapticNeuron.neuroTransmitterInput = max(min(synapse.postSynapticNeuron.neuroTransmitterInput, 1), -1)
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

            # Map MSE in [0,1] to hormoneLevel in [-1,1]
            hormoneLevel = 1 - 2 * mse

            # Clamp hormoneLevel to [-1,1]
            hormoneLevel = max(min(hormoneLevel, 1), -1)

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

class Synapse:
    def __init__(self, preNeuron, postNeuron):
        self.preSynapticNeuron = preNeuron
        self.postSynapticNeuron = postNeuron
        self.connectionStrength = random.uniform(-0.1, 0.1)  # Adjusted
        self.hasFired = False
        self.contributed_to_error = False  # New attribute to track error contribution

    def updateConnectionStrength(self, hormoneMultiplier):
        delta = random.uniform(0.0001, 0.003)  # Adjusted
        if self.hasFired:
            self.connectionStrength += hormoneMultiplier * delta
        self.hasFired = False

        # Clamp connectionStrength to [-1, 1]
        self.connectionStrength = max(min(self.connectionStrength, 1.0), -1.0)

    def disconnect(self):
        # Remove this synapse from the presynaptic neuron's postsynaptic connections
        self.preSynapticNeuron.postsynapticConnections.remove(self)
        # Remove this synapse from the postsynaptic neuron's presynaptic connections
        self.postSynapticNeuron.presynapticConnections.remove(self)

    def reconnect_to(self, new_post_neuron):
        self.postSynapticNeuron = new_post_neuron
        # Add this synapse to the new postsynaptic neuron's presynaptic connections
        new_post_neuron.presynapticConnections.append(self)
        # Add this synapse back to the presynaptic neuron's postsynaptic connections
        self.preSynapticNeuron.postsynapticConnections.append(self)

    def __str__(self):
        return f"Synapse from {self.preSynapticNeuron.position} to {self.postSynapticNeuron.position} with strength {self.connectionStrength}"

class InputNeuron:
    def __init__(self, pos):
        self.position = pos
        self.neuroTransmitterInput = 0
        self.postsynapticConnections = []

class Neuron:
    def __init__(self, pos):
        self.position = pos
        self.neuroTransmitterInput = 0
        self.presynapticConnections = []
        self.postsynapticConnections = []

class OutputNeuron:
    def __init__(self, pos):
        self.position = pos
        self.neuroTransmitterInput = 0
        self.outputValue = 0
        self.presynapticConnections = []
#End Brain Game

def show_instructions(game_name):
    """Displays the instructions for the selected game."""
    if game_name == "Tic Tac Toe":
        print("\n--- Tic Tac Toe ---")
        print("Instructions:")
        print("1. You play as 'X' and the bot plays as 'O'.")
        print("2. The board is a 3x3 grid.")
        print("3. Enter your move as row and column indices (0, 1, or 2).")
        print("4. The first to get three in a row wins!")
    elif game_name == "Brain Game":
        print("\n--- Brain Game ---")
        print("Instructions:")
        print("1. Draw three numbers clicking enter each time to submit it")
        print("2. The third number should be the same number as one of the first two")
        print("3. Enter the numbers the first two drawings represent")
        print("4. The network will then probably unsuccsefuly predict what the number for the third drawing is.")
        print("5. Click and hold the left mouse button to draw and the right to erase.")
    elif game_name == "Pong Game":
        '''Instructions for pong game'''
        print("\n--- Pong Game  ---")
        print("Instructions:")
        print("1. This is a 2 player game.")
        print("2. The left paddle will be controlled by the keys 'W' and 'S'.")
        print("3. The right paddle will be controlled by the arrow keys.")
        print("4. The goal is to get the ball past the opposing paddle.")
        print("5. The first to score five points wins!")
#tic tac toe game code 
def tic_tac_toe():
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"
#print the board
    while True:
        print_board(board)

        if current_player == "X":
            while True:
                user_input = input("Enter your move (row and column) or 'exit' to quit: ")
                if user_input.lower() == 'exit':
                    print("Thanks for playing! Exiting the game.")
                    return  # Exit the game

                try:
                    row, col = map(int, user_input.split())
                    if board[row][col] != " ":
                        print("Invalid move! That spot is already taken. Try again.")
                        continue
                    break  # Valid move, exit the loop
                except (ValueError, IndexError):
                    print("Invalid input! Please enter row and column as two numbers (0, 1, or 2).")

        else:
            row, col = bot_move(board)
            print(f"Bot chooses: {row} {col}")

        board[row][col] = current_player
        winner = check_winner(board)
        if winner:
            print_board(board)
            print(f"{winner} wins!")
            break
        if check_draw(board):
            print_board(board)
            print("It's a draw!")
            break
        current_player = "O" if current_player == "X" else "X"
#creating the board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != " ":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != " ":
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return board[0][2]
    return None

def check_draw(board):
    return all(cell != " " for row in board for cell in row)

def bot_move(board):
    available_moves = [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]
    return random.choice(available_moves)

def pong_game():
    # Initialize Pygame
    pygame.init()

    # Screen dimensions
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pong")
    pygame.display.toggle_fullscreen

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # Paddle dimensions
    PADDLE_WIDTH = 10
    PADDLE_HEIGHT = 100
    BALL_SIZE = 20

    # Game variables
    player1_score = 0
    player2_score = 0
    WINNING_SCORE = 5

    # Paddle positions
    player1_y = (SCREEN_HEIGHT // 2) - (PADDLE_HEIGHT // 2)
    player2_y = (SCREEN_HEIGHT // 2) - (PADDLE_HEIGHT // 2)
    player1_x = 20
    player2_x = SCREEN_WIDTH - 20 - PADDLE_WIDTH

    # Ball position and speed
    ball_x = SCREEN_WIDTH // 2
    ball_y = SCREEN_HEIGHT // 2
    ball_speed_x = 4
    ball_speed_y = 4

    # Paddle speed
    paddle_speed = 6

    # Clock
    clock = pygame.time.Clock()

    # Fonts
    font = pygame.font.Font(None, 74)

    # Draw text
    def draw_text(text, font, color, x, y):
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (x, y))

    # Game loop
    running = True
    while running:
        screen.fill(BLACK)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                main_menu()
                return


        # Paddle movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and player1_y > 0:
            player1_y -= paddle_speed
        if keys[pygame.K_s] and player1_y < SCREEN_HEIGHT - PADDLE_HEIGHT:
            player1_y += paddle_speed
        if keys[pygame.K_UP] and player2_y > 0:
            player2_y -= paddle_speed
        if keys[pygame.K_DOWN] and player2_y < SCREEN_HEIGHT - PADDLE_HEIGHT:
            player2_y += paddle_speed

        # Ball movement
        ball_x += ball_speed_x
        ball_y += ball_speed_y

        # Ball collision with top and bottom walls
        if ball_y <= 0 or ball_y >= SCREEN_HEIGHT - BALL_SIZE:
            ball_speed_y *= -1

        # Ball collision with paddles
        if (player1_x < ball_x < player1_x + PADDLE_WIDTH and
            player1_y < ball_y < player1_y + PADDLE_HEIGHT) or \
           (player2_x < ball_x + BALL_SIZE < player2_x + PADDLE_WIDTH and
            player2_y < ball_y < player2_y + PADDLE_HEIGHT):
            ball_speed_x *= -1

        # Ball goes out of bounds
        if ball_x < 0:
            player2_score += 1
            ball_x, ball_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
            ball_speed_x *= -1
        if ball_x > SCREEN_WIDTH:
            player1_score += 1
            ball_x, ball_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
            ball_speed_x *= -1

        # Draw paddles and ball
        pygame.draw.rect(screen, WHITE, (player1_x, player1_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, WHITE, (player2_x, player2_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.ellipse(screen, WHITE, (ball_x, ball_y, BALL_SIZE, BALL_SIZE))
        pygame.draw.line(screen, WHITE, (SCREEN_WIDTH // 2, 0), (SCREEN_WIDTH // 2, SCREEN_HEIGHT))

        # Draw scores
        draw_text(str(player1_score), font, WHITE, SCREEN_WIDTH // 4, 20)
        draw_text(str(player2_score), font, WHITE, 3 * SCREEN_WIDTH // 4, 20)

        # Check for win
        if player1_score == WINNING_SCORE:
            draw_text("Player 1 Wins!", font, WHITE, SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
            pygame.display.flip()
            #pygame.time.wait(3000)
            running = False
            keepWindowOpen = True
        elif player2_score == WINNING_SCORE:
            draw_text("Player 2 Wins!", font, WHITE, SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
            pygame.display.flip()
            #pygame.time.wait(3000)
            running = False
            keepWindowOpen = True

        # Update display
        pygame.display.flip()
        clock.tick(60)

    while keepWindowOpen:
     # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                keepWindowOpen = False
    #Go back to main menu
    main_menu()


def brain_game():
    #intialize pygame
    pygame.init()

    #Create pygame screen
    screenWidth = 1000
    screenHeight = 1000
    screen = pygame.display.set_mode((screenWidth,screenHeight))
    pygame.display.set_caption("Digit Drawer")

    white = (255,255,255)
    black = (0,0,0)

    running = True

    screen.fill(white)

    clock = pygame.time.Clock()

    inputNum = 0

    input1 = [0,] * 100
    input2 = [0,] * 100
    input3 = [0,] * 100

    inputs = (input1, input2, input3)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break

            #Code to draw a pixel
            if pygame.mouse.get_pressed()[0]:
                mousePos = pygame.mouse.get_pos()

                xPixel = int(mousePos[0] / 100)
                yPixel = int(mousePos[1] / 100)

                pixel = yPixel * 10 + xPixel

                inputs[inputNum][pixel] = 1

                xPixel *= 100
                yPixel *= 100

                screen.fill(black, (xPixel,yPixel,100,100))

            #Code to erase a pixel
            elif pygame.mouse.get_pressed()[2]:
                mousePos = pygame.mouse.get_pos()

                xPixel = int(mousePos[0] / 100)
                yPixel = int(mousePos[1] / 100)

                pixel = yPixel * 10 + xPixel

                inputs[inputNum][pixel] = 0

                xPixel *= 100
                yPixel *= 100

                screen.fill(white, (xPixel,yPixel,100,100))

            if pygame.key.get_pressed()[pygame.K_RETURN]:
                if inputNum == 2:
                    pygame.quit()
                    running = False
                    break
                else:
                    screen.fill(white)
                    inputNum += 1

        if running:
            #pygame.display.update()
            pygame.display.flip()
            clock.tick(120)

    brainSize = (10, 10, 10)
    inputSize = (10, 10, 1)
    outputSize = (10, 1)
    # inputs = (
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    # Row 0
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    # Row 1
    #     0, 0, 0, 0, 1, 1, 0, 0, 0, 0,    # Row 2
    #     0, 0, 0, 0, 1, 1, 0, 0, 0, 0,    # Row 3
    #     0, 0, 0, 0, 1, 1, 0, 0, 0, 0,    # Row 4
    #     0, 0, 0, 0, 1, 1, 0, 0, 0, 0,    # Row 5
    #     0, 0, 0, 0, 1, 1, 0, 0, 0, 0,    # Row 6
    #     0, 0, 0, 0, 1, 1, 0, 0, 0, 0,    # Row 7
    #     0, 0, 0, 0, 1, 1, 0, 0, 0, 0,    # Row 8
    #     0, 0, 0, 1, 1, 1, 1, 0, 0, 0     # Row 9
    # )

    expectedOutput1 = [0.0,] * outputSize[0]
    expectedOutput2 = [0.0,] * outputSize[0]

    userInput1 = int(input("Enter the first number you drew: "))
    userInput2 = int(input("Enter the Second number you drew: "))

    while userInput1 < 0 or userInput1 > 9 or userInput2 < 0 or userInput2 > 9:
        print("Invalid Input!")
        userInput1 = int(input("Enter the first number you drew: "))
        userInput2 = int(input("Enter the Second number you drew: "))


    expectedOutput1[userInput1] = 1.0
    expectedOutput2[userInput2] = 1.0

    outputNull = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    neuralBrain = Brain(brainSize, inputSize, outputSize)

    neuralBrain.generateNeurons(300)

    # # Function call to build all synapses with the parameter in the format (input neuron num connections, neuron num connections)
    neuralBrain.buildAllConnections((5, 5))

    # neuralBrain.getStatus()

    for i in range(3):
        print(f"Iteration {i+1}")
        neuralBrain.run(tuple(inputs[0]), tuple(expectedOutput1))

    for i in range(3):
        print(f"Iteration {i+1}")
        neuralBrain.run(tuple(inputs[1]), tuple(expectedOutput2))

    for i in range(3):
        print(f"Iteration {i+1}")
        neuralBrain.run(tuple(inputs[0]), tuple(expectedOutput1))

    for i in range(3):
        print(f"Iteration {i+1}")
        neuralBrain.run(tuple(inputs[1]), tuple(expectedOutput2))

    for i in range(3):
        print(f"Iteration {i+1}")
        neuralBrain.run(tuple(inputs[0]), tuple(expectedOutput1))

    for i in range(3):
        print(f"Iteration {i+1}")
        neuralBrain.run(tuple(inputs[1]), tuple(expectedOutput2))


    neuralBrain.run(tuple(inputs[2]), outputNull)
    count = 0
    for pos in neuralBrain.outputNeurons:
        if round(neuralBrain.outputNeurons[pos].outputValue) == 1.0:
            print("========================")
            print("========================")
            print(f"The Predicted Number is Probably Not: {count}")
            print("========================")
            print("========================")
            return

        count += 1

    print("========================")
    print("========================")
    print(f"The Predicted Number is not working! But you got to draw and that is pretty much the game this is just a bonus! This was a prototype project!")
    print("========================")
    print("========================")
def main_menu():
    while True:
        print("\n--- Main Menu ---")
        print("1. Tic Tac Toe")
        print("2. Brain Game")
        print("3. Pong Game")
        print("4. Exit")

        choice = input("Select a game (1-4): ")
        if choice == "1":
            show_instructions("Tic Tac Toe")
            tic_tac_toe()
        elif choice == "2":
            show_instructions("Brain Game")
            brain_game()  # Call the Brain game function
        elif choice == "3":
            show_instructions("Pong Game")
            pong_game()   # Call the Pong game function
        elif choice == "4":
            print("Thanks for playing! Gig'em!")
            sys.exit()
        else:
            print("Invalid choice! Please select again.")

if __name__ == "__main__":
    main_menu()
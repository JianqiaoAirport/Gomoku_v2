# game config
PLANE_SIZE = 15

# data generation config
NUMBER_of_BATCHES_FOR_EACH_PROCESS = 6
MAX_SIMULATION_WHEN_GENERATING_DATA = 3
NUMBER_of_GAMES_IN_EACH_BATCH = 3
NUMBER_of_SAMPLES_IN_EACH_GAME = 1
GPU_WHEN_GENERATING_DATA = "3"

# training
NUMBER_of_UPDATE_NEURAL_NETWORK = 6
NUMBER_of_BATTLES_WHEN_EVALUATING = 5
MAX_SIMULATION_WHEN_EVALUATING = 3
MIN_BATCH = 12  # number_of_games_for_each_process*number_of_samples_in_each_game*8/min_batch 应该是整数
GPU_WHEN_TRAINING = "4"

# monte carlo tree search
C_PUCT = 5.0

# neural network
LEARNING_RATE = 0.0005
L2_REG = 0.005



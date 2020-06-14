import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# Determines as a whole what constants are being used

# Number of epochs for training
n_epochs = 20
# Whether cuda is available or not
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Genetic Alg Settings
popsize = 50
generations = 30

# Probabilities
# Seeds
seeded = False
if seeded:
    seed_no = 3
else:
    seed_no = None

# Objectives
obj = "memory"



# Prob to change act. function
act_prob = 0.2
# Prob to change number of nodes
node_prob = 0.8
# Prob to change number of layers
layers_prob = 0.05
# nodes/gauss to determine variance for changing number of nodes
# Min is 1, variance is nodes/var
nodes_var = 2

# Chosen hyper-params:
# activation_list = {0:F.leaky_relu, 1:F.sigmoid, 2: F.relu, 3: F.selu}
activation_list = {0: nn.LeakyReLU(), 1: nn.Sigmoid(), 2: nn.ReLU(), 3: nn.SELU()}

# Problem changes
current_problem = "hypercube"
if current_problem == "test":
    input_nodes = 4
    output_nodes = 1
    problem_type = "regression"
elif current_problem == "mnist":
    input_nodes = 784
    output_nodes = 10
    problem_type = "classification"
elif current_problem == "hypercube":
    input_nodes = 10
    output_nodes = 5
    problem_type = "classification"
elif current_problem == "moons":
    input_nodes = 2
    output_nodes = 2
    problem_type = "classification"
elif current_problem == "complex_reg":
    input_nodes = 8
    output_nodes = 1
    problem_type = "regression"

# Determines problem features of different test sets
problem_features = {"test":[4,1,"regression"], "hypercube":[10,5,"classification"],
                    "moons":[2,2,"classification"], "complex_reg":[8,1,"regression"]}

# For hyper-parameter testing
def testing_go():
    from Actual.geneticalg import test
    import time
    problems = ['test', 'hypercube', 'moons', 'complex_reg']
    activation_prob = [0.8, 0.6, 0.4, 0.2]
    for problem in problems:
        results = []
        start_time = time.time()
        for p in activation_prob:
            np.save('actualconstants', np.array([problem, p]), allow_pickle=True)
            res = test()
            results.append(res)
        print("Problem time taken is: ", time.time() - start_time)
        np.save('results{}'.format(problem), results, allow_pickle=True)




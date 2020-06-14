import pymoo
from pymoo.model.problem import Problem
import time
import numpy as np
import copy
import autograd.numpy as anp
import torch.nn.functional as F
import torch
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.rnsga2 import RNSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.model.sampling import Sampling
from pymoo.model.mutation import Mutation
from pymoo.model.crossover import Crossover
from pymoo.model.duplicate import ElementwiseDuplicateElimination, HashDuplicateElimination
from pymoo.model.callback import Callback
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.display import Display
from Actual import neuralmain
from Actual import training
from Actual import constants



class MyProblem(Problem):


    def make_tuple(self, li):
        return tuple(tuple([c[0], tuple(c[1])]) for c in li)

    def __init__(self,):
        super().__init__(n_var=1,
                         n_obj=2, n_constr=0, elementwise_evaluation=True)

    # Evaluates an individual
    def _evaluate(self, x, out, *args, **kwargs):
        # x[0] gets the individual, because it is an array of size 1

        x[0].model = x[0].create_model()
        f1 = self.run_model(x[0])
        # Occasionally returns nan
        if np.isnan(f1):
            print("We got one!")
            f1 = float("inf")

        if constants.obj == "memory":
            f2 = self.mem_size(x[0].model)
            # print(f"Layers are {x[0].layers} and mem is {f2} and FLOPs are {self.flops(x[0].layers)}")
        else:
            f2 = self.flops(x[0].layers)

        # f2 = self.mem_size(x[0].model)
        # f3 = self.flops(x[0].layers)

        # out["F"] = anp.column_stack([f1, f2])
        # out["F"] = anp.column_stack([f1, f2, f3])

    # Calculates flops
    # For fully connected layers, calc. FLOPS as (2I - 1)O
    # Where I is the input layer, and O the output layer
    def flops(self, x):
        flops = 0
        for layer in x:
            flops += (2*layer[1][0] - 1) * layer[1][1]
        return flops

    def mem_size(self, x):
        from Actual import pytorch_modelsize as siz
        # x = next(x.parameters())
        # size = sys.getsizeof(x)
        # size = siz.get_size(x.parameters())
        total_size = 0
        for i in x.parameters():
            for j in i:
                if j.size() != torch.Size([]):
                    for k in j:
                        total_size += siz.get_size(k)
                else:
                    total_size += siz.get_size(j)
        return total_size

    # Runs model
    def run_model(self, x):
        return training.modeltest(x.model)


class Individual():

    def __init__(self, layers=None):
        # no of layers and act. functions,
        # for each sample
        # first num is num of act functions, second is num of layers
        input_nodes = constants.input_nodes
        act_functions = constants.activation_list
        no_layers = np.random.randint(2, 4)

        if not layers:
            layers = []
            for x in range(no_layers):
                # Gets a random activation function
                ran = np.random.randint(0, len(act_functions))
                # act_function = act_functions[ran]
                nodes = self.make_nodes(layers, no_layers)
                # Make new layer
                # layers.append([act_function, nodes])
                layers.append([ran, nodes])


        self.layers = layers
        # learning rate
        self.lr = 1e-3
        # Calls create model
        # Makes model then returns it
        self.model = None
        # self.model = self.create_model()

    def make_nodes(self, layers, no_layers):
        # Get starting nodes
        start_nodes = constants.input_nodes
        # Has input and output nodes
        nodes = []
        # If first layer, get input nodes
        if not layers:
            input_nodes = start_nodes
        else:
            # Get output nodes from last layer
            input_nodes = layers[-1][1][1]
        # Check whether it's an output or not
        if len(layers) == no_layers - 1:
            output_nodes = constants.output_nodes
        else:
            # dist around half nodes for no. of nodes
            output_nodes = round(np.random.normal(start_nodes, start_nodes / 2))
            if output_nodes < 1:
                output_nodes = 1
        return [input_nodes, output_nodes]


    def create_model(self):
        return neuralmain.createModel(self.layers)



class MySampling(Sampling):

    def _do(self, problem, n_sample, **kwargs):
        # Creates number of samples
        samples = [[Individual()] for x in range(n_sample)]
        return samples

# Need to mutate individual layers
# Use gaussian for integers?
# For the moment weights and models are retrained every evaluation
# May change in future
class MyMutation(Mutation):
    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob

    # vars in [[ind],[ind] ... ] form
    def _do(self, problem, X, **kwargs):
        # loops through every var
        # return X
        for i in range(len(X)):
            # Assigns current variable to cur
            cur = X[i][0]
            # Goes through each layer for the individual
            layers = self.mutate_layers(cur.layers)
            X[i][0].layers = layers
            # Not sure if needed
            # X[i][0].model = X[i][0].create_model()
        return X

    def mutate_layers(self, layers):
        act_functions = constants.activation_list
        # Modifies activation functions
        for i in range(len(layers)):
            # chance for each layer to change the activation function
            ran = np.random.random()
            if ran < constants.act_prob:
                # layers[i][0] = act_functions[np.random.randint(0, len(act_functions))]
                layers[i][0] = np.random.randint(0, len(act_functions))

            # Chance to change number of nodes

        # Will modify if > 1 layers
        if len(layers) > 1:
            layers = self.mutate_nodes(layers)


        # chance to add or remove a layer
        ran = np.random.random()
        # Add a layer
        if constants.layers_prob/2 <= ran <= constants.layers_prob:
            # Gets a random activation function
            # act_function = act_functions[np.random.randint(0, len(act_functions))]
            act_function = np.random.randint(0, len(act_functions))
            # dist around half nodes for no. of nodes
            # Gets number of nodes at end, and uses that
            no_nodes = round(np.random.normal(layers[-1][1][0], layers[-1][1][0]/constants.nodes_var))
            if no_nodes < 1:
                no_nodes = 1
            # Make new layer
            layers.append([act_function, [no_nodes, constants.output_nodes]])
            # Make sure penultimate layer has right no. of output nodes
            layers[-2][1][1] = no_nodes
        # Remove a layer
        elif ran < constants.layers_prob/2:
            if len(layers) > 2:
                # Remove the last layer and change the output nodes
                layers.pop()
                # Output nodes of last layer
                layers[-1][1][1] = constants.output_nodes

        return layers

    # Number of nodes follows a normal distribution
    # Mean is current number, variance is number of nodes/10?
    def mutate_nodes(self, layers):
        for i in range(1, len(layers)):
            ran = np.random.random()
            if ran < constants.node_prob:
                # Gets input nodes of the layer
                nodes = layers[i][1][0]
                # loops until nodes != original
                original = nodes
                while nodes == original or nodes < 1:
                    nodes = round(np.random.normal(nodes, nodes/constants.nodes_var))
                    if nodes < 1:
                        nodes = 1
                # Changes output nodes of previous layer
                layers[i-1][1][1] = nodes
                # Changes input nodes of current layer
                layers[i][1][0] = nodes

        return layers


# Crossover bit strings
class MyCrossover(Crossover):

    def __init__(self):

        # Define the crossover: no. of parents + children
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        # Output with shape (n offsprings, n matings, n var)
        # Because there are equal numbers of parents and offsprings, shape of X is kept
        # Check copies
        # return X
        Y = np.full_like(X, None, dtype=np.object)

        _, n_matings, n_var = X.shape

        for k in range(n_matings):
            # Two parents
            Y[0, k, 0], Y[1, k, 0] = copy.deepcopy(X[0, k, 0]), copy.deepcopy(X[1, k, 0])
            Y[0, k, 0].layers, Y[1, k, 0].layers = self.pair_crossover(Y[0, k, 0].layers, Y[1, k, 0].layers)

        return Y

    # Crossover using activation strings
    def pair_crossover(self, layers1, layers2):
        split_point = np.random.randint(0, min(len(layers1), len(layers2)))
        if np.random.random() < 0.5:
            layers1, layers2 = layers1[:split_point] + layers2[split_point:], layers2[:split_point] + layers1[split_point:]
        else:
            layers1, layers2 = layers2[:split_point] + layers1[split_point:], layers1[:split_point] + layers2[split_point:]
        # Make sure that they are of correct format
        if len(layers1) == 1:
            # If just one layer, make sure that it's in and out correct
            layers1[0][1][0] = constants.input_nodes
            layers1[0][1][1] = constants.output_nodes
        else:
            for i in range(1,len(layers1)):
                layers1[i][1][0] = layers1[i - 1][1][1]

        if len(layers2) == 1:
            # If just one layer, make sure that it's in and out correct
            layers2[0][1][0] = constants.input_nodes
            layers2[0][1][1] = constants.output_nodes
        else:
            for i in range(1,len(layers2)):
                layers2[i][1][0] = layers2[i - 1][1][1]

        return layers1, layers2

class MyDisplay(Display):

    # Displays the progress of the program when it is running
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        smol = self.smallest_error(algorithm)
        self.output.append("Min error (err.)", smol[0])
        self.output.append("Min error (mem_size.)", smol[1])
        self.output.append("Min error (flops)", smol[2])
        self.output.append("Average F val, ", np.nanmean(algorithm.pop.get("F"),  axis=0))
        self.output.append("Smallest error solution ", self.smallest_error_model(algorithm))

# Gets smallest error value
    def smallest_error(self, algorithm):
        pop = algorithm.pop.get("F")
        return min(pop, key=lambda x:x[0])

# Gets smallest error model
    def smallest_error_model(self, algorithm):
        pop = algorithm.pop
        smallest = [float('inf'), 0]
        for x in pop:
            if x.F[0] < smallest[0]:
                smallest[0] = x.F[0]
                smallest[1] = x.X[0].layers
        return smallest[1]

# Generates a Pareto set of solutions
    def pareto_set(self, solutions):
        if len(solutions[0]) > 2:
            return False
        sols = sorted(solutions, key=lambda x: x[0])
        li = []
        smallest = float("inf")
        for i, x in enumerate(sols):
            if x[1] < smallest:
                li.append(x)
                smallest = x[1]
        return li

# Finds and deletes duplicates
class MyDuplicateElimination(ElementwiseDuplicateElimination):

    # Determines if two individuals are equal
    def is_equal(self, a, b):
        a = a.X[0]
        b = b.X[0]
        a_layers = a.layers
        b_layers = b.layers
        return a_layers == b_layers


# Runs the algorithm as a whole and returns the result
def run_alg():

    termination = get_termination("n_gen", constants.generations)

    algorithm = NSGA2(pop_size=constants.popsize,
                      sampling=MySampling(),
                      crossover=MyCrossover(),
                      mutation=MyMutation(),
                      display=MyDisplay(),
                      eliminate_duplicates=MyDuplicateElimination())

    result = minimize(MyProblem(),
                      algorithm,
                      termination,
                      # seed=constants.seed_no,
                      save_history=True,
                      verbose=True)

    return result

# Framework for running hyper-parameter tests
def test():
    import importlib
    importlib.reload(constants)
    res = []
    start_time = time.time()
    print("{} and {}".format(constants.current_problem, constants.act_prob))
    for i in range(0, 1):
        result = run_alg()
        res.append(result.F)
        print(time.time() - start_time, " seconds")
    return res


# if True:
#     start_time = time.time()
#     # #
#     # #
#     # # def runalg():
#     # ##
#     termination = get_termination("n_gen", constants.generations)
#
#     algorithm = NSGA2(pop_size=constants.popsize,
#                       sampling=MySampling(),
#                       crossover=MyCrossover(),
#                       mutation=MyMutation(),
#                       display=MyDisplay(),
#                       eliminate_duplicates=MyDuplicateElimination())
#
#     result = minimize(MyProblem(),
#                       algorithm,
#                       termination,
#                       # seed=constants.seed_no,
#                       save_history=True,
#                       verbose=True)
#
#
#     print(time.time() - start_time, " seconds")
#
#
#
#     print("")
#     # Currently x seems to be broken in terms of being the same as F
#     for individual in result.X:
#         print(individual[0].layers)
#         print(individual[0].model)
#
#     # print(mod)
#     print(sorted(result.F, key=lambda k: k[0]))
#
#     for x in result.pop:
#         cur = x.X[0]
#         print("Layers are: ", cur.layers)
#         print("Score is: ", x.F)
#         mp = MyProblem()
#         print("Flops should be: ", mp.flops(cur.layers))
#         print("")
#
#
#     # for x in result.history[00].pop:
#     #    ...:     cur = x.X[0]
#     #    ...:     print("Layers are: ", cur.layers)
#     #    ...:     print("Nodes are: ",cur.nodes)
#     #    ...:     print("Score is: ",x.F)
#     #    ...:     print("")
#
#
#
#     plot = Scatter()
#     plot.add(result.F)
#     plot.show()



# if __name__ == '__main__':
#     import time
#     start_time = time.time()
#     runalg()
#     print(time.time() - start_time, " seconds")

# Concerns - same training data for each cycle
# Ends up with same sorts of model
# Is it actually changing the models or just optimising what's already there

# yy = sorted([result.history[4].pop[i].F for i in range(50)], key = lambda k : k[0])

# Seed = 3, n_gen 5 to 6
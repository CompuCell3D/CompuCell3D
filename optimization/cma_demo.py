import cma
from cma import CMAEvolutionStrategy
from random import random



# def fcn(**args):
#     x = args[1]
#     y = args[2]
#     return x**2+(y-2)**2

# def fcn(x):
#     return (x-2)**2


def fcn(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2


# cma.fcts.elli
# optim = CMAEvolutionStrategy(9 * [0.5], 0.3)
# optim = CMAEvolutionStrategy(2 * [1.0], 20)
optim = CMAEvolutionStrategy(2 * [1.0], 5, {'bounds': [0, 10]})
# a new CMAEvolutionStrategy instance

# this loop resembles optimize()
while not optim.stop():  # iterate
    X = optim.ask()  # get candidate solutions
    f = [fcn(x) for x in X]  # evaluate solutions
    #  in case do something else that needs to be done
    optim.tell(X, f)  # do all the real "update" work
    optim.disp(20)  # display info every 20th iteration
    optim.logger.add()  # log another "data line"

print('termination by', optim.stop())
print('best f-value =', optim.result()[1])
print('best solution =', optim.result()[0])
# optim.logger.plot()  # if matplotlib is available

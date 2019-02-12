from RoadRunnerPy import RoadRunnerPy

model_file = 'oscli.sbml'
step_size = 0.02

initial_conditions = {}
initial_conditions['S1'] = 2.0
initial_conditions['S2'] = 3.0

rr = RoadRunnerPy(_path=model_file)
rr.stepSize = step_size
rr.loadSBML(_externalPath=model_file)

# setting initial conditions - this has to be done after loadingSBML
for name, value in initial_conditions.iteritems():
    try:  # have to catch exceptions in case initial conditions contain "unsettable" entries such as reaction rate etc...
        rr.model[name] = value
    except:
        pass

for i in range(20):
    rr.timestep()

rr_from = rr
path_from = rr_from.path
initial_conditions_from = rr_from.model

rr_to = RoadRunnerPy(_path=path_from)
rr_to.stepSize = step_size
rr_to.loadSBML(_externalPath=path_from)



print
# setting initial conditions - this has to be done after loadingSBML
for name, value in initial_conditions_from.iteritems():
    try:
        rr_to.model[name] = value
    except:
        pass




from RoadRunnerPy import RoadRunnerPy

# setting stepSize

# loading SBML and LLVM-ing it


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


# from roadrunner import RoadRunner
rr_to_1 = RoadRunnerPy(sbml=rr_from.getCurrentSBML())
rr_to_1.stepSize = step_size
print

for i in range(20):
    rr_from.timestep()
    rr_to_1.timestep()
    print 'STEP=',i
    print 'orig S1=',rr_from.model['S1'],' new S1=',rr_to_1.model['S1']
    print 'orig S2=', rr_from.model['S2'], ' new S2=', rr_to_1.model['S2']

rr_to = RoadRunnerPy(_path=path_from)
rr_to.stepSize = step_size
rr_to.loadSBML(_externalPath=path_from)

# # setting initial conditions - this has to be done after loadingSBML
# for name, value in initial_conditions_from.iteritems():
#     if name.startswith('init('): continue
#     try:  # have to catch exceptions in case initial conditions contain "unsettable" entries such as reaction rate etc...
#         rr_to.model[name] = value
#     except:
#         pass
#


independent_floating_species = rr.getIndependentFloatingSpeciesIds()

print independent_floating_species

boundary_species = rr.getIndependentFloatingSpeciesIds()
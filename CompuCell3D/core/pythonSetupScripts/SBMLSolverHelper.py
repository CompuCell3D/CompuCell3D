import os
import sys

# this function will replace all API functions which refer to SBMLSolver in the event that no SBMLSolver event is installed
def SBMLSolverError(self, *args, **kwrds):
    import inspect
    line = inspect.stack()[1][2]
    call = inspect.stack()[1][4]
    raise AttributeError('SBMLSolverError line :' + str(line) + ' call:' + str(
        call) + ' Trying to access one of the SBML solver methods but SBMLSolver engine (e.g. RoadRunner) has not been installed with your CompuCell3D package')


class SBMLSolverHelper(object):
    @classmethod
    def removeAttribute(cls, name):
        print 'cls=', cls
        return delattr(cls, name)

    def __init__(self):

        # in case user passes simulate options we set the here 
        # this dictionary translates old options valid for earlier rr versions to new ones
        self.option_name_dict = {
            'relative': 'relative_tolerance',
            'absolute': 'absolute_tolerance',
            'steps': 'maximum_num_steps'
        }

        try:

            import roadrunner

        except ImportError, e:
            # replacing SBMLSolver API with wrror messages
            # delattr(SteppableBasePy, 'addSBMLToCell')
            SBMLSolverAPI = ['getSBMLGlobalOptions', 'setSBMLGlobalOptions', 'addSBMLToCell', 'addSBMLToCellTypes',
                             'addSBMLToCellIds', 'addFreeFloatingSBML',
                             'deleteSBMLFromCellIds', 'deleteSBMLFromCellTypes', 'deleteSBMLFromCell',
                             'timestepCellSBML', 'timestepFreeFloatingSBML', 'timestepSBML',
                             'setStepSizeForCell', 'setStepSizeForCellIds', 'setStepSizeForCellTypes',
                             'setStepSizeForFreeFloatingSBML',
                             'getSBMLSimulator', 'getSBMLState', 'setSBMLState', 'getSBMLValue', 'setSBMLValue',
                             'normalizePath', 'copySBMLs']

            import types
            for apiName in SBMLSolverAPI:
                SBMLSolverHelper.removeAttribute(apiName)
                setattr(SBMLSolverHelper, apiName, types.MethodType(SBMLSolverError, SBMLSolverHelper))

    def addSBMLToCell(self, _modelFile='', _modelName='', _cell=None, _stepSize=1.0, _initialConditions={},
                      _coreModelName='', _modelPathNormalized='', _options=None, _currentStateSBML=None):

        """
        Attaches RoadRunner SBML solver to a particular cell. The sbml solver is stored as an element
        of the cell's dictionary - cell.dict['SBMLSolver'][_modelName]. The function has a dual operation mode. 
        When user provides _currentStateSBML, _cell _modelName, _stepSize the addSBMLToCell function creates a clone 
        of a solver whose state is described by the _currentStateSBML . If _currentStateSBML is None then the new SBML solver 
        is being created,  SBML file (_modelFile) loaded and initial conditions are applied. It is important to always set 
        _stepSize to make sure that after calling timestep() fcn the solver advances appropriate delta time

        :param _modelFile {str}: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path 
        :param _modelName {str}: name of the model - this is a label used to store mode in the cell.dict['SBMLSolver'] dictionary  
        :param _cell {CellG object}: cc3d cell object 
        :param _stepSize {float}: time step- determines how much in "real" time units timestep() fcn advances SBML solver 
        :param _initialConditions {dict}: initial conditions dictionary
        :param _coreModelName {str}: deprecated, kept for backward compatibility reasons 
        :param _modelPathNormalized {str}: deprecated, kept for backward compatibility reasons: 
        :param _options {dict}: dictionary that currently only defines what type of ODE solver to choose. In the newer versions of RR this might be not necessary. The keys that are supported are the following: 
             

        absolute - determines absolute tolerance default 1e-10
        relative - determines relative tolerance default 1e-5
        stiff - determines if using stiff solver or not default False
        
        :param _currentStateSBML {str}: string representation  of the SBML representing current state of the solver. 
         
        :return: None 
        """

        import CompuCell

        coreModelName = _modelName
        if coreModelName == '':
            coreModelName, ext = os.path.splitext(os.path.basename(_modelFile))

        modelPathNormalized = self.normalizePath(_modelFile)

        dict_attrib = CompuCell.getPyAttrib(_cell)

        sbmlDict = {}
        if dict_attrib.has_key('SBMLSolver'):
            sbmlDict = dict_attrib['SBMLSolver']
        else:
            dict_attrib['SBMLSolver'] = sbmlDict

        from RoadRunnerPy import RoadRunnerPy
        if _currentStateSBML is None:
            rr = RoadRunnerPy(_path=_modelFile)
            # setting stepSize
            rr.stepSize = _stepSize
            # loading SBML and LLVM-ing it
            rr.loadSBML(_externalPath=modelPathNormalized)

        else:
            rr = RoadRunnerPy(sbml=_currentStateSBML)
            # setting stepSize
            rr.stepSize = _stepSize

        # storing rr instance in the cell dictionary
        sbmlDict[coreModelName] = rr

        # setting initial conditions - this has to be done after loadingSBML
        for name, value in _initialConditions.iteritems():
            try:  # have to catch exceptions in case initial conditions contain "unsettable" entries such as reaction rate etc...
                rr.model[name] = value
            except:
                pass
                # we are turning off dynamic python properties because rr is not used in the interactive mode.
                # rr.options.disablePythonDynamicProperties = True

        # setting output results array size 
        rr.selections = []  # by default we do not request any output array at each intergration step

        if _options:
            for name, value in _options.iteritems():

                try:
                    setattr(rr.getIntegrator(), name, value)
                except AttributeError:
                    setattr(rr.getIntegrator(), self.option_name_dict[name], value)
        else:
            
            # check for global options
            
            globalOptions = self.getSBMLGlobalOptions()
            if globalOptions:
                for name, value in globalOptions.iteritems():
                    try:
                        setattr(rr.getIntegrator(), name, value)
                    except (AttributeError, ValueError) as e:
                        setattr(rr.getIntegrator(), self.option_name_dict[name], value)
                        # setattr(rr.simulateOptions,name,value)


    def getSBMLGlobalOptions(self):
        """
        returns global options for the SBML solver - deprecated as newer version of CC3D
        :return {dict}: global SBML solver options
        """
        import CompuCellSetup
        return CompuCellSetup.globalSBMLSimulatorOptions

    def setSBMLGlobalOptions(self, _options):
        """
        Deprecated  - sets global SBML options
        :param _options {dictionary}:
        :return: None
        """
        import CompuCellSetup
        CompuCellSetup.globalSBMLSimulatorOptions = _options

    def addSBMLToCellTypes(self, _modelFile='', _modelName='', _types=[], _stepSize=1.0, _initialConditions={},
                           _options={}):
        """
        Adds SBML Solver to all cells of given cell type - internally it calls addSBMLToCell(fcn).
        Used during initialization of the simulation. It is important to always set
        _stepSize to make sure that after calling timestep() fcn the solver advances appropriate delta time

        :param _modelFile {str}: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param _modelName {str}: name of the model - this is a label used to store mode in the cell.dict['SBMLSolver'] dictionary
        :param _types {list of integers}: list of cell types
        :param _stepSize {float}: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param _initialConditions {dict}: initial conditions dictionary
        :param _options {dict}: dictionary that currently only defines what type of ODE solver to choose.
        In the newer versions of RR this might be not necessary. The keys that are supported are the following:

        absolute - determines absolute tolerance default 1e-10
        relative - determines relative tolerance default 1e-5
        stiff - determines if using stiff solver or not default False

        :return: None
        """

        if 'steps' in _options.keys():
            print '-----------WARNING-----------------\n\n steps option for SBML solver is deprecated.'

        coreModelName = _modelName
        if coreModelName == '':
            coreModelName, ext = os.path.splitext(os.path.basename(_modelFile))

        modelPathNormalized = self.normalizePath(_modelFile)

        for cell in self.cellListByType(*_types):
            self.addSBMLToCell(_modelFile=_modelFile, _modelName=_modelName, _cell=cell, _stepSize=_stepSize,
                               _initialConditions=_initialConditions, _coreModelName=coreModelName,
                               _modelPathNormalized=modelPathNormalized, _options=_options)

    def addSBMLToCellIds(self, _modelFile, _modelName='', _ids=[], _stepSize=1.0, _initialConditions={}, _options={}):
        """
        Adds SBML Solver to all cells with give ids - internally it calls addSBMLToCell(fcn).
        Used during initialization of the simulation. It is important to always set
        _stepSize to make sure that after calling timestep() fcn the solver advances appropriate delta time

        :param _modelFile {str}: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param _modelName {str}: name of the model - this is a label used to store mode in the cell.dict['SBMLSolver'] dictionary

        :param _ids {list}: list of cell ids that will get new SBML Sovler
        :param _stepSize {float}: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param _initialConditions {dict}: initial conditions dictionary
        :param _options {dict}: dictionary that currently only defines what type of ODE solver to choose.
        In the newer versions of RR this might be not necessary. The keys that are supported are the following:

        absolute - determines absolute tolerance default 1e-10
        relative - determines relative tolerance default 1e-5
        stiff - determines if using stiff solver or not default False

        :return: None
        """

        if 'steps' in _options.keys():
            print '-----------WARNING-----------------\n\n steps option for SBML solver is deprecated.'

        coreModelName = _modelName
        if coreModelName == '':
            coreModelName, ext = os.path.splitext(os.path.basename(_modelFile))

        modelPathNormalized = self.normalizePath(_modelFile)
        # print 'will try to add SBML to ids=',_ids
        for id in _ids:
            cell = self.inventory.attemptFetchingCellById(id)
            # print 'THIS IS CELL ID=',cell.id
            if not cell:
                continue

            self.addSBMLToCell(_modelFile=_modelFile, _modelName=_modelName, _cell=cell, _stepSize=_stepSize,
                               _initialConditions=_initialConditions, _coreModelName=coreModelName,
                               _modelPathNormalized=modelPathNormalized, _options=_options)

    def addFreeFloatingSBML(self, _modelFile, _modelName, _stepSize=1.0, _initialConditions={}, _options={}):
        """
        Adds free floating SBML model - not attached to any cell. The model will be identified/referenced by the _modelName

        :param _modelFile {str}: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param _modelName {str}: name of the model - this is a label used to store mode in the cell.dict['SBMLSolver'] dictionary

        :param _stepSize {float}: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param _initialConditions {dict}: initial conditions dictionary
        :param _options {dict}: dictionary that currently only defines what type of ODE solver to choose.
        In the newer versions of RR this might be not necessary. The keys that are supported are the following:

        absolute - determines absolute tolerance default 1e-10
        relative - determines relative tolerance default 1e-5
        stiff - determines if using stiff solver or not default False

        :return: None
        """

        modelPathNormalized = self.normalizePath(_modelFile)
        try:
            f = open(modelPathNormalized, 'r')
            f.close()
        except IOError, e:
            if self.simulator.getBasePath() != '':
                modelPathNormalized = os.path.abspath(os.path.join(self.simulator.getBasePath(), modelPathNormalized))

        from RoadRunnerPy import RoadRunnerPy
        rr = RoadRunnerPy(_path=_modelFile)
        rr.loadSBML(_externalPath=modelPathNormalized)

        # setting stepSize
        rr.stepSize = _stepSize

        # storing
        import CompuCellSetup
        CompuCellSetup.freeFloatingSBMLSimulator[_modelName] = rr

        # setting initial conditions - this has to be done after loadingSBML
        for name, value in _initialConditions.iteritems():
            rr.model[name] = value
            # rr.options.disablePythonDynamicProperties = True

        # setting output results array size 
        rr.selections = []  # by default we do not request any output array at each intergration step

        # in case user passes simulate options we set the here        
        if _options:
            for name, value in _options.iteritems():

                try:
                    setattr(rr.getIntegrator(), name, value)
                except (AttributeError, ValueError) as e:
                    setattr(rr.getIntegrator(), self.option_name_dict[name], value)
        else:
            # check for global options

            globalOptions = self.getSBMLGlobalOptions()
            if globalOptions:
                for name, value in globalOptions.iteritems():
                    # print ' 2 name , value=',(name , value)
                    # print 'name=',name,' value=',value
                    # if name=='steps':
                    # continue
                    try:
                        setattr(rr.getIntegrator(), name, value)
                    except (AttributeError, ValueError) as e:
                        setattr(rr.getIntegrator(), self.option_name_dict[name], value)
                        # setattr(rr.simulateOptions,name,value)

                        # if _options:
                        # for name , value in _options.iteritems():
                        # setattr(rr.simulateOptions,name,value)
                        # else: # check for global options
                        # globalOptions=self.getSBMLGlobalOptions()
                        # if globalOptions:
                        # for name , value in globalOptions.iteritems():
                        # setattr(rr.simulateOptions,name,value)

    def deleteSBMLFromCellIds(self, _modelName, _ids=[]):
        """
        Deletes  SBML model from cells whose ids match those stered int he _ids list
        :param _modelName {str}: model name
        :param _ids {list}: list of cell ids
        :return: None
        """
        import CompuCell
        for id in _ids:
            cell = self.inventory.attemptFetchingCellById(id)
            if not cell:
                continue

            dict_attrib = CompuCell.getPyAttrib(cell)
            try:
                sbmlDict = dict_attrib['SBMLSolver']
                del sbmlDict[_modelName]
            except LookupError, e:
                pass

    def deleteSBMLFromCellTypes(self, _modelName, _types=[]):
        """
        Deletes  SBML model from cells whose type match those stered int he _ids list
        :param _modelName {str}: model name
        :param _types: list of cell cell types
        :return: None

        """
        import CompuCell
        for cell in self.cellListByType(*_types):
            dict_attrib = CompuCell.getPyAttrib(cell)
            try:
                sbmlDict = dict_attrib['SBMLSolver']
                del sbmlDict[_modelName]
            except LookupError, e:
                pass

    def deleteSBMLFromCell(self, _modelName='', _cell=None):
        """
        Deletes SBML from a particular cell
        :param _modelName {str}: model name
        :param _cell {obj}: CellG cell obj
        :return: None
        """
        import CompuCell
        dict_attrib = CompuCell.getPyAttrib(_cell)
        try:
            sbmlDict = dict_attrib['SBMLSolver']
            del sbmlDict[_modelName]
        except LookupError, e:
            pass

    def deleteFreeFloatingSBML(self, _modelName):
        """
        Deletes free floatfin SBLM model
        :param _modelName {str}: model name
        :return: None
        """

        import CompuCellSetup
        try:
            del CompuCellSetup.freeFloatingSBMLSimulator[_modelName]
        except LookupError, e:
            pass

    def timestepCellSBML(self):
        """
        advances (integrats forward) models stored as attributes of cells
        :return: None
        """
        import CompuCell

        # timestepping SBML attached to cells
        for cell in self.cellList:
            dict_attrib = CompuCell.getPyAttrib(cell)
            if dict_attrib.has_key('SBMLSolver'):
                sbmlDict = dict_attrib['SBMLSolver']

                for modelName, rrTmp in sbmlDict.iteritems():
                    rrTmp.timestep()  # integrating SBML

    def setStepSizeForCell(self, _modelName='', _cell=None, _stepSize=1.0):
        """
        Sets integration step size for SBML model attached to _cell

        :param _modelName {str}: model name
        :param _cell {object}: CellG cell object
        :param _stepSize {float}: integrtion step size
        :return: None
        """
        import CompuCell
        dict_attrib = CompuCell.getPyAttrib(_cell)

        try:
            sbmlSolver = dict_attrib['SBMLSolver'][_modelName]
        except LookupError, e:
            return

        sbmlSolver.stepSize = _stepSize

    def setStepSizeForCellIds(self, _modelName='', _ids=[], _stepSize=1.0):
        """
        Sets integration step size for SBML model attached to cells of given ids

        :param _modelName {str}: model name
        :param _ids {list}: list of cell ids
        :param _stepSize {float}: integrtion step size
        :return: None
        """
        for id in _ids:
            cell = self.inventory.attemptFetchingCellById(id)
            if not cell:
                continue
            self.setStepSizeForCell(_modelName=_modelName, _cell=cell, _stepSize=_stepSize)

    def setStepSizeForCellTypes(self, _modelName='', _types=[], _stepSize=1.0):
        """
        Sets integration step size for SBML model attached to cells of given ids

        :param _modelName {str}: model name
        :param _types {list}: list of cell types
        :param _stepSize {float}: integrtion step size
        :return: None
        """
        for cell in self.cellListByType(*_types):
            self.setStepSizeForCell(_modelName=_modelName, _cell=cell, _stepSize=_stepSize)

    def setStepSizeForFreeFloatingSBML(self, _modelName='', _stepSize=1.0):
        """
        Sets integration step size for free floating SBML
        :param _modelName {str}: model name
        :param _stepSize {float}: integration time step
        :return: None
        """
        try:
            import CompuCellSetup
            sbmlSolver = CompuCellSetup.freeFloatingSBMLSimulator[_modelName]
        except LookupError, e:
            return

        sbmlSolver.stepSize = _stepSize

    def timestepFreeFloatingSBML(self):
        """
        Integrates forward all free floating SBML solvers
        :return: None
        """
        import CompuCellSetup

        for modelName, rr in CompuCellSetup.freeFloatingSBMLSimulator.iteritems():
            rr.timestep()

    def timestepSBML(self):
        """
        Integrates forward all free floating SBML solvers and all sbmlsolvers attached to cells
        :return: None
        """
        self.timestepCellSBML()
        self.timestepFreeFloatingSBML()

    def getSBMLSimulator(self, _modelName, _cell=None):
        """
        Returns a reference to RoadRunnerPy or None
        :param _modelName {str}: model name
        :param _cell {object}: CellG cell object
        :return {instance of RoadRunnerPy} or {None}:
        """

        import CompuCell
        import CompuCellSetup
        if not _cell:
            try:

                return CompuCellSetup.freeFloatingSBMLSimulator[_modelName]

            except LookupError, e:
                return None
        else:
            try:
                dict_attrib = CompuCell.getPyAttrib(_cell)
                return dict_attrib['SBMLSolver'][_modelName]
            except LookupError, e:
                return None

    def getSBMLState(self, _modelName, _cell=None):
        """
        Returns dictionary-like object representing state of the SBML solver - instance of the RoadRunner.model
        which behaves as a python dictionary but has many entries some of which are non-assignable /non-mutable

        :param _modelName {str}: model name
        :param _cell {object}: CellG object
        :return {instance of RoadRunner.model}: dict-like object
        """
        # might use roadrunner.SelectionRecord.STATE_VECTOR to limit dictionary iterations to only valuses which are settable
        # for now, though, we return full rr.model dictionary-like object

        # return dict(sbmlSimulator.model.items(roadrunner.SelectionRecord.STATE_VECTOR))
        sbmlSimulator = self.getSBMLSimulator(_modelName, _cell)
        try:
            return sbmlSimulator.model
        except:
            if _cell:
                raise RuntimeError("Could not find model " + _modelName + ' attached to cell.id=', _cell.id)
            else:
                raise RuntimeError("Could not find model " + _modelName + ' in the list of free floating SBML models')

    def getSBMLStateAsPythonDict(self, _modelName, _cell=None):
        """
        Returns Python dictionary representing state of the SBML solver

        :param _modelName {str}: model name
        :param _cell {object}: CellG object
        :return {dict}: dictionary representing state of the SBML Solver
        """
        return self.getSBMLState(_modelName, _cell)

    def setSBMLState(self, _modelName, _cell=None, _state={}):
        """
        Sets SBML state for the solver - only for advanced uses. Requires detailed knowledge of how underlying
        SBML solver (roadrunner) works
        :param _modelName {str}: model name
        :param _cell {object}: CellG object
        :param _state {dict}: dictionary with state variables to set
        :return: None
        """

        sbmlSimulator = self.getSBMLSimulator(_modelName, _cell)

        if not sbmlSimulator:
            return False
        else:

            if _state == sbmlSimulator.model:  # no need to do anything when all the state changes are done on model
                return True

            for name, value in _state.iteritems():
                try:
                    sbmlSimulator.model[name] = value
                except:  # in case user decides to set unsettable quantities e.g. reaction rates
                    pass

            return True

    def getSBMLValue(self, _modelName, _valueName='', _cell=None):
        """
        Retrieves value of the SBML state variable
        :param _modelName {str}: model name
        :param _valueName {str}: name of the state variable
        :param _cell {object}: CellG object
        :return {float}: value of the state variable
        """
        sbmlSimulator = self.getSBMLSimulator(_modelName, _cell)
        if not sbmlSimulator:
            if _cell:
                raise RuntimeError("Could not find model " + _modelName + ' attached to cell.id=', _cell.id)
            else:
                raise RuntimeError("Could not find model " + _modelName + ' in the list of free floating SBML models')
        else:
            return sbmlSimulator[_valueName]

    def setSBMLValue(self, _modelName, _valueName='', _value=0.0, _cell=None):
        """
        Sets SBML solver state variable
        :param _modelName {str}: model name
        :param _valueName {str}: name of the stae variable
        :param _value {float}: value of the state variable
        :param _cell {object}: CellG object
        :return: None
        """
        sbmlSimulator = self.getSBMLSimulator(_modelName, _cell)
        if not sbmlSimulator:
            return False
        else:
            sbmlSimulator.model[_valueName] = _value
            return True

    def copySBMLs(self, _fromCell, _toCell, _sbmlNames=[], _options=None):
        """
        Copies SBML solvers (with their states - effectively clones the solver) from one cell to another
        :param _fromCell {object}: source CellG cell
        :param _toCell {object}: target CellG cell
        :param _sbmlNames: list of SBML model name whose solver are to be copied
        :param _options {dict}: - deprecated - list of SBML solver options
        :return: None
        """
        sbmlNamesToCopy = []
        import CompuCell
        if not (len(_sbmlNames)):
            # if user does not specify _sbmlNames we copy all SBML networks
            try:
                dict_attrib = CompuCell.getPyAttrib(_fromCell)
                sbmlDict = dict_attrib['SBMLSolver']
                sbmlNamesToCopy = sbmlDict.keys()
            except LookupError, e:
                pass
        else:
            sbmlNamesToCopy = _sbmlNames

        try:
            dict_attrib_from = CompuCell.getPyAttrib(_fromCell)
            sbmlDictFrom = dict_attrib_from['SBMLSolver']
        except LookupError, e:
            return

        try:
            dict_attrib_to = CompuCell.getPyAttrib(_toCell)
            sbmlDictTo = dict_attrib_to['SBMLSolver']
        except LookupError, e:
            # if _toCell does not have SBMLSolver dictionary entry we simply add it
            dict_attrib_to['SBMLSolver'] = {}
            sbmlDictTo = dict_attrib_to['SBMLSolver']

        for sbmlName in sbmlNamesToCopy:

            rrFrom = sbmlDictFrom[sbmlName]
            currentStateSBML = sbmlDictFrom[sbmlName].getCurrentSBML()
            self.addSBMLToCell(_modelName=sbmlName, _cell=_toCell, _stepSize=rrFrom.stepSize, _options=_options,
                               _currentStateSBML=currentStateSBML)


    def normalizePath(self, _path):
        """
        Checks if file exists and if not it joins basepath (path to the root of the cc3d project) with path
        :param _path {str}: relative path to CC3D resource
        :return {str}: absolute path to CC3D resource
        """

        pathNormalized = _path
        try:
            f = open(pathNormalized, 'r')
            f.close()
        except IOError, e:
            if self.simulator.getBasePath() != '':
                pathNormalized = os.path.abspath(os.path.join(self.simulator.getBasePath(), pathNormalized))

        return pathNormalized

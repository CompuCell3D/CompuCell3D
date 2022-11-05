import time
from cc3d.core.SteppablePy import SteppablePy
from cc3d import CompuCellSetup


class SteppableRegistry(SteppablePy):
    def __init__(self):
        super(SteppableRegistry, self).__init__()
        self.steppableList = []
        self.runBeforeMCSSteppableList = []
        self.steppableDict = {}  # {steppableClassName:[steppable inst0,steppable inst1,...]}
        self._simulator = None
        from collections import defaultdict
        self.profiler_dict = defaultdict(lambda: defaultdict(float))  # {steppable_class_name:{object_hash:runtime}}

    @property
    def simulator(self):
        return self._simulator

    @simulator.setter
    def simulator(self,simulator):
        self._simulator = simulator

    def set_sim(self,sim):
        """
        sets simulato object
        :param sim: {Simulator}
        :return: None
        """
        self.sim = sim

    def get_profiler_report(self):

        profiler_report = []

        for steppable_name, steppable_obj_dict in self.profiler_dict.items():
            for steppable_obj_hash, run_time in steppable_obj_dict.items():
                profiler_report.append([steppable_name, str(steppable_obj_hash), run_time])

        return profiler_report

    def allSteppables(self):
        for steppable in self.steppableList:
            yield steppable
        for steppable in self.runBeforeMCSSteppableList:
            yield steppable

    def core_init(self):
        """

        :return:
        """

        for steppable in self.runBeforeMCSSteppableList:
            steppable.simulator = self.simulator
            steppable.core_init()

        for steppable in self.steppableList:
            steppable.simulator = self.simulator
            steppable.core_init()

    def registerSteppable(self, _steppable):

        try:
            if _steppable.runBeforeMCS:
                self.runBeforeMCSSteppableList.append(_steppable)
            else:
                self.steppableList.append(_steppable)
        except AttributeError:
            self.steppableList.append(_steppable)

        # storing steppable in the dictionary
        try:
            self.steppableDict[_steppable.__class__.__name__].append(_steppable)
        except LookupError as e:
            self.steppableDict[_steppable.__class__.__name__] = [_steppable]

    def getSteppablesByClassName(self, _className):
        try:
            return self.steppableDict[_className]
        except LookupError as e:
            return []

    def init(self, _simulator):
        for steppable in self.runBeforeMCSSteppableList:
            steppable.init(_simulator)

        for steppable in self.steppableList:
            steppable.init(_simulator)

    def extraInit(self, _simulator):
        for steppable in self.runBeforeMCSSteppableList:
            steppable.extraInit(_simulator)

        for steppable in self.steppableList:
            steppable.extraInit(_simulator)

    def restart_steering_panel(self):
        """
        Function used during restart only to bring up the steering panel
        :return: None
        """
        pg = CompuCellSetup.persistent_globals
        try:
            if len(list(pg.steering_param_dict.keys())):
                pg.add_steering_panel(panel_data=list(pg.steering_param_dict.values()))
        except:
            print('Could not create steering panel')

    def start(self):
        pg = CompuCellSetup.persistent_globals
        for steppable in self.runBeforeMCSSteppableList:
            steppable.start()
            # handling steering panel
            steppable.add_steering_panel()
            if hasattr(steppable, 'initialize_automatic_tasks'):
                steppable.initialize_automatic_tasks()

            if hasattr(steppable, 'perform_automatic_tasks'):
                steppable.perform_automatic_tasks()

        for steppable in self.steppableList:
            steppable.start()
            # handling steering panel
            steppable.add_steering_panel()
            if hasattr(steppable, 'initialize_automatic_tasks'):
                steppable.initialize_automatic_tasks()
            if hasattr(steppable, 'perform_automatic_tasks'):
                steppable.perform_automatic_tasks()

        # handling steering panel
        try:
            if len(list(pg.steering_param_dict.keys())):
                pg.add_steering_panel(panel_data=list(pg.steering_param_dict.values()))
        except:
            print('Could not create steering panel')

    def step(self, _mcs):

        for steppable in self.steppableList:

            # this executes given steppable every "frequency" Monte Carlo Steps
            if not _mcs % steppable.frequency:

                try:
                    steppable.mcs = _mcs
                except AttributeError:
                    pass

                begin = time.time()

                steppable.step(_mcs)
                if hasattr(steppable, 'perform_automatic_tasks'):
                    steppable.perform_automatic_tasks()

                end = time.time()

                self.profiler_dict[steppable.__class__.__name__][hex(id(steppable))] += (end - begin) * 1000

                steppable.process_steering_panel_data_wrapper()

        # we reset dirty flag (indicates that a parameter was changed)
        # of the steering parameters via the call to the SteppableBasePy "set_steering_param_dirty"
        # function. At the moment we do not implement fine control over which parameters are dirty. For now if
        # any of the parameters is dirty we will rerun all the code that handles steering. this means
        # that we will process every single steering parameters as if it was dirty (i.e. changed recently)

        steppable_list_aggregate = self.steppableList + self.runBeforeMCSSteppableList
        # setting steering parameter dirty flag to False - this code runs at the end of the steppables call
        if len(steppable_list_aggregate):
            steppable_list_aggregate[0].set_steering_param_dirty(flag=False)

    def stepRunBeforeMCSSteppables(self, _mcs):

        for steppable in self.runBeforeMCSSteppableList:
            if not _mcs % steppable.frequency:  # this executes given steppable every "frequency" Monte Carlo Steps
                begin = time.time()

                steppable.step(_mcs)

                end = time.time()

                self.profiler_dict[steppable.__class__.__name__][hex(id(steppable))] += (end - begin) * 1000
                steppable.process_steering_panel_data_wrapper()

    def finish(self):
        for steppable in self.runBeforeMCSSteppableList:
            steppable.finish()

        for steppable in self.steppableList:
            steppable.finish()

    def on_stop(self):
        for steppable in self.runBeforeMCSSteppableList:
            steppable.on_stop()

        for steppable in self.steppableList:
            steppable.on_stop()

    def cleanup(self):
        print('inside cleanup')
        for steppable in self.runBeforeMCSSteppableList:
            steppable.cleanup()

        for steppable in self.steppableList:
            steppable.cleanup()

        self.clean_after_simulation()

    def clean_after_simulation(self):
        self.steppableList = []
        self.runBeforeMCSSteppableList = []
        self.steppableDict = {}

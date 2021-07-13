# -*- coding: utf-8 -*-
"""
This module handles  reading of serialized simulation output. It has
API of Simulation Thread and from the point of view of player
it behaves as a "regular" simulation with one exception that
instead running actual simulation during call to the step function it reads previously
saves simulation snapshot (currently stored as vtk file)
"""

from cc3d.core.CMLResultsReader import CMLResultReader as CMLResultsReaderBase
from cc3d.core.enums import *
from PyQt5.QtCore import *
from cc3d.cpp import CompuCell
from cc3d.cpp import PlayerPython
from cc3d.core import XMLUtils
import re
from cc3d.core.XMLUtils import CC3DXMLListPy
from cc3d.cpp import CC3DXML
import cc3d.CompuCellSetup as CompuCellSetup
import vtk
import os
import os.path
from . import SimulationThread
from cc3d.player5.Utilities.utils import extract_address_int_from_vtk_object
# from cc3d.player5.Simulation.CMLResultReader import CMLResultReader
import cc3d.player5.Simulation


class DataReader(QThread):
    data_read = pyqtSignal(int, name='data_read')

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.cml_reader = parent
        self.step = 0

    def set_step(self, step):
        self.step = step

    def run(self):
        """
        reads simulation data. emits a signal after the read finished
        :return:
        """
        read_state = self.cml_reader.read_simulation_data_non_blocking(self.step)
        if read_state:
            self.data_read.emit(self.step)
        else:
            self.data_read.emit(-1)


class CMLResultReader(SimulationThread.SimulationThread, CMLResultsReaderBase):
    data_read = pyqtSignal(int, name='data_read')
    initial_data_read = pyqtSignal(bool, name='initial_data_read')
    subsequent_data_read = pyqtSignal(int, name='subsequent_data_read')
    final_data_read = pyqtSignal(bool, name='final_data_read')

    def __init__(self, parent):
        CMLResultsReaderBase.__init__(self, parent=parent)
        SimulationThread.SimulationThread.__init__(self, parent)

        # NOTE: to implement synchronization between threads we use semaphores.
        # If you use mutexes for this then if you lock mutex in one thread and try to unlock
        # from another thread than on Linux it will not work. Semaphores are better for this

        # list of *.vtk files containing graphics lattice data
        self.currentStep = 0
        self.__direct_access_flag = False
        self.__mcs_direct_access = 0
        self.__fileNumber = -1  # self.counter=0
        self.newFileBeingLoaded = False
        self.readFileSem = QSemaphore(1)

        # C++ map<int,string> providing mapping from type id to type name
        self.typeIdTypeNameCppMap = None
        self.customVis = None
        self.__initialized = False

        self.stepCounter = 0
        self.reading = False
        self.state = STOP_STATE

        self.stay_in_current_step = False

        self.recently_read_file_number = -1

    def set_stay_in_current_step(self, flag: bool) -> None:
        """

        :param flag:
        :return:
        """
        self.stay_in_current_step = flag

    def get_stay_in_current_step(self) -> bool:
        """

        :return:
        """
        return self.stay_in_current_step

    def got_data(self, file_number):
        """
        slot handling data read
        :param file_number:
        :return:
        """
        self.recently_read_file_number = file_number

        self.data_reader.data_read.disconnect(self.got_data)
        self.reading = False

        if file_number == 0 and not self.__initialized:
            self.initial_data_read.emit(True)
            self.__initialized = True

        if file_number < 0:
            # read did not succeed
            self.set_run_state(state=STOP_STATE)
            self.final_data_read.emit(True)
        else:
            # read successful
            self.subsequent_data_read.emit(file_number)

    def set_run_state(self, state: int) -> None:
        """
        sets current run state
        :param state:
        :return:
        """

        self.state = state

    def keep_going(self):
        """
        executes step fcn if self.state == RUN_STATE
        :return:
        """

        if self.state == RUN_STATE:
            self.step()

    def step(self) -> None:
        """
        executes single step for CMLResultReplay
        :return:
        """

        # ignoring step requests while reading operation is pending
        if self.reading:
            return

        # this section repeats the current step - pretends that the file was read again
        # used to give users a chance to change initial view in the graphics widget to ensure that all screenshots
        # are saved including screenshot for the first file

        if self.stay_in_current_step:
            self.stay_in_current_step = False
            self.reading = False
            if self.recently_read_file_number >= 0:
                self.subsequent_data_read.emit(self.recently_read_file_number)

            return

        if self.__direct_access_flag:
            self.stepCounter = self.__mcs_direct_access
            self.__direct_access_flag = False

        self.data_reader = DataReader(parent=self)
        self.data_reader.set_step(self.stepCounter)
        self.data_reader.data_read.connect(self.got_data)

        self.stepCounter += 1

        # flag indicating that reading is taking place
        self.reading = True
        self.data_reader.start()

    def resetDimensionChangeMonitoring(self):
        """

        :return:
        """

        # this reset is necessary to avoid recursive calls in the
        # SimpleTabView -- this code needs to be changed because at this point it looks horrible
        self.fieldDimPrevious = self.fieldDim

    @property
    def data_ready(self):
        return not self.newFileBeingLoaded

    def read_simulation_data_non_blocking(self, file_number: int) -> bool:
        """
        reads content of the serialized file
        :param file_number: {int}
        :return: {bool} flag whether read was successful or not
        """

        # this flag is used to prevent calling  draw function
        self.newFileBeingLoaded = True

        read_res, extracted_mcs = self.read_simulation_data(file_number)

        if not read_res:
            return False

        self.currentStep = extracted_mcs
        # # # self.currentStep = self.frequency * _i # this is how we set CMS for CML reading before
        self.setCurrentStep(self.currentStep)

        self.newFileBeingLoaded = False

        return True

    def set_current_step_direct_access(self, mcs: int) -> None:
        """
        function used by lattice files data panel module to directly set
        current step
        :param mcs: {int} current mcs - directo access from gui pabel
        :return:
        """
        self.__mcs_direct_access = mcs
        self.__direct_access_flag = True

    def steerUsingGUI(self, _sim):
        """
        dummy overwrite of base class method
        :param _sim:
        :return:
        """

    def run(self):
        """
        dummy overwrite of base class method
        :return:
        """

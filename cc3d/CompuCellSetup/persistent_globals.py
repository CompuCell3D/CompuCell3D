import os
import time
from os.path import join, exists, basename

from cc3d.core.SteppableRegistry import SteppableRegistry
from cc3d.core.FieldRegistry import FieldRegistry
# from cc3d.core.enums import *
# import numpy as np
# from .ExtraFieldAdapter import ExtraFieldAdapter


# class FieldRegistry:
#     # def __init__(self, persistent_globals=None):
#     def __init__(self):
#         self.__field_dict = {}
#
#         # dictionary of fields to create - needed because extra field creation uses lazy-evaluation
#         # format {field_name:field_type}
#         self.__fields_to_create = {}
#
#         self.dim = None
#         self.simthread = None
#
#         self.enable_ad_hoc_field_creation = False
#
#         # dictionary with field creating functions
#         self.field_creating_fcns = {
#             SCALAR_FIELD_NPY: self.create_scalar_field,
#             SCALAR_FIELD_CELL_LEVEL: self.create_scalar_field_cell_level,
#             VECTOR_FIELD_NPY: self.create_vector_field,
#             VECTOR_FIELD_CELL_LEVEL: self.create_vector_field_cell_level,
#
#         }
#
#     def create_field(self, field_name: str, field_type: int) -> ExtraFieldAdapter:
#         """
#
#         :param field_name:
#         :param field_type:
#         :return:
#         """
#         # todo - need to add mechanism to inform player about new field
#
#         field_adapter = self.schedule_field_creation(field_name=field_name, field_type=field_type)
#
#         if self.enable_ad_hoc_field_creation:
#             self.create_fields()
#
#         return field_adapter
#
#     def schedule_field_creation(self, field_name: str, field_type: int) -> ExtraFieldAdapter:
#         """
#         records which fields to create
#         :param field_name:
#         :param field_type:
#         :return:
#         """
#
#         field_adapter = ExtraFieldAdapter(name=field_name, field_type=field_type)
#
#         self.__fields_to_create[field_name] = field_adapter
#
#         return field_adapter
#
#     def fetch_field_adapter(self, field_name: str) -> ExtraFieldAdapter:
#         """
#         Convenience function that fetches field adapter or returns None
#         :param field_name:
#         :return:
#         """
#
#         try:
#             return self.__fields_to_create[field_name]
#         except KeyError:
#             print('Could not create field ', field_name)
#
#     def create_scalar_field(self, field_name: str) -> None:
#
#         """
#
#         Creates scalar field
#
#         :param field_name:
#         :return:
#         """
#
#         field_adapter = self.fetch_field_adapter(field_name=field_name)
#         if field_adapter is None:
#             return
#
#         fieldNP = np.zeros(shape=(self.dim.x, self.dim.y, self.dim.z), dtype=np.float32)
#         ndarrayAdapter = self.simthread.callingWidget.fieldStorage.createFloatFieldPy(self.dim, field_name)
#         # initializing  numpyAdapter using numpy array (copy dims and data ptr)
#         ndarrayAdapter.initFromNumpy(fieldNP)
#         self.addNewField(ndarrayAdapter, field_name, SCALAR_FIELD)
#         self.addNewField(fieldNP, field_name + '_npy', SCALAR_FIELD_NPY)
#
#         field_adapter.set_ref(fieldNP)
#
#     def create_scalar_field_cell_level(self, field_name: str) -> None:
#         """
#         Creates scalar field cell level
#
#         :param field_name:
#         :return:
#         """
#
#         field_adapter = self.fetch_field_adapter(field_name=field_name)
#         if field_adapter is None:
#             return
#
#         field_ref = self.simthread.callingWidget.fieldStorage.createScalarFieldCellLevelPy(field_name)
#         self.addNewField(field_ref, field_name, SCALAR_FIELD_CELL_LEVEL)
#         field_adapter.set_ref(field_ref)
#
#     def create_vector_field(self, field_name: str) -> None:
#         """
#         Creates vector field pixel-level
#         :param field_name:
#         :return:
#         """
#
#         field_adapter = self.fetch_field_adapter(field_name=field_name)
#         if field_adapter is None:
#             return
#
#         fieldNP = np.zeros(shape=(self.dim.x, self.dim.y, self.dim.z, 3), dtype=np.float32)
#         ndarrayAdapter = self.simthread.callingWidget.fieldStorage.createVectorFieldPy(self.dim, field_name)
#
#         # initializing  numpyAdapter using numpy array (copy dims and data ptr)
#         ndarrayAdapter.initFromNumpy(fieldNP)
#         self.addNewField(ndarrayAdapter, field_name, VECTOR_FIELD)
#         self.addNewField(fieldNP, field_name + '_npy', VECTOR_FIELD_NPY)
#
#         field_adapter.set_ref(fieldNP)
#
#     def create_vector_field_cell_level(self, field_name: str) -> None:
#         """
#         Creates vector fiedl cell-level
#         :param field_name:
#         :return:
#         """
#
#         field_adapter = self.fetch_field_adapter(field_name=field_name)
#         if field_adapter is None:
#             return
#
#         field_ref = self.simthread.callingWidget.fieldStorage.createVectorFieldCellLevelPy(field_name)
#         self.addNewField(field_ref, field_name, VECTOR_FIELD_CELL_LEVEL)
#         field_adapter.set_ref(field_ref)
#
#     def create_fields(self) -> None:
#         """
#         Creates Fields that are scheduled to be created. Called after constructors of steppalbes has been called
#         Typically fields are requested int he constructor although we will add functionality to create fields at
#         any point in the simulation
#
#         :return: None
#         """
#
#         for field_name, field_adapter in self.__fields_to_create.items():
#             try:
#                 field_creating_fcn = self.field_creating_fcns[field_adapter.field_type]
#             except KeyError:
#                 print('Could not create field. Could not locate field creating functions for ', field_name)
#                 continue
#
#             # we are creating fields only when internal field reference is None
#             if field_adapter.get_ref() is None:
#                 field_creating_fcn(field_name)
#                 if self.simthread is not None:
#                     self.simthread.add_visualization_field(field_name, field_adapter.field_type)
#
#         self.enable_ad_hoc_field_creation = True
#
#     def get_fields_to_create_dict(self) -> dict:
#         """
#
#         :return:
#         """
#
#         return self.__fields_to_create
#
#     def get_field_dict(self) -> dict:
#         """
#
#         :return:
#         """
#
#         return self.__field_dict
#
#     def addNewField(self, _field, _fieldName, _fieldType):
#         self.__field_dict[_fieldName] = [_field, _fieldType]
#
#     def getFieldNames(self):
#         return self.__field_dict.keys()
#
#     def getScalarFields(self):
#         scalarFieldsDict = {}
#         for fieldName in self.__field_dict:
#             if self.__field_dict[fieldName][1] == SCALAR_FIELD:
#                 scalarFieldsDict[fieldName] = self.__field_dict[fieldName][0]
#
#         return scalarFieldsDict
#
#     def getScalarFieldsCellLevel(self):
#         scalarFieldsCellLevelDict = {}
#         for fieldName in self.__field_dict:
#             if self.__field_dict[fieldName][1] == SCALAR_FIELD_CELL_LEVEL:
#                 scalarFieldsCellLevelDict[fieldName] = self.__field_dict[fieldName][0]
#
#         return scalarFieldsCellLevelDict
#
#     def getVectorFields(self):
#         vectorFieldsDict = {}
#         for fieldName in self.__field_dict:
#             if self.__field_dict[fieldName][1] == VECTOR_FIELD:
#                 vectorFieldsDict[fieldName] = self.__field_dict[fieldName][0]
#
#         return vectorFieldsDict
#
#     def getVectorFieldsCellLevel(self):
#         vectorFieldsCellLevelDict = {}
#         for fieldName in self.__field_dict:
#             if self.__field_dict[fieldName][1] == VECTOR_FIELD_CELL_LEVEL:
#                 vectorFieldsCellLevelDict[fieldName] = self.__field_dict[fieldName][0]
#
#         return vectorFieldsCellLevelDict
#
#     def getFieldData(self, _fieldName):
#         try:
#             return self.__field_dict[_fieldName][0], self.__field_dict[_fieldName][1]  # field, field type
#         except (LookupError, IndexError) as e:
#             return None, None


class PersistentGlobals:
    def __init__(self):
        self.cc3d_xml_2_obj_converter = None
        self.steppable_registry = SteppableRegistry()

        # c++ object reference Simulator.cpp
        self.simulator = None

        #  Simulation Thread - either from the player or from CML
        self.simthread = None

        # hook to player - initialized in the player in the prepareForNewSimulation method
        # this object is not None only when GUI run is requested
        self.view_manager = None

        # an object that writes or reads fields from disk
        self.cml_field_handler = None

        self.simulation_initialized = False
        self.simulation_file_name = None
        self.user_stop_simulation_flag = False

        # class - container that stores information about the fields
        self.field_registry = FieldRegistry()

    @property
    def workspace_dir(self)->str:
        """

        :return:
        """
        workspace_dir = os.path.join(os.path.expanduser('~'), 'CC3DWorkspace')
        if not exists(workspace_dir):
            os.mkdir(workspace_dir)

        return workspace_dir

    @property
    def timestamp_string(self):
        current_time = time.localtime()
        str_f_time = time.strftime
        timestamp_str = "_" + str_f_time("%m", current_time) + "_" + str_f_time("%d",current_time) + "_" + str_f_time(
            "%Y", current_time) + "_" + str_f_time("%H", current_time) + "_" + str_f_time("%M",current_time) + "_" + str_f_time(
            "%S", current_time)

        return timestamp_str

    @property
    def screenshot_directory(self):
        if self.simulation_file_name is None:
            return None
        else:
            sim_base_name = basename(self.simulation_file_name)
            sim_base_name = sim_base_name.replace('.', '_')
            sim_base_name += self.timestamp_string

            return join(self.workspace_dir,sim_base_name)

    def clean(self):
        """

        :return:
        """

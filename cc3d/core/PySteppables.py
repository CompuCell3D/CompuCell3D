import itertools
from collections import OrderedDict
from cc3d.core.iterators import *
from cc3d.core.enums import *
from cc3d.core.ExtraFieldAdapter import ExtraFieldAdapter
# from cc3d.CompuCellSetup.simulation_utils import stop_simulation
from cc3d.CompuCellSetup.simulation_utils import extract_type_names_and_ids
from cc3d import CompuCellSetup
from cc3d.core.XMLUtils import dictionaryToMapStrStr as d2mss
from cc3d.core.XMLDomUtils import XMLElemAdapter
from typing import Union
import types


class SteppablePy:
    def __init__(self):
        self.runBeforeMCS = 0

    def core_init(self):
        """

        :return:
        """

    def start(self):
        """

        :return:
        """

    def step(self, mcs):
        """

        :param _mcs:
        :return:
        """

    def finish(self):
        """

        :return:
        """

    def cleanup(self):
        """

        :return:
        """


class SteppableBasePy(SteppablePy):
    (CC3D_FORMAT, TUPLE_FORMAT) = range(0, 2)

    def __init__(self, simulator=None, frequency=1):
        SteppablePy.__init__(self)
        # SBMLSolverHelper.__init__(self)
        self.frequency = frequency
        self.simulator = simulator
        self.__modulesToUpdateDict = OrderedDict()

        # legacy API
        self.addNewPlotWindow = self.add_new_plot_window
        self.createScalarFieldPy = self.create_scalar_field_py
        self.everyPixelWithSteps = self.every_pixel_with_steps
        self.everyPixel = self.every_pixel

    def core_init(self):

        self.potts = self.simulator.getPotts()
        self.cellField = self.potts.getCellFieldG()
        self.dim = self.cellField.getDim()
        self.inventory = self.simulator.getPotts().getCellInventory()
        self.clusterInventory = self.inventory.getClusterInventory()
        self.cellList = CellList(self.inventory)
        self.cellListByType = CellListByType(self.inventory)
        self.clusterList = ClusterList(self.inventory)
        self.clusters = Clusters(self.inventory)
        self.mcs = -1

        self.plot_dict = {}  # {plot_name:plotWindow  - pW object}

        persistent_globals = CompuCellSetup.persistent_globals
        persistent_globals.attach_dictionary_to_cells()

        type_id_type_name_dict = extract_type_names_and_ids()

        for type_id, type_name in type_id_type_name_dict.items():
            self.typename_to_attribute(cell_type_name=type_name, type_id=type_id)
            # setattr(self, type_name.upper(), type_id)

    def typename_to_attribute(self, cell_type_name: str, type_id: int) -> None:
        """
        sets steppable attribute based on type name
        Performs basic sanity checks
        :param cell_type_name:{str}
        :param type_id:{str}
        :return:
        """

        if cell_type_name.isspace() or not len(cell_type_name.strip()):
            raise AttributeError('cell type "{}" contains whitespaces'.format(cell_type_name))

        if not cell_type_name[0].isalpha():
            raise AttributeError('Invalid cell type "{}" . Type name must start with a letter'.format(cell_type_name))

        cell_type_name_attr = cell_type_name.upper()

        try:
            getattr(self, cell_type_name_attr)
            attribute_already_exists = True
        except AttributeError:
            attribute_already_exists = False

        if attribute_already_exists:
            raise AttributeError('Could not convert cell type {cell_type} to steppable attribute. '
                                 'Attribute {attr_name} already exists . Please change your cell type name'.format(
                cell_type=cell_type_name, attr_name=cell_type_name_attr
            ))

        setattr(self, cell_type_name_attr, type_id)

    def stop_simulation(self):
        """
        Stops simulation
        :return:
        """

        CompuCellSetup.stop_simulation()

    def init(self, _simulator):
        """

        :param _simulator:
        :return:
        """

    def add_steering_panel(self):
        """

        :return:
        """

    def process_steering_panel_data_wrapper(self):
        """

        :return:
        """

    def set_steering_param_dirty(self, flag=False):
        """

        :return:
        """

    def add_new_plot_window(self, title, xAxisTitle, yAxisTitle, xScaleType='linear', yScaleType='linear', grid=True,
                            config_options=None):

        if title in self.plot_dict.keys():
            raise RuntimeError('PLOT WINDOW: ' + title + ' already exists. Please choose a different name')

        pW = CompuCellSetup.simulation_player_utils.add_new_plot_window(title, xAxisTitle, yAxisTitle, xScaleType,
                                                                        yScaleType, grid, config_options=config_options)
        self.plot_dict = {}  # {plot_name:plotWindow  - pW object}

        return pW

    def create_scalar_field_py(self, fieldName: str) -> ExtraFieldAdapter:
        """
        Created extra visualization field
        :param fieldName: {str}
        :return:
        """

        return CompuCellSetup.simulation_player_utils.create_extra_field(field_name=fieldName,
                                                                         field_type=SCALAR_FIELD_NPY)

    def create_scalar_field_cell_level_py(self, fieldName: str) -> ExtraFieldAdapter:
        """
        Creates extra visualization field
        :param fieldName: {str}
        :return:
        """
        return CompuCellSetup.simulation_player_utils.create_extra_field(field_name=fieldName,
                                                                         field_type=SCALAR_FIELD_CELL_LEVEL)

    def create_vector_field_py(self, fieldName: str) -> ExtraFieldAdapter:
        """
        Creates extra visualization vector field (voxel-based)
        :param fieldName: {str}
        :return:
        """

        return CompuCellSetup.simulation_player_utils.create_extra_field(field_name=fieldName,
                                                                         field_type=VECTOR_FIELD_NPY)

    def create_vector_field_cell_level_py(self, fieldName: str) -> ExtraFieldAdapter:
        """
        Creates extra visualization vector field (voxel-based)
        :param fieldName: {str}
        :return:
        """

        return CompuCellSetup.simulation_player_utils.create_extra_field(field_name=fieldName,
                                                                         field_type=VECTOR_FIELD_CELL_LEVEL)

    def every_pixel_with_steps(self, step_x, step_y, step_z):
        """
        Helper function called by every_pixel method. See documentation of every_pixel for details
        :param step_x:
        :param step_y:
        :param step_z:
        :return:
        """
        for x in range(0, self.dim.x, step_x):
            for y in range(0, self.dim.y, step_y):
                for z in range(0, self.dim.z, step_z):
                    yield x, y, z

    def every_pixel(self, step_x=1, step_y=1, step_z=1):
        """
        Returns iterator that walks through pixels of the lattixe. Step variables
        determine if we walk through every pixel - step=1 in this case
        or if we jump step variables then are > 1
        :param step_x:
        :param step_y:
        :param step_z:
        :return:
        """

        if step_x == 1 and step_y == 1 and step_z == 1:

            return itertools.product(range(self.dim.x), range(self.dim.y), range(self.dim.z))
        else:
            return self.every_pixel_with_steps(step_x, step_y, step_z)

    def get_xml_element(self, id: str) -> Union[XMLElemAdapter, None]:
        """
        Fetches XML element by id. Returns XMLElementAdapter object that provides natural way of manipulating
        properties of the underlying XML element

        :param id: {str}  xml element identifier - must be present in the xml
        :return: {XMLElemAdapter} xml element adapter
        """
        xml_id_locator = CompuCellSetup.persistent_globals.xml_id_locator

        if xml_id_locator is None:
            return None

        element_adapter = xml_id_locator.get_xml_element(id=id)

        return element_adapter


    # def registerXMLElementUpdate(self, *args):
    #     '''this function registers core module XML Element from wchich XML subelement has been fetched.It returns XML subelement
    #     '''
    #     # element,coreElement=None,None
    #     # info=sys.version_info
    #     # if info[0]>=2 and info[1]>5:
    #     #     element,coreElement=self.getXMLElementAndModuleRoot(*args,returnModuleRoot=True)  # does not work in python 2.5 - syntax error
    #     # else:
    #     element, coreElement = self.getXMLElementAndModuleRoot(args, returnModuleRoot=True)
    #
    #     coreNameComposite = coreElement.getName()
    #     if coreElement.findAttribute('Name'):
    #         coreNameComposite += coreElement.getAttribute('Name')
    #     elif coreElement.findAttribute('Type'):
    #         coreNameComposite += coreElement.getAttribute('Type')
    #
    #     if element:
    #
    #         # now will register which modules were modified we will use this information when we call update function
    #         currentMCS = self.simulator.getStep()
    #         try:
    #             moduleDict = self.__modulesToUpdateDict[currentMCS]
    #             try:
    #                 moduleDict[coreNameComposite]
    #             except LookupError:
    #                 moduleDict['NumberOfModules'] += 1
    #                 moduleDict[coreNameComposite] = [coreElement, moduleDict['NumberOfModules']]
    #                 # # # print 'moduleDict[NumberOfModules]=',moduleDict['NumberOfModules']
    #
    #         except LookupError:
    #             self.__modulesToUpdateDict[currentMCS] = {coreNameComposite: [coreElement, 0], 'NumberOfModules': 0}
    #
    #     return element
    #
    # def getXMLAttributeValue(self, attr, *args):
    #     element = self.getXMLElement(*args)
    #     if element is not None:
    #         if element.findAttribute(attr):
    #             return element.getAttribute(attr)
    #         else:
    #             raise LookupError('Could not find attribute ' + attr + ' in ' + args)
    #     else:
    #         return None
    #
    # def setXMLAttributeValue(self, attr, value, *args):
    #     element = self.registerXMLElementUpdate(*args)
    #     if element:
    #         if element.findAttribute(attr):
    #             element.updateElementAttributes(d2mss({attr: value}))
    #
    # def updateXML(self):
    #     currentMCS = self.simulator.getStep()
    #     try:
    #         # trying to get dictionary of  modules for which XML has been modified during current step
    #         moduleDict = self.__modulesToUpdateDict[currentMCS]
    #     except LookupError:
    #         # if such dictionary does not exist we clean self.__modulesToUpdateDict deleteing whatever was stored before
    #         self.__modulesToUpdateDict = {}
    #         return
    #
    #     try:
    #         numberOfModules = moduleDict['NumberOfModules']
    #         del moduleDict['NumberOfModules']
    #     except LookupError:
    #         pass
    #
    #     # [1][1] refers to number denoting the order in which module was added
    #     # [1][1] refers to added element with order number being [1][1]
    #     list_of_tuples = sorted(moduleDict.items(), key=lambda x: x[1][1])
    #
    #     # # # print 'list_of_tuples=',list_of_tuples
    #     for elem_tuple in list_of_tuples:
    #         self.simulator.updateCC3DModule(elem_tuple[1][0])
    #
    # def getXMLElement(self, *args):
    #     element = None
    #
    #     if not len(args):
    #         return None
    #
    #     if type(args[0]) is not list:  # it is CC3DXMLElement
    #         element = args[0]
    #     else:
    #         element, moduleRoot = self.getXMLElementAndModuleRoot(*args)
    #
    #     return element if element else None
    #
    # def getXMLElementValue(self, *args):
    #
    #     element = self.getXMLElement(*args)
    #     return element.getText() if element else None
    #
    # def getXMLElementAndModuleRoot(self, *args, **kwds):
    #     ''' This fcn fetches xml element value and returns it as text. Potts, Plugin and steppable are special names and roots of these elements are fetched using simulator
    #         The implementation of this plugin may be simplified. Current implementation is least invasive and requires no changes apart from modifying PySteppables.
    #         This Function greatly simplifies access to XML data - one line  easily replaces  many lines of code
    #     '''
    #
    #     # depending on Python version we might need to pass "extra-tupple-wrapped"
    #     # positional arguments especially in situation when variable list arguments
    #     # are mixed with keyword arguments during function call
    #     if isinstance(args[0], tuple):
    #         args = args[0]
    #
    #     if not isinstance(args[0], list):  # it is CC3DXMLElement
    #         return args[0]
    #
    #     coreModuleElement = None
    #     tmpElement = None
    #     for arg in args:
    #         if type(arg) is list:
    #             if arg[0] == 'Potts':
    #                 coreModuleElement = self.simulator.getCC3DModuleData('Potts')
    #                 tmpElement = coreModuleElement
    #             elif arg[0] == 'Plugin':
    #                 counter = 0
    #                 for attrName in arg:
    #                     if attrName == 'Name':
    #                         pluginName = arg[counter + 1]
    #                         coreModuleElement = self.simulator.getCC3DModuleData('Plugin', pluginName)
    #                         tmpElement = coreModuleElement
    #                         break
    #                     counter += 1
    #
    #             elif arg[0] == 'Steppable':
    #                 counter = 0
    #                 for attrName in arg:
    #                     if attrName == 'Type':
    #                         steppableName = arg[counter + 1]
    #                         coreModuleElement = self.simulator.getCC3DModuleData('Steppable', steppableName)
    #                         tmpElement = coreModuleElement
    #                         break
    #                     counter += 1
    #             else:
    #                 # print 'XML FETCH=',arg
    #                 attrDict = None
    #                 if len(arg) >= 3:
    #                     attrDict = {}
    #                     for attr_tuple in zip(arg[1::2], arg[2::2]):
    #                         if attr_tuple[0] in attrDict.keys():
    #                             raise LookupError('Duplicate attribute name in the access path ' + str(args))
    #                         else:
    #                             attrDict[attr_tuple[0]] = attr_tuple[1]
    #                     attrDict = d2mss(attrDict)
    #                     # attrDict=d2mss(dict((tuple[0],tuple[1]) for tuple in izip(arg[1::2],arg[2::2])))
    #
    #                 if coreModuleElement is not None:
    #                     elemName = arg[0]
    #                     tmpElement = tmpElement.getFirstElement(arg[0],
    #                                                             attrDict) if attrDict is not None else tmpElement.getFirstElement(
    #                         arg[0])
    #
    #     if tmpElement is None:
    #         raise LookupError('Could not find element With the following access path', args)
    #
    #     if 'returnModuleRoot' in kwds.keys():
    #         return tmpElement, coreModuleElement
    #
    #     return tmpElement, None

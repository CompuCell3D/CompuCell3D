from os.path import join
from os.path import exists
from os import makedirs
import cc3d.CompuCellSetup

import PlayerPython

# self.FIELD_TYPES = ("CellField", "ConField", "ScalarField", "ScalarFieldCellLevel" , "VectorField","VectorFieldCellLevel")

MODULENAME = '------- CMLFieldHandler.py: '


class CMLFieldHandler:

    def __init__(self):

        self.field_storage = PlayerPython.FieldStorage()
        self.field_writer = PlayerPython.FieldWriter()
        self.field_writer.setFieldStorage(self.field_storage)

        # not being used currently
        # self.fieldWriter.setFileTypeToBinary(False)

        self.field_types = {}
        self.output_freq = 1
        self.sim = None
        self.output_dir_name = ""
        self.output_file_core_name = "Step"
        self.out_file_number_of_digits = 0
        self.do_not_output_field_list = []
        self.FIELD_TYPES = (
            "CellField", "ConField", "ScalarField", "ScalarFieldCellLevel", "VectorField", "VectorFieldCellLevel")

    def initialize(self, field_storage=None):
        """
        Initializes CMLFieldHandler
        :param field_storage:
        :return:
        """

        peristent_globals = cc3d.CompuCellSetup.persistent_globals
        self.field_writer.init(peristent_globals.simulator)
        if field_storage is not None:
            self.field_writer.setFieldStorage(field_storage)

        self.out_file_number_of_digits = len(str(peristent_globals.simulator.getNumSteps()))
        self.get_info_about_fields()

    # def doNotOutputField(self,_fieldName):
    #     if not _fieldName in  self.doNotOutputFieldList:
    #         self.doNotOutputFieldList.append(_fieldName)
    #
    # def setFieldStorage(self,_fieldStorage):
    #     self.fieldStorage = _fieldStorage
    #     self.fieldWriter.setFieldStorage(self.fieldStorage)
    #
    # def createVectorFieldPy(self,_dim,_fieldName):
    #     import CompuCellSetup
    #     return CompuCellSetup.createVectorFieldPy(_dim,_fieldName)
    #
    # def createVectorFieldCellLevelPy(self,_fieldName):
    #     import CompuCellSetup
    #     return CompuCellSetup.createVectorFieldCellLevelPy(_fieldName)
    #
    # def createFloatFieldPy(self, _dim,_fieldName):
    #     import CompuCellSetup
    #     return CompuCellSetup.createFloatFieldPy(_dim,_fieldName)
    #
    # def createScalarFieldCellLevelPy(self,_fieldName):
    #     import CompuCellSetup
    #     return CompuCellSetup.createScalarFieldCellLevelPy(_fieldName)
    #
    # def clearGraphicsFields(self):
    #     pass
    #
    # def setMaxNumberOfSteps(self,_max):
    #     self.outFileNumberOfDigits = len(str(_max))

    def write_fields(self, _mcs):

        for field_name in self.field_types.keys():
            if self.field_types[field_name] == self.FIELD_TYPES[0]:
                self.field_writer.addCellFieldForOutput()
            elif self.field_types[field_name] == self.FIELD_TYPES[1]:
                self.field_writer.addConFieldForOutput(field_name)
            elif self.field_types[field_name] == self.FIELD_TYPES[2]:
                self.field_writer.addScalarFieldForOutput(field_name)
            elif self.field_types[field_name] == self.FIELD_TYPES[3]:
                self.field_writer.addScalarFieldCellLevelForOutput(field_name)
            elif self.field_types[field_name] == self.FIELD_TYPES[4]:
                self.field_writer.addVectorFieldForOutput(field_name)
            elif self.field_types[field_name] == self.FIELD_TYPES[5]:
                self.field_writer.addVectorFieldCellLevelForOutput(field_name)

        mcsFormattedNumber = str(_mcs).zfill(self.out_file_number_of_digits)

        # e.g. /path/Step_01.vtk
        latticeDataFileName = join(self.output_dir_name, self.output_file_core_name + "_" + mcsFormattedNumber + ".vtk")

        self.field_writer.writeFields(latticeDataFileName)
        self.field_writer.clear()

    def writeXMLDescriptionFile(self, _fileName=""):
        from os.path import join
        """
        This function will write XML description of the stored fields. It has to be called after 
        initialization of theCMLFieldHandler is completed
        """
        import CompuCellSetup
        latticeTypeStr = CompuCellSetup.ExtractLatticeType()
        if latticeTypeStr == "":
            latticeTypeStr = "Square"

        typeIdTypeNameDict = CompuCellSetup.ExtractTypeNamesAndIds()
        # print "typeIdTypeNameDict",typeIdTypeNameDict

        from XMLUtils import ElementCC3D
        dim = self.sim.getPotts().getCellFieldG().getDim()
        numberOfSteps = self.sim.getNumSteps()
        latticeDataXMLElement = ElementCC3D("CompuCell3DLatticeData", {"Version": "1.0"})
        latticeDataXMLElement.ElementCC3D("Dimensions", {"x": str(dim.x), "y": str(dim.y), "z": str(dim.z)})
        latticeDataXMLElement.ElementCC3D("Lattice", {"Type": latticeTypeStr})
        latticeDataXMLElement.ElementCC3D("Output",
                                          {"Frequency": str(self.output_freq), "NumberOfSteps": str(numberOfSteps),
                                           "CoreFileName": self.output_file_core_name,
                                           "Directory": self.output_dir_name})
        # output information about cell type names and cell ids. It is necessary during generation of the PIF files from VTK output
        for typeId in typeIdTypeNameDict.keys():
            latticeDataXMLElement.ElementCC3D("CellType",
                                              {"TypeName": str(typeIdTypeNameDict[typeId]), "TypeId": str(typeId)})

        fieldsXMLElement = latticeDataXMLElement.ElementCC3D("Fields")
        for fieldName in self.field_types.keys():
            fieldsXMLElement.ElementCC3D("Field", {"Name": fieldName, "Type": self.field_types[fieldName]})
        # writing XML description to the disk
        if _fileName != "":
            latticeDataXMLElement.CC3DXMLElement.saveXML(str(_fileName))
        else:
            latticeDataFileName = join(self.output_dir_name, self.output_file_core_name + "LDF.dml")
            latticeDataXMLElement.CC3DXMLElement.saveXML(str(latticeDataFileName))

    def prepareSimulationStorageDir(self, _dirName):

        if self.output_freq:

            if exists(_dirName):
                self.output_dir_name = _dirName
            else:
                try:
                    makedirs(_dirName)
                    self.output_dir_name = _dirName
                except:
                    # if directory cannot be created the simulation data will not be saved even if user requests it
                    self.output_freq = 0
                    print(MODULENAME, "prepareSimulationStorageDir: COULD NOT MAKE DIRECTORY")
                    raise IOError
        else:
            print('\n\n', MODULENAME, "prepareSimulationStorageDir(): Lattice output frequency is invalid")

    def get_info_about_fields(self):

        sim = cc3d.CompuCellSetup.persistent_globals.simulator
        # there will always be cell field
        self.field_types["Cell_Field"] = self.FIELD_TYPES[0]

        # extracting information about concentration vectors
        concFieldNameVec = sim.getConcentrationFieldNameVector()
        for fieldName in concFieldNameVec:
            if not fieldName in self.do_not_output_field_list:
                self.field_types[fieldName] = self.FIELD_TYPES[1]

        # inserting extra scalar fields managed from Python script
        scalarFieldNameVec = self.field_storage.getScalarFieldNameVector()
        for fieldName in scalarFieldNameVec:

            if not fieldName in self.do_not_output_field_list:
                self.field_types[fieldName] = self.FIELD_TYPES[2]

        # inserting extra scalar fields cell levee managed from Python script
        scalarFieldCellLevelNameVec = self.field_storage.getScalarFieldCellLevelNameVector()
        for fieldName in scalarFieldCellLevelNameVec:

            if not fieldName in self.do_not_output_field_list:
                self.field_types[fieldName] = self.FIELD_TYPES[3]

        # inserting extra vector fields  managed from Python script
        vectorFieldNameVec = self.field_storage.getVectorFieldNameVector()
        for fieldName in vectorFieldNameVec:

            if not fieldName in self.do_not_output_field_list:
                self.field_types[fieldName] = self.FIELD_TYPES[4]

        # inserting extra vector fields  cell level managed from Python script
        vectorFieldCellLevelNameVec = self.field_storage.getVectorFieldCellLevelNameVector()
        for fieldName in vectorFieldCellLevelNameVec:

            if not fieldName in self.do_not_output_field_list:
                self.field_types[fieldName] = self.FIELD_TYPES[5]

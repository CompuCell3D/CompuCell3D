# -*- coding: utf-8 -*-
import os
import shutil
import sys
import contextlib

from cc3d.core.ParameterScanUtils import ParameterScanUtils
from cc3d.core.ParameterScanUtils import ParameterScanData
from cc3d.core import DefaultSettingsData as settings_data
from cc3d.core.XMLUtils import ElementCC3D
from cc3d.core.XMLUtils import Xml2Obj
from cc3d.core.XMLUtils import CC3DXMLListPy
from typing import Optional

MODULENAME = '------- pythonSetupScripts/CC3DSimulationDataHandler.py: '


def findRelativePathSegments(basePath, p, rest=[]):
    """
        This function finds relative path segments of path p with respect to base path    
        It returns list of relative path segments and flag whether operation succeeded or not    
    """
    h, t = os.path.split(p)
    pathMatch = False
    if h == basePath:
        pathMatch = True
        return [t] + rest, pathMatch
    print("(h,t,pathMatch)=", (h, t, pathMatch))
    if len(h) < 1: return [t] + rest, pathMatch
    if len(t) < 1: return [h] + rest, pathMatch
    return findRelativePathSegments(basePath, h, [t] + rest)


def find_relative_path(basePath, p):
    relative_path_segments, path_match = findRelativePathSegments(basePath, p)
    if path_match:
        relative_path = ""
        for i in range(len(relative_path_segments)):
            segment = relative_path_segments[i]
            relative_path += segment
            if i != len(relative_path_segments) - 1:
                # we use unix style separators - they work on all (3) platforms
                relative_path += "/"
        return relative_path
    else:
        return p


class GenericResource(object):
    def __init__(self, resource_name=''):
        self.resourceName = resource_name


class CC3DResource(GenericResource):
    def __init__(self):
        GenericResource.__init__(self, 'CC3DResource')
        self.path = ""
        self.type = ""
        self.module = ""
        self.origin = ""  # e.g. serialized
        self.copy = True

    def __str__(self):
        return "ResourcePath: " + str(self.path) + "\n" + "Type: " + str(self.type) + "\n" + "Module: " + str(
            self.module) + "\n" + "Origin: " + str(self.origin) + "\n"

    def __repr__(self):
        return self.__str__()


class CC3DSerializerResource(GenericResource):
    def __init__(self):
        GenericResource.__init__(self, 'CC3DSerializerResource')
        self.outputFrequency = 0
        self.allowMultipleRestartDirectories = False
        self.fileFormat = 'text'
        self.restartDirectory = ''

    def disable_restart(self):
        self.restartDirectory = ''

    def enable_restart(self, restart_dir=''):
        if restart_dir != '':
            self.restartDirectory = restart_dir
        else:
            self.restartDirectory = 'restart'

    def append_xml_stub(self, root_elem):
        # from XMLUtils import ElementCC3D
        print(MODULENAME, 'IN APPEND XML STUB')
        print('self.restartDirectory=', self.restartDirectory)

        if self.outputFrequency > 0:
            attribute_dict = {"OutputFrequency": self.outputFrequency,
                              "AllowMultipleRestartDirectories": self.allowMultipleRestartDirectories,
                              "FileFormat": self.fileFormat}
            root_elem.ElementCC3D('SerializeSimulation', attribute_dict)
        if self.restartDirectory != '':
            attribute_dict = {"RestartDirectory": self.restartDirectory}
            root_elem.ElementCC3D('RestartSimulation', attribute_dict)
            print(MODULENAME, 'attribute_dict=', attribute_dict)


class CC3DParameterScanResource(CC3DResource):
    def __init__(self):
        CC3DResource.__init__(self)
        self.resourceName = 'CC3DParameterScanResource'
        self.type = 'ParameterScan'
        self.basePath = ''

        self.parameterScanFileToDataMap = {}
        self.fileTypeForEditor = 'json'

        # ParameterScanUtils is the class where all parsing and parameter scan data processing takes place
        self.psu = ParameterScanUtils()

    def addParameterScanData(self, psd: ParameterScanData, original_value: Optional[str] = None):
        self.psu.addParameterScanData(psd, original_value=original_value)

    def readParameterScanSpecs(self):
        self.psu.readParameterScanSpecs(self.path)

    def write_parameter_scan_specs(self):
        self.psu.write_parameter_scan_specs(self.path)


class CC3DSimulationData:
    def __init__(self):

        self.pythonScriptResource = CC3DResource()
        self.xmlScriptResource = CC3DResource()
        self.pifFileResource = CC3DResource()
        self.windowScriptResource = CC3DResource()
        self.playerSettingsResource = None
        self.windowDict = {}

        self.serializerResource = None
        self.parameterScanResource = None

        # dictionary of resource files with description (types, plugin, etc)
        self.resources = {}
        # full path to project file
        self.path = ''
        # full path to the directory of project file
        self.basePath = ''

        self.version = "3.5.1"

    @property
    def custom_settings_path(self):
        return os.path.join(self.basePath, 'Simulation', settings_data.SETTINGS_FILE_NAME)

    @property
    def pythonScript(self):
        return self.pythonScriptResource.path

    @pythonScript.setter
    def pythonScript(self, _val):
        self.pythonScriptResource.path = _val

    @property
    def xmlScript(self):
        #         print 'returning self.xmlScriptResource.path=',self.xmlScriptResource.path
        return self.xmlScriptResource.path

    @xmlScript.setter
    def xmlScript(self, _val):
        self.xmlScriptResource.path = _val

    @property
    def pifFile(self):
        return self.pifFileResource.path

    @pifFile.setter
    def pifFile(self, _val):
        self.pifFileResource.path = _val

    @property
    def windowScript(self):
        return self.windowScriptResource.path

    @windowScript.setter
    def windowScript(self, _val):
        self.windowScriptResource.path = _val

    #     def __str__(self):
    #         return "CC3DSIMULATIONDATA: "+self.basePath+"\n"+"\tpython file: "
    #         +self.pythonScript+"\n"+"\txml file: "+self.xmlScript+"\n"+\
    #         "\tpifFile="+self.pifFile+"\n"+"\twindow script="+self.windowScript + str(self.resources)

    def addNewParameterScanResource(self):
        self.parameterScanResource = CC3DParameterScanResource()
        self.parameterScanResource.path = os.path.abspath(os.path.join(self.basePath,
                                                                       'Simulation/ParameterScanSpecs.json'))

    def removeParameterScanResource(self):

        with contextlib.suppress(FileNotFoundError, OSError):
            os.remove(self.parameterScanResource.path)

        self.parameterScanResource = None

    def addNewSerializerResource(self, _outFreq=0, _multipleRestartDirs=False, _format='text', _restartDir=''):
        self.serializerResource = CC3DSerializerResource()

        self.serializerResource.serializerOutputFrequency = _outFreq
        self.serializerResource.serializerAllowMultipleRestartDirectories = _multipleRestartDirs
        self.serializerResource.serializerFileFormat = _format
        self.serializerResource.restartDirectory = _restartDir

    def getResourcesByType(self, _type):
        resourceList = []
        for key, resource in self.resources.items():
            if resource.type == _type:
                resourceList.append(resource)
        return resourceList

    def restartEnabled(self):
        if self.serializerResource:
            return self.serializerResource.restartDirectory != ''
        else:
            return False

    def removeSerializerResource(self):
        self.serializerResource = None

    def addNewResource(self, _fileName, _type):  # called by Twedit

        if _type == "XMLScript":
            self.xmlScriptResource.path = os.path.abspath(_fileName)
            self.xmlScriptResource.type = "XML"
            return

        if _type == "PythonScript":
            self.pythonScriptResource.path = os.path.abspath(_fileName)
            self.pythonScriptResource.type = "Python"
            return

        if _type == "PIFFile":
            # we have to check if there is  PIF file assigned resource. If so we do not want to modify 
            # this resource, rather add another one as a generic type of resource

            if self.pifFileResource.path == '':
                # # #                 self.pifFile = os.path.abspath(_fileName)
                #                 print '_fileName=',_fileName
                self.pifFileResource.path = os.path.abspath(_fileName)
                self.pifFileResource.type = "PIFFile"

                # we will also add PIF File as generic resource

        # adding generic resource type - user specified        
        full_path = os.path.abspath(_fileName)
        resource = CC3DResource()
        resource.path = full_path
        resource.type = _type
        self.resources[full_path] = resource
        print(MODULENAME)
        print("self.resources=", self.resources)
        return

    def removeResource(self, _fileName):
        fileName = os.path.abspath(_fileName)
        print('TRYING TO REMOVE RESOURCE _fileName=', _fileName)
        # file name can be associated with many resources - we have to erase all such associations
        if fileName == self.xmlScript:
            self.xmlScriptResource = CC3DResource()

        if fileName == self.pythonScript:
            self.pythonScriptResource = CC3DResource()

        if fileName == self.pifFile:
            self.pifFileResource = CC3DResource()

        try:
            del self.resources[fileName]
        except LookupError as e:
            pass

        print('After removing resources')

        print(self.resources)
        return


class CC3DSimulationDataHandler:
    def __init__(self, _tabViewWidget=None):
        self.cc3dSimulationData = CC3DSimulationData()
        self.tabViewWidget = _tabViewWidget

    def copy_simulation_data_files(self, _dir):

        simulation_path = os.path.join(_dir, 'Simulation')

        if not os.path.exists(simulation_path):
            os.makedirs(simulation_path)

        # copy project file
        try:
            shutil.copy(self.cc3dSimulationData.path,
                        os.path.join(_dir, os.path.basename(self.cc3dSimulationData.path)))
        except:  # ignore any copy errors
            pass

        if self.cc3dSimulationData.pythonScript != "":
            try:
                shutil.copy(self.cc3dSimulationData.pythonScript,
                            os.path.join(simulation_path, os.path.basename(self.cc3dSimulationData.pythonScript)))
            except shutil.SameFileError:
                pass

        if self.cc3dSimulationData.xmlScript != "":
            try:
                shutil.copy(self.cc3dSimulationData.xmlScript,
                            os.path.join(simulation_path, os.path.basename(self.cc3dSimulationData.xmlScript)))
            except shutil.SameFileError:
                pass

        if self.cc3dSimulationData.pifFile != "":
            try:
                shutil.copy(self.cc3dSimulationData.pifFile,
                            os.path.join(simulation_path, os.path.basename(self.cc3dSimulationData.pifFile)))
            except shutil.SameFileError:
                pass

        if self.cc3dSimulationData.windowScript != "":
            try:
                shutil.copy(self.cc3dSimulationData.windowScript,
                            os.path.join(simulation_path, os.path.basename(self.cc3dSimulationData.windowScript)))
            except shutil.SameFileError:
                pass

        if self.cc3dSimulationData.parameterScanResource:
            try:
                shutil.copy(self.cc3dSimulationData.parameterScanResource.path,
                            os.path.join(simulation_path,
                                         os.path.basename(self.cc3dSimulationData.parameterScanResource.path)))
            except shutil.SameFileError:
                pass

            # copy resource files
        file_names = self.cc3dSimulationData.resources.keys()

        for file_name in file_names:
            try:
                if self.cc3dSimulationData.resources[file_name].copy:
                    shutil.copy(file_name, os.path.join(simulation_path, os.path.basename(file_name)))
            except:
                # ignore any copy errors
                pass

    def read_cc3_d_file_format(self, file_name):
        """
        This function reads the CompuCell3D (.cc3d -XML)file. Which contains the file paths to
        all the resources in used in the project. 'cc3dSimulationData' object in this class holds
        all file paths and read data.

        :param file_name: file path for the
        :return:
        """
        # Import XML utils to read the .cc3d xml file
        xml2_obj_converter = Xml2Obj()

        # Get the full file path .cc3d xml file
        file_full_path = os.path.abspath(file_name)
        self.cc3dSimulationData.basePath = os.path.dirname(file_full_path)
        self.cc3dSimulationData.path = file_full_path
        bp = self.cc3dSimulationData.basePath

        # Read the .cc3d xml and get the root element
        root_element = xml2_obj_converter.Parse(file_full_path)  # this is simulation element

        # Check if custom settings file (Simulation/_settings.xml) exists.
        custom_settings_flag = os.path.isfile(
            os.path.join(self.cc3dSimulationData.basePath, 'Simulation', settings_data.SETTINGS_FILE_NAME))

        if custom_settings_flag:
            # If setting file is there load it to resources as PlayerSettings
            self.cc3dSimulationData.playerSettingsResource = CC3DResource()
            self.cc3dSimulationData.playerSettingsResource.path = os.path.abspath(
                os.path.join(self.cc3dSimulationData.basePath, 'Simulation', settings_data.SETTINGS_FILE_NAME))

            self.cc3dSimulationData.playerSettingsResource.type = "PlayerSettings"
            print('GOT CUSTOM SETTINGS : ', self.cc3dSimulationData.playerSettingsResource.path)

        # Get the version of the file
        if root_element.findAttribute('version'):
            version = root_element.getAttribute('version')
            self.cc3dSimulationData.version = version

        # Get the model xml file
        if root_element.getFirstElement("XMLScript"):
            # If XML file exists load in resources as XMLScript
            xml_script_relative = root_element.getFirstElement("XMLScript").getText()
            self.cc3dSimulationData.xmlScriptResource.path = os.path.abspath(
                os.path.join(bp, xml_script_relative))  # normalizing path to xml script
            self.cc3dSimulationData.xmlScriptResource.type = "XMLScript"

        # Get the python script for the model
        if root_element.getFirstElement("PythonScript"):
            # If python file exists load in resources as PythonScript
            python_script_relative = root_element.getFirstElement("PythonScript").getText()
            self.cc3dSimulationData.pythonScriptResource.path = os.path.abspath(
                os.path.join(bp, python_script_relative))  # normalizing path to python script
            self.cc3dSimulationData.pythonScriptResource.type = "PythonScript"

        # Get the PIF file resource for the model
        if root_element.getFirstElement("PIFFile"):
            # If PIF file exists load in resources as PIFFile
            pif_file_relative = root_element.getFirstElement("PIFFile").getText()
            self.cc3dSimulationData.pifFileResource.path = os.path.abspath(
                os.path.join(bp, pif_file_relative))  # normalizing path
            self.cc3dSimulationData.pifFileResource.type = "PIFFile"

        # Read the SerializeSimulation element which have the data on serialization of the resources.
        # todo - remove this section - we no longer need serializer resource

        if root_element.getFirstElement("SerializeSimulation"):
            serialize_elem = root_element.getFirstElement("SerializeSimulation")
            self.cc3dSimulationData.serializerResource = CC3DSerializerResource()
            if serialize_elem:
                if serialize_elem.findAttribute("OutputFrequency"):
                    self.cc3dSimulationData.serializerResource.outputFrequency = serialize_elem.getAttributeAsInt(
                        "OutputFrequency")

                if serialize_elem.findAttribute("AllowMultipleRestartDirectories"):
                    self.cc3dSimulationData.serializerResource.allowMultipleRestartDirectories = serialize_elem.getAttributeAsBool(
                        "AllowMultipleRestartDirectories")

                if serialize_elem.findAttribute("FileFormat"):
                    self.cc3dSimulationData.serializerResource.fileFormat = serialize_elem.getAttribute("FileFormat")

        if root_element.getFirstElement("RestartSimulation"):
            restart_elem = root_element.getFirstElement("RestartSimulation")
            if not self.cc3dSimulationData.serializerResource:
                self.cc3dSimulationData.serializerResource = CC3DSerializerResource()

            if restart_elem.findAttribute("RestartDirectory"):
                self.cc3dSimulationData.serializerResource.restartDirectory = restart_elem.getAttribute(
                    "RestartDirectory")

        # Reading parameter scan resources in the .cc3d file
        if root_element.getFirstElement("ParameterScan"):
            ps_file = root_element.getFirstElement("ParameterScan").getText()
            self.cc3dSimulationData.parameterScanResource = CC3DParameterScanResource()
            self.cc3dSimulationData.parameterScanResource.path = os.path.abspath(
                os.path.join(bp, ps_file))  # normalizing path to python script
            self.cc3dSimulationData.parameterScanResource.type = 'ParameterScan'
            # setting same base path for parameter scan as for the project
            # - necessary to get relative paths in the parameterSpec file
            self.cc3dSimulationData.parameterScanResource.basePath = self.cc3dSimulationData.basePath
            self.cc3dSimulationData.parameterScanResource.readParameterScanSpecs()

            # reading content of XML parameter scan specs
            # ------------------------------------------------------------------ IMPORTANT IMPOTRANT ----------------
            # WE HAVE TO CALL MANUALLY readParameterScanSpecs because
            # if it is called each time CC3DSiulationDataHandler calls readCC3DFileFormat
            # it may cause problems with parameter scan
            # namely one process will attempt to read parameter scan specs while another might try
            # to write to it and error will get thrown and synchronization gets lost
            # plus readCC3DFileFormat should read .cc3d only , not files which are included from .cc3d
            # ------------------------------------------------------------------ IMPORTANT IMPOTRANT ----------------

        # Reading the remaining resources in the .cc3d file
        resourceList = CC3DXMLListPy(root_element.getElements("Resource"))
        for resourceElem in resourceList:
            cc3dResource = CC3DResource()
            cc3dResource.path = os.path.abspath(os.path.join(bp, resourceElem.getText()))

            if resourceElem.findAttribute("Type"):
                cc3dResource.type = resourceElem.getAttribute("Type")

            if resourceElem.findAttribute("Module"):
                cc3dResource.module = resourceElem.getAttribute("Module")

            if resourceElem.findAttribute("Origin"):
                cc3dResource.origin = resourceElem.getAttribute("Origin")

            if resourceElem.findAttribute("Copy"):
                copyAttr = resourceElem.getAttribute("Copy")
                if copyAttr.lower() == "no":
                    cc3dResource.copy = False

            self.cc3dSimulationData.resources[cc3dResource.path] = cc3dResource

    def format_resource_element(self, resource, element_name=""):
        el_name = ""
        if element_name != "":
            el_name = element_name
        else:
            el_name = "Resource"

        attribute_dict = {}
        if resource.type != "":
            attribute_dict["Type"] = resource.type

        if resource.module != "":
            attribute_dict["Module"] = resource.module

        if resource.origin != "":
            attribute_dict["Origin"] = resource.origin

        if not resource.copy:
            attribute_dict["Copy"] = "No"

        return el_name, attribute_dict, find_relative_path(self.cc3dSimulationData.basePath, resource.path)

    def write_cc3d_file_format(self, file_name):

        csd = self.cc3dSimulationData
        simulation_element = ElementCC3D("Simulation", {"version": csd.version})

        if csd.xmlScriptResource.path != "":
            el_name, attribute_dict, path = self.format_resource_element(csd.xmlScriptResource, "XMLScript")
            simulation_element.ElementCC3D(el_name, attribute_dict, path)

        if csd.pythonScriptResource.path != "":
            el_name, attribute_dict, path = self.format_resource_element(csd.pythonScriptResource, "PythonScript")
            simulation_element.ElementCC3D(el_name, attribute_dict, path)

        if csd.pifFileResource.path != "":
            el_name, attribute_dict, path = self.format_resource_element(csd.pifFileResource, "PIFFile")
            simulation_element.ElementCC3D(el_name, attribute_dict, path)

        if csd.windowScriptResource.path != "":
            el_name, attribute_dict, path = self.format_resource_element(csd.windowScriptResource, "WindowScript")
            simulation_element.ElementCC3D(el_name, attribute_dict, path)

        resources_dict = {}
        # storing resources in a dictionary using resource type as a key
        for resource_key, resource in csd.resources.items():
            if resource.type == "PIFFile" and csd.pifFileResource.path == resource.path:
                print(MODULENAME, "IGNORING RESOURCE =", resource.path)
                continue

            try:
                resources_dict[resource.type].append(resource)
            except LookupError:
                resources_dict[resource.type] = [resource]

            # sort resources according to path name
        for resource_type, resource_list in resources_dict.items():
            resource_list = sorted(resource_list, key=lambda x: x.path)

            # after sorting have to reinsert list into dictionary to have it available later
            resources_dict[resource_type] = resource_list

        sorted_resource_type_names = list(resources_dict.keys())
        sorted_resource_type_names.sort()

        for resource_type in sorted_resource_type_names:
            for resource in resources_dict[resource_type]:
                el_name, attribute_dict, path = self.format_resource_element(resource)
                simulation_element.ElementCC3D(el_name, attribute_dict, path)

        if csd.serializerResource:
            csd.serializerResource.append_xml_stub(simulation_element)

        if csd.parameterScanResource:
            el_name, attribute_dict, path = self.format_resource_element(csd.parameterScanResource, 'ParameterScan')
            simulation_element.ElementCC3D(el_name, attribute_dict, path)

        simulation_element.CC3DXMLElement.saveXML(str(file_name))

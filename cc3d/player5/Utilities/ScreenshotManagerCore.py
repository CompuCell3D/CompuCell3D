# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from cc3d.player5.Utilities import ScreenshotData
import json
import cc3d.core.Version as Version

MODULENAME = '---- ScreenshotManager.py: '

class ScreenshotManagerCore(object):
    def __init__(self):

        self.screenshotDataDict = {}
        self.screenshotCounter3D = 0
        self.screenshotGraphicsWidget = None
        self.gd = None
        self.bsd = None
        self.screenshot_number_of_digits = 10

        self.screenshot_file_parsers = {
            '3.7.9':self.readScreenshotDescriptionFile_JSON_379,
            '3.7.10': self.readScreenshotDescriptionFile_JSON_379,
            '3.8.0': self.readScreenshotDescriptionFile_JSON_379,
            '4.0.0':self.readScreenshotDescriptionFile_JSON_379,

        }

    def cleanup(self):
        """
        Implementes cleanup actions
        :return: None
        """
        raise NotImplementedError()

    def produceScreenshotCoreName(self, _scrData):
        return str(_scrData.plotData[0]) + "_" + str(_scrData.plotData[1])

    def produceScreenshotName(self, _scrData):
        screenshotName = "Screenshot"
        screenshotCoreName = "Screenshot"

        if _scrData.spaceDimension == "2D":
            screenshotCoreName = self.produceScreenshotCoreName(_scrData)
            screenshotName = screenshotCoreName + "_" + _scrData.spaceDimension + "_" + _scrData.projection + "_" + str(
                _scrData.projectionPosition)
        elif _scrData.spaceDimension == "3D":
            screenshotCoreName = self.produceScreenshotCoreName(_scrData)
            screenshotName = screenshotCoreName + "_" + _scrData.spaceDimension + "_" + str(self.screenshotCounter3D)
        return (screenshotName, screenshotCoreName)


    def writeScreenshotDescriptionFile_JSON(self, filename):
        """
        Writes JSON format of screenshot description file
        :param filename: {str} file name
        :return: None
        """

        root_elem = OrderedDict()
        root_elem['Version'] = Version.getVersionAsString()
        root_elem['ScreenshotData'] = OrderedDict()


        scr_container_elem = root_elem['ScreenshotData']

        for name in self.screenshotDataDict:
            scr_data = self.screenshotDataDict[name]
            scr_container_elem[name] = OrderedDict()
            scr_elem = scr_container_elem[name]
            scr_elem['Plot'] = {'PlotType': str(scr_data.plotData[1]), 'PlotName': str(scr_data.plotData[0])}

            if scr_data.spaceDimension == '2D':
                scr_elem['Dimension'] = '2D'

                scr_elem['Projection'] = {'ProjectionPlane': scr_data.projection,
                                                          'ProjectionPosition': int(scr_data.projectionPosition)}

            if scr_data.spaceDimension == '3D':
                scr_elem['Dimension'] = '3D'
                scr_elem['Projection'] = {'ProjectionPlane': None, 'ProjectionPosition': None}

            scr_elem['CameraClippingRange'] = {
                'Min': str(scr_data.clippingRange[0]),
                'Max': str(scr_data.clippingRange[1])
            }

            scr_elem['CameraFocalPoint'] = {
                'x': str(scr_data.focalPoint[0]),
                'y': str(scr_data.focalPoint[1]),
                'z': str(scr_data.focalPoint[2])
            }

            scr_elem['CameraPosition'] = {
                'x': str(scr_data.position[0]),
                'y': str(scr_data.position[1]),
                'z': str(scr_data.position[2])
            }

            scr_elem['CameraViewUp'] = {
                'x': str(scr_data.viewUp[0]),
                'y': str(scr_data.viewUp[1]),
                'z': str(scr_data.viewUp[2])
            }

            scr_elem['Size'] = {
                    'Width': int(scr_data.win_width),
                    'Height': int(scr_data.win_height)
            }

            scr_elem['CellBorders'] = bool(scr_data.cell_borders_on)
            scr_elem['Cells'] = bool(scr_data.cells_on)
            scr_elem['ClusterBorders'] = bool(scr_data.cluster_borders_on)
            scr_elem['CellGlyphs'] = bool(scr_data.cell_glyphs_on)
            scr_elem['FPPLinks'] = bool(scr_data.fpp_links_on)
            scr_elem['BoundingBox'] = bool(scr_data.bounding_box_on)
            scr_elem['LatticeAxes'] = bool(scr_data.lattice_axes_on)
            scr_elem['LatticeAxesLabels'] = bool(scr_data.lattice_axes_labels_on)

            if scr_data.invisible_types is None:
                scr_data.invisible_types = []
            scr_elem['TypesInvisible'] = scr_data.invisible_types

            scr_elem['metadata'] = scr_data.metadata

        with open(str(filename), 'w') as f_out:
            f_out.write(json.dumps(root_elem, indent=4))

    def writeScreenshotDescriptionFile(self, filename):
        """
        Writes screenshot description file
        :param filename: {str} file name
        :return: None
        """

        self.writeScreenshotDescriptionFile_JSON(filename=filename)


    def readScreenshotDescriptionFile_JSON(self,filename):
        """
        parses screenshot description JSON file and stores instances ScreenshotData in appropriate
        container
        :param filename: {str} json file name
        :return: None
        """

        with open(filename,'r') as f_in:
            root_elem = json.load(f_in)

        if root_elem is None:
            print(('Could not read screenshot description file {}'.format(filename)))
            return

        version = root_elem['Version']
        scr_data_container = root_elem['ScreenshotData']


        # will replace it with a dict, for now leaving it as an if statement

        try:

            screenshot_file_parser_fcn = self.screenshot_file_parsers[version]
        except KeyError:
            raise RuntimeError('Unknown version of screenshot description: {}. Make sure '
                               'that <b>ScreenshotManagerCore.py</b> defines <b>screenshot_file_parser</b> '
                               'for this version of CompuCell3D'.format(version))

        screenshot_file_parser_fcn(scr_data_container)

        # if version == '3.7.9':
        #     self.readScreenshotDescriptionFile_JSON_379(scr_data_container)
        # else:
        #     raise RuntimeError('Unknown version of screenshot description: {}'.format(version))




    def readScreenshotDescriptionFile_JSON_379(self,scr_data_container):
        """
        parses screenshot description JSON file and stores instances ScreenshotData in appropriate
        container
        :param scr_data_container: {dict} ScreenShotData json dict
        :return: None
        """

        # with open(filename,'r') as f_in:
        #     root_elem = json.load(f_in)
        #
        # if root_elem is None:
        #     print('Could not read screenshot description file {}'.format(filename))
        #     return
        #
        #
        # scr_data_container = root_elem['ScreenshotData']

        for scr_name, scr_data_elem in list(scr_data_container.items()):
            scr_data = ScreenshotData()
            scr_data.screenshotName = scr_name

            scr_data.plotData = tuple([str(x) for x in (scr_data_elem['Plot']['PlotName'], scr_data_elem['Plot']['PlotType'])])
            scr_data.spaceDimension = str(scr_data_elem['Dimension'])
            try:
                scr_data.projection = str(scr_data_elem['Projection']['ProjectionPlane'])
                scr_data.projectionPosition = scr_data_elem['Projection']['ProjectionPosition']
            except KeyError:
                pass
            scr_data.win_width = scr_data_elem['Size']['Width']
            scr_data.win_height = scr_data_elem['Size']['Height']

            scr_data.cell_borders_on = scr_data_elem['CellBorders']
            scr_data.cells_on = scr_data_elem['Cells']
            scr_data.cluster_borders_on = scr_data_elem['ClusterBorders']
            scr_data.cell_glyphs_on = scr_data_elem['CellGlyphs']
            scr_data.fpp_links_on = scr_data_elem['FPPLinks']
            scr_data.bounding_box_on = scr_data_elem['BoundingBox']
            scr_data.lattice_axes_on = scr_data_elem['LatticeAxes']
            scr_data.lattice_axes_labels_on = scr_data_elem['LatticeAxesLabels']
            scr_data.invisible_types = scr_data_elem['TypesInvisible']

            cam_settings = []

            clipping_range_element = scr_data_elem['CameraClippingRange']
            cam_settings.append(float(clipping_range_element['Min']))
            cam_settings.append(float(clipping_range_element['Max']))

            focal_point_element = scr_data_elem['CameraFocalPoint']
            cam_settings.append(float(focal_point_element['x']))
            cam_settings.append(float(focal_point_element['y']))
            cam_settings.append(float(focal_point_element['z']))

            position_element = scr_data_elem['CameraPosition']
            cam_settings.append(float(position_element['x']))
            cam_settings.append(float(position_element['y']))
            cam_settings.append(float(position_element['z']))

            view_up_element = scr_data_elem['CameraViewUp']
            cam_settings.append(float(view_up_element['x']))
            cam_settings.append(float(view_up_element['y']))
            cam_settings.append(float(view_up_element['z']))

            scr_data.extractCameraInfoFromList(cam_settings)

            # getting rid of unicode in the keys
            metadata_dict = {}
            for k, v in list(scr_data_elem['metadata'].items()):
                metadata_dict[str(k)] = v

            scr_data.metadata = metadata_dict

            # checking for extra metadata entries added
            # you may reset this list after bumping up the version of json file
            # todo fix - we will be permissive as far as DisplayMinMaxInfo
            # extra_metadata_keys = ['DisplayMinMaxInfo']
            extra_metadata_keys = []

            for key in extra_metadata_keys:
                if key not in list(metadata_dict.keys()):
                    raise KeyError('Missing key in the metadata: {}'.format(key))

            # scr_data.screenshotGraphicsWidget = self.screenshotGraphicsWidget

            self.screenshotDataDict[scr_data.screenshotName] = scr_data


    def readScreenshotDescriptionFile(self, filename):
        """
        Reads screenshot description file
        :param filename:{str} scr descr file name
        :return: None
        """

        self.readScreenshotDescriptionFile_JSON(filename=filename)

    def safe_writeScreenshotDescriptionFile(self, out_fname):
        """
        writes screenshot descr file in a safe mode. any problems are reported via warning
        :param out_fname: {str}
        :return: None
        """
        raise NotImplementedError()

    def serialize_screenshot_data(self):
        """
        Method called immediately after we add new screenshot via camera button. It serializes all screenshots data
        for future reference/reuse
        :return: None
        """
        raise NotImplementedError

    def add2DScreenshot(self, _plotName, _plotType, _projection, _projectionPosition, _camera, metadata):
        """
        adds screenshot stub based on current specification of graphics window
        Typically called from GraphicsFrameWidget
        :param _plotName:
        :param _plotType:
        :param _projection:
        :param _projectionPosition:
        :param _camera:
        :return:
        """
        raise NotImplementedError()

    def add3DScreenshot(self, _plotName, _plotType, _camera,metadata):
        """
        adds screenshot stub based on current specification of graphics window
        Typically called from GraphicsFrameWidget
        :param _plotName:
        :param _plotType:
        :param _projection:
        :param _projectionPosition:
        :param _camera:
        :return:
        """

        raise NotImplementedError()

    def get_basic_simulation_data(self):
        """
        Returns an instance of BasicSimulationData. Needs to be reimplemented or bsd member needs to be set
        :return: {BasicSimulationData}
        """
        return self.bsd

    # called from SimpleTabView:handleCompletedStep{Regular,CML*}
    def outputScreenshots(self, general_screenshot_directory_name :str, mcs:str):
        """
        Outputs screenshot
        :param general_screenshot_directory_name:
        :param mcs:
        :return:
        """

        bsd = self.get_basic_simulation_data()
        if self.gd is None or bsd is None:
            print('GenericDrawer or basic simulation data is None. Could not output screenshots')
            return

        # fills string with 0's up to self.screenshotNumberOfDigits width
        mcsFormattedNumber = str(mcs).zfill(self.screenshot_number_of_digits)

        for i, screenshot_name in enumerate(self.screenshotDataDict.keys()):
            screenshot_data = self.screenshotDataDict[screenshot_name]

            if not screenshot_name:
                screenshot_name = 'screenshot_' + str(i)

            screenshot_dir = os.path.join(general_screenshot_directory_name, screenshot_name)

            # will create screenshot directory if directory does not exist
            if not os.path.isdir(screenshot_dir):
                os.mkdir(screenshot_dir)

            screenshot_fname = os.path.join(screenshot_dir, screenshot_name + "_" + mcsFormattedNumber + ".png")

            self.gd.clear_display()
            self.gd.draw(screenshot_data=screenshot_data, bsd=bsd, screenshot_name=screenshot_name)
            self.gd.output_screenshot(screenshot_fname=screenshot_fname)

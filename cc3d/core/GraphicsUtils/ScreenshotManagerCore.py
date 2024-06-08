# -*- coding: utf-8 -*-
from typing import Union
import os
from os.path import dirname, join, exists
from collections import OrderedDict
from cc3d.core.GraphicsUtils.ScreenshotData import ScreenshotData
import json
import cc3d
from cc3d import CompuCellSetup
from typing import Optional

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
            '3.7.9': self.read_screenshot_description_file_json_379,
            '3.7.10': self.read_screenshot_description_file_json_379,
            '3.8.0': self.read_screenshot_description_file_json_379,
            '4.x': self.read_screenshot_description_file_json_379,

        }

        self.ad_hoc_screenshot_dict = {}
        self.output_error_flag = False
        self.padding = 4
        self.screenshot_config_counter = 0
        self.screenshot_config_counter_separator = "___"

    def fetch_screenshot_description_file_parser_fcn(self, screenshot_file_version: str):
        """
        Fetches "best" parsing function give the version in the screenshot description file
        :return:
        """

        try:
            screenshot_file_parser_fcn = self.screenshot_file_parsers[screenshot_file_version]
            return screenshot_file_parser_fcn
        except KeyError:
            version_list = screenshot_file_version.split('.')
            for version_max_idx in [2, 1]:
                version_tmp = '.'.join(version_list[:version_max_idx]) + '.x'
                try:
                    screenshot_file_parser_fcn = self.screenshot_file_parsers[version_tmp]
                    return screenshot_file_parser_fcn
                except KeyError:
                    pass

        raise KeyError(f'Could not find parser for the following version of screenshot description file: '
                       f'{screenshot_file_version}')

    @staticmethod
    def get_screenshot_dir_name():
        return CompuCellSetup.persistent_globals.output_directory

    def cleanup(self):
        """
        Implementes cleanup actions
        :return: None
        """
        raise NotImplementedError()

    def produce_screenshot_core_name(self, _scrData):
        return str(_scrData.plotData[0]) + "_" + str(_scrData.plotData[1])

    def produce_screenshot_name(self, _scrData):
        screenshot_name = "Screenshot"
        screenshot_core_name = "Screenshot"

        if _scrData.spaceDimension == "2D":
            screenshot_core_name = self.produce_screenshot_core_name(_scrData)
            screenshot_name = screenshot_core_name + "_" + _scrData.spaceDimension + "_" + _scrData.projection + "_" + str(
                _scrData.projectionPosition)
        elif _scrData.spaceDimension == "3D":
            screenshot_core_name = self.produce_screenshot_core_name(_scrData)
            screenshot_name = screenshot_core_name + "_" + _scrData.spaceDimension + "_" + str(self.screenshotCounter3D)
        return (screenshot_name, screenshot_core_name)

    @staticmethod
    def starting_screenshot_description_data():
        """Generates the start of screenshot description data"""

        root_elem = OrderedDict()
        root_elem['Version'] = cc3d.__version__
        root_elem['ScreenshotData'] = OrderedDict()
        return root_elem

    @staticmethod
    def append_screenshot_description_data(root_elem: dict, data_elem: dict, screenshot_uid: Optional[str]=None):
        """
        Append a screenshot data element

        :param root_elem: screenshot data
        :param data_elem: data element for a field
        """

        key = data_elem['Plot']['PlotName'] if screenshot_uid is None else screenshot_uid
        root_elem['ScreenshotData'][key] = data_elem

    def generate_screenshot_description_data(self):
        """Generates screenshot description data"""

        root_elem = self.starting_screenshot_description_data()

        for screenshot_uid in self.screenshotDataDict:
            self.append_screenshot_description_data(root_elem, self.screenshotDataDict[screenshot_uid].to_json(), screenshot_uid)

        return root_elem

    def write_screenshot_description_file_json(self, filename):
        """
        Writes JSON format of screenshot description file
        :param filename: {str} file name
        :return: None
        """

        with open(str(filename), 'w') as f_out:
            f_out.write(json.dumps(self.generate_screenshot_description_data(), indent=4))

    def write_screenshot_description_file(self, filename):
        """
        Writes screenshot description file
        :param filename: {str} file name
        :return: None
        """

        self.write_screenshot_description_file_json(filename=filename)

    def read_screenshot_description_data(self, root_elem):
        """
        parses screenshot description data and stores instances ScreenshotData in appropriate container
        :param root_elem: data
        :return: None
        """

        version = root_elem['Version']
        scr_data_container = root_elem['ScreenshotData']

        # will replace it with a dict, for now leaving it as an if statement

        try:
            screenshot_file_parser_fcn = self.fetch_screenshot_description_file_parser_fcn(version)
            # screenshot_file_parser_fcn = self.screenshot_file_parsers[version]
        except KeyError:
            raise RuntimeError('Unknown version of screenshot description: {}. Make sure '
                               'that <b>ScreenshotManagerCore.py</b> defines <b>screenshot_file_parser</b> '
                               'for this version of CompuCell3D'.format(version))

        screenshot_file_parser_fcn(scr_data_container)
        self.find_highest_screenshot_config_counter()

    def read_screenshot_description_file_json(self, filename):
        """
        parses screenshot description JSON file and stores instances ScreenshotData in appropriate
        container
        :param filename: {str} json file name
        :return: None
        """

        with open(filename, 'r') as f_in:
            root_elem = json.load(f_in)

        if root_elem is None:
            print(('Could not read screenshot description file {}'.format(filename)))
            return

        self.read_screenshot_description_data(root_elem)

    def read_screenshot_description_file_json_379(self, scr_data_container):
        """
        parses screenshot description JSON file and stores instances ScreenshotData in appropriate
        container
        :param scr_data_container: {dict} ScreenShotData json dict
        :return: None
        """
        for scr_name, scr_data_elem in list(scr_data_container.items()):
            self.screenshotDataDict[scr_name] = ScreenshotData.from_json(scr_data_elem, scr_name)

    def find_highest_screenshot_config_counter(self):
        for screenshot_uid in self.screenshotDataDict.keys():

            counter_str = screenshot_uid.split(self.screenshot_config_counter_separator)[-1]
            try:
                val = int(counter_str)
                if val > self.screenshot_config_counter:
                    self.screenshot_config_counter = val
            except ValueError:
                pass





    @staticmethod
    def get_screenshot_filename() -> Union[str, None]:
        """

        :return:
        """
        sim_file_name = CompuCellSetup.persistent_globals.simulation_file_name
        if sim_file_name is None and sim_file_name != '':
            print('Unknown simulation file name . Cannot locate screenshot_data folder')
            return None

        guessed_screenshot_name = join(dirname(sim_file_name), 'screenshot_data', 'screenshots.json')

        return guessed_screenshot_name

    def read_screenshot_description_file(self, screenshot_fname=None):
        """
        Reads screenshot description file. checks persisten globals or the simulation name and
        looks for screenshot_data folder in the project location
        :return: None
        """

        if screenshot_fname is not None:
            screenshot_filename = screenshot_fname
        else:
            screenshot_filename = self.get_screenshot_filename()

        if not exists(screenshot_filename):
            return
            # raise RuntimeError('Could not locate screenshot description file: {}'.format(screenshot_filename))

        self.read_screenshot_description_file_json(filename=screenshot_filename)

    def safe_write_screenshot_description_file(self, out_fname):
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

    def add_2d_screenshot(self, _plotName, _plotType, _projection, _projectionPosition, _camera, metadata):
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

    def add_3d_screenshot(self, _plotName, _plotType, _camera, metadata):
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

    def has_ad_hoc_screenshots(self) -> bool:
        """
        Returns a flag that tells if ad_hoc screenshots have been requested
        :return:
        """

        return len(self.ad_hoc_screenshot_dict) > 0

    def add_ad_hoc_screenshot(self, mcs: int, screenshot_label: str):
        """
        adds request to store ad_hoc screenshot
        :param mcs:
        :param screenshot_label:
        :return:
        """
        self.ad_hoc_screenshot_dict[screenshot_label] = mcs

    def output_screenshots_impl(self, mcs: int, screenshot_label_list: list):
        """
        implementation function ofr taking screenshots
        :param mcs:
        :param screenshot_label_list:
        :return:
        """
        if self.output_error_flag:
            return

        screenshot_directory_name = self.get_screenshot_dir_name()

        bsd = self.get_basic_simulation_data()
        if self.gd is None or bsd is None:
            print('GenericDrawer or basic simulation data is None. Could not output screenshots')
            return

        # fills string with 0's up to self.screenshotNumberOfDigits width
        mcs_formatted_number = str(mcs).zfill(self.screenshot_number_of_digits)

        for i, screenshot_name in enumerate(screenshot_label_list):
            try:
                screenshot_data = self.screenshotDataDict[screenshot_name]
            except KeyError:
                print(f'Could not find screenshot description for the following label: {screenshot_name}')
                continue

            if not screenshot_name:
                screenshot_name = 'screenshot_' + str(i)

            screenshot_dir = os.path.join(screenshot_directory_name, screenshot_name)

            # will create screenshot directory if directory does not exist
            if not os.path.isdir(screenshot_dir):
                os.mkdir(screenshot_dir)

            screenshot_fname = os.path.join(screenshot_dir, screenshot_name + "_" + mcs_formatted_number + ".png")

            self.gd.clear_display()
            self.gd.draw(screenshot_data=screenshot_data, bsd=bsd, screenshot_name=screenshot_name)
            self.gd.output_screenshot(screenshot_fname=screenshot_fname, screenshot_data=screenshot_data)

    def output_screenshots(self, mcs: int) -> None:
        """
        Outputs screenshot        
        :param mcs:
        :return:
        """

        if len(self.ad_hoc_screenshot_dict):
            self.output_screenshots_impl(mcs=mcs, screenshot_label_list=list(self.ad_hoc_screenshot_dict.keys()))
            # resetting ad_hoc_screenshot_dict
            self.ad_hoc_screenshot_dict = {}
        else:
            self.output_screenshots_impl(mcs=mcs, screenshot_label_list=list(self.screenshotDataDict.keys()))


class ScreenshotManagerCC3DPy(ScreenshotManagerCore):
    """
    Subclass with necessary hooks for Python API
    """
    def __init__(self, screenshot_dir_name):
        super().__init__()

        def get_screenshot_dir_name():
            return screenshot_dir_name
        self.get_screenshot_dir_name = get_screenshot_dir_name

    def cleanup(self):
        """
        Implementes cleanup actions
        :return: None
        """
        pass

    def safe_write_screenshot_description_file(self, out_fname):
        raise EnvironmentError

    def serialize_screenshot_data(self):
        raise EnvironmentError

    def add_2d_screenshot(self, _plotName, _plotType, _projection, _projectionPosition, _camera, metadata):
        raise EnvironmentError

    def add_3d_screenshot(self, _plotName, _plotType, _camera, metadata):
        raise EnvironmentError

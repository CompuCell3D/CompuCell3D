import os.path
from pathlib import Path
from typing import Union
from zipfile import ZipFile
from glob import glob
from PyQt5.QtWidgets import *
from os.path import *
import os


class Unzipper:
    def __init__(self, ui):
        self.__ui = ui

    def unzip_project(self, file_name_path: Path) -> Union[str, None]:
        """
        unzips project files and returns a path to .cc3d project in the unpacked folder
        :param file_name_path:
        :return:
        """

        proposed_dir = file_name_path.parent.joinpath(file_name_path.stem)
        unzip_dirname = self.find_available_dir_name(proposed_dir=proposed_dir)
        if not unzip_dirname:
            return

        if not Path(unzip_dirname).exists():
            unzip_dirname.mkdir(parents=True, exist_ok=True)
            QMessageBox.information(self.__ui, 'About to unzip CC3D project',
                                    f'Will unzip <b>.cc3d</b> project into <br> '
                                    f'<br> <i>{unzip_dirname}</i> <br>')

        dir_empty = self.check_dir_empty(directory=unzip_dirname)
        if not dir_empty:
            ret = QMessageBox.question(self.__ui, 'Directory not empty',
                                       f'The directory you selected '
                                       f'<br> <i>{unzip_dirname}</i> <br>'
                                       f'is not empty would you like to overwrite use it to '
                                       f'unpack <b>.cc3d</b> project? <br>'
                                       f'Warning: you may corrupt data in this directory',
                                       QMessageBox.Yes | QMessageBox.No
                                       )
            if ret == QMessageBox.No:
                return

        with ZipFile(file_name_path, 'r') as zip_file:
            zip_file.extractall(unzip_dirname)

        cc3d_file_path_glob = glob(join(str(unzip_dirname), '**/*.cc3d'), recursive=True)
        if len(cc3d_file_path_glob) != 1:
            QMessageBox.warning(self.__ui, 'Could not uniquely identify .cc3d project',
                                f'Could not uniquely identify a <b>.cc3d</b> project in the unzipped folder:  '
                                f'<br> <i>{unzip_dirname}</i> <br>'
                                f'Please use <b>CC3D Project -> Open CC3D project...</b> menu option to select '
                                f'which <b>.cc3d</b> project you want to open')
            return
        else:
            return cc3d_file_path_glob[0]

    @staticmethod
    def check_dir_empty(directory):
        """
        Checks if directory is empty or not. If directory dies not exists it returns True
        :param directory:
        :return:
        """
        if not Path(directory).exists():
            return True

        return len(os.listdir(directory)) == 0

    def find_available_dir_name(self, proposed_dir):
        """
        Returns available directory name - if proposed directory exists it asks user to create and pick different one
        :param proposed_dir:
        :return:
        """
        if not proposed_dir.exists():
            return proposed_dir

        QMessageBox.information(self.__ui, 'Select Folder For Unzipping .cc3d Project',
                                f'Could not create default unzip folder: '
                                f'<br> <i>{proposed_dir}</i> <br>'
                                f'(perhaps it already exists). <br>  '
                                f'Please select folder where you want to unzip .cc3d project '
                                )

        new_dirname = QFileDialog.getExistingDirectory(self.__ui, 'Create or Select a Directory '
                                                                  'to Extract Zip Archive to',
                                                       str(proposed_dir.parent))
        return new_dirname

from cc3d.twedit5.twedit.utils.global_imports import *
from glob import glob
from os.path import join, dirname

class TweditFileDialog(QFileDialog):
    def __init__(self, parent):
        super(TweditFileDialog, self).__init__(parent)
        self.extensions_to_look_for = set()
        self.directoryEntered.connect(self.scan_dir)
        self.default_file = None
        self.default_file_used = False

    def add_extension(self, ext):
        self.extensions_to_look_for.update([ext])

    def set_default_file(self, default_file:str):
        """

        :return:
        """

        self.default_file = default_file
        self.default_file_used = False

    def scan_dir(self, dir_name):
        print('entering_dir=', dir_name)
        if self.default_file and not self.default_file_used:
            self.setDirectory(dirname(self.default_file))
            self.default_file_used = True

        # self.selectFile(current_file_path)


        print('Entered Dir ', dir_name)
        extensions = sorted(list(self.extensions_to_look_for))
        # self.setNameFilter(';'.join(extensions))
        # self.setNameFilters(list(map(lambda ext_local: '*'+ext_local, extensions)))

        files_grabbed = []
        for ext in extensions:

            files_grabbed.extend(glob(join(dir_name, f'*{ext}')))

        files_grabbed = sorted(files_grabbed)

        try:
            file_to_select = abspath(files_grabbed[0])
            print('selecting', file_to_select)
            # self.fileSelected.emit(file_to_select)
            self.selectUrl(QUrl('xxx'))

            # self.selectFile(abspath(files_grabbed[0]))
            # self.selectFile(abspath(basename(files_grabbed[0])))

            # self.selectUrl(QUrl(os.path.basename(files_grabbed[0])))

        except IndexError:
            print('no.cc3d_file in ', dir_name)
            # self.selectFile()

        print ('seleced files=', self.selectedFiles())
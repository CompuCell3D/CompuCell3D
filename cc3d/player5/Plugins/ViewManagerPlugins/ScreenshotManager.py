# -*- coding: utf-8 -*-
import warnings
import os, sys
from os.path import join, exists, dirname
import string
from cc3d.core.utils import mkdir_p
import Configuration
from Utilities import ScreenshotData, ScreenshotManagerCore
from cc3d.player5.GraphicsOffScreen.GenericDrawer import GenericDrawer


class ScreenshotManager(ScreenshotManagerCore):
    def __init__(self, _tabViewWidget):
        ScreenshotManagerCore.__init__(self)

        from weakref import ref
        self.tabViewWidget = ref(_tabViewWidget)
        tvw = self.tabViewWidget()

        self.basicSimulationData = tvw.basicSimulationData
        self.basicSimulationData = tvw.basicSimulationData
        self.screenshot_number_of_digits = len(str(self.basicSimulationData.numberOfSteps))

        self.maxNumberOfScreenshots = 20  # we limit max number of screenshots to discourage users from using screenshots as their main analysis tool

        self.screenshotGraphicsWidget = None

        self.gd = GenericDrawer()
        self.gd.set_field_extractor(field_extractor=tvw.fieldExtractor)

    def cleanup(self):
        # have to do cleanup to ensure some of the memory intensive resources e.g. self.screenshotGraphicsWidget get deallocated
        if self.screenshotGraphicsWidget:
            print('JUST BEFORE CLOSING self.screenshotGraphicsWidget')
            # this close and assignment do not do much for the non-mdi layout
            self.screenshotGraphicsWidget.close()
            self.screenshotGraphicsWidget = None
        self.tabViewWidget = None
        self.basicSimulationData = None

    def safe_writeScreenshotDescriptionFile(self, out_fname):
        """
        writes screenshot descr file in a safe mode. any problems are reported via warning
        :param out_fname: {str}
        :return: None
        """
        mkdir_p(dirname(out_fname))
        self.writeScreenshotDescriptionFile(out_fname)

    def serialize_screenshot_data(self):
        """
        Method called immediately after we add new screenshot via camera button. It serializes all screenshots data
        for future reference/reuse
        :return: None
        """

        tvw = self.tabViewWidget()

        out_dir_name = tvw.getOutputDirName()
        sim_fname = tvw.getSimFileName()

        out_fname = join(out_dir_name, 'screenshot_data', 'screenshots.json')
        out_fname_in_sim_dir = join(dirname(sim_fname), 'screenshot_data', 'screenshots.json')

        # writing in the simulation output dir
        self.safe_writeScreenshotDescriptionFile(out_fname)

        # writing in the original simulation location
        self.safe_writeScreenshotDescriptionFile(out_fname_in_sim_dir)

    def store_gui_vis_config(self, scrData):
        """
        Stores visualization settings such as cell borders, on/or cell on/off etc...

        :param scrData: {instance of ScreenshotDescriptionData}
        :return: None
        """

        tvw = self.tabViewWidget()
        if tvw:
            tvw.updateActiveWindowVisFlags(self.screenshotGraphicsWidget)

        scrData.cell_borders_on = tvw.borderAct.isChecked()
        scrData.cells_on = tvw.cellsAct.isChecked()
        scrData.cluster_borders_on = tvw.clusterBorderAct.isChecked()
        scrData.cell_glyphs_on = tvw.cellGlyphsAct.isChecked()
        scrData.fpp_links_on = tvw.FPPLinksAct.isChecked()
        scrData.lattice_axes_on = Configuration.getSetting('ShowHorizontalAxesLabels') or Configuration.getSetting(
            'ShowVerticalAxesLabels')
        scrData.lattice_axes_labels_on = Configuration.getSetting("ShowAxes")
        scrData.bounding_box_on = Configuration.getSetting("BoundingBoxOn")

        invisible_types = Configuration.getSetting("Types3DInvisible")
        invisible_types = invisible_types.strip()

        if invisible_types:
            scrData.invisible_types = list([int(x) for x in invisible_types.split(',')])
        else:
            scrData.invisible_types = []

    # called from GraphicsFrameWidget
    def add2DScreenshot(self, _plotName, _plotType, _projection, _projectionPosition,
                        _camera, metadata=None):
        if len(self.screenshotDataDict) > self.maxNumberOfScreenshots:
            print("MAX NUMBER OF SCREENSHOTS HAS BEEN REACHED")

        scrData = ScreenshotData()
        scrData.spaceDimension = "2D"
        scrData.plotData = (_plotName, _plotType)

        x_size = Configuration.getSetting("Screenshot_X")
        y_size = Configuration.getSetting("Screenshot_Y")

        scrData.projection = _projection
        scrData.projectionPosition = int(_projectionPosition)

        #        import pdb; pdb.set_trace()

        (scrName, scrCoreName) = self.produceScreenshotName(scrData)

        print("  add2DScreenshot():  THIS IS NEW SCRSHOT NAME", scrName)  # e.g. Cell_Field_CellField_2D_XY_150

        if not scrName in self.screenshotDataDict:
            scrData.screenshotName = scrName
            scrData.screenshotCoreName = scrCoreName
            scrData.screenshotGraphicsWidget = self.screenshotGraphicsWidget  # = GraphicsFrameWidget (rf. __init__)

            scrData.win_width = x_size
            scrData.win_height = y_size

            if metadata is not None:
                scrData.metadata = metadata

            tvw = self.tabViewWidget()
            if tvw:
                tvw.updateActiveWindowVisFlags(self.screenshotGraphicsWidget)

            self.store_gui_vis_config(scrData=scrData)
            scrData.extractCameraInfo(_camera)  # so "camera" icon (save images) remembers camera view

            # on linux there is a problem with X-server/Qt/QVTK implementation and calling resize right after additional QVTK
            # is created causes segfault so possible "solution" is to do resize right before taking screenshot.
            # It causes flicker but does not cause segfault.
            # User should NOT close or minimize this "empty" window (on Linux anyway).
            if sys.platform == 'Linux' or sys.platform == 'linux' or sys.platform == 'linux2':
                #                pass
                self.screenshotDataDict[scrData.screenshotName] = scrData
            else:
                self.screenshotDataDict[scrData.screenshotName] = scrData
        else:
            print("Screenshot ", scrName, " already exists")

        # serializing all screenshots
        self.serialize_screenshot_data()

    def add3DScreenshot(self, _plotName, _plotType, _camera, metadata=None):  # called from GraphicsFrameWidget
        if len(self.screenshotDataDict) > self.maxNumberOfScreenshots:
            print("MAX NUMBER OF SCREENSHOTS HAS BEEN REACHED")
        scrData = ScreenshotData()
        scrData.spaceDimension = "3D"
        scrData.plotData = (_plotName, _plotType)

        x_size = Configuration.getSetting("Screenshot_X")
        y_size = Configuration.getSetting("Screenshot_Y")

        (scrName, scrCoreName) = self.produceScreenshotName(scrData)

        okToAddScreenshot = True
        for name in self.screenshotDataDict:
            scrDataFromDict = self.screenshotDataDict[name]
            if scrDataFromDict.screenshotCoreName == scrCoreName and scrDataFromDict.spaceDimension == "3D":
                if scrDataFromDict.compareCameras(_camera):
                    print("CAMERAS ARE THE SAME")
                    okToAddScreenshot = False
                    break
                else:
                    print("CAMERAS ARE DIFFERENT")

        if (not scrName in self.screenshotDataDict) and okToAddScreenshot:
            scrData.screenshotName = scrName
            scrData.screenshotCoreName = scrCoreName
            scrData.screenshotGraphicsWidget = self.screenshotGraphicsWidget

            scrData.win_width = x_size
            scrData.win_height = y_size

            if metadata is not None:
                scrData.metadata = metadata

            tvw = self.tabViewWidget()
            if tvw:
                tvw.updateActiveWindowVisFlags(self.screenshotGraphicsWidget)

            self.store_gui_vis_config(scrData=scrData)

            scrData.extractCameraInfo(_camera)

            # on linux there is a problem with X-server/Qt/QVTK implementation and calling resize right after additional QVTK
            # is created causes segfault so possible "solution" is to do resize right before taking screenshot.
            # It causes flicker but does not cause segfault
            # User should NOT close or minimize this "empty" window (on Linux anyway).
            if sys.platform == 'Linux' or sys.platform == 'linux' or sys.platform == 'linux2':
                self.screenshotDataDict[scrData.screenshotName] = scrData
                self.screenshotCounter3D += 1
            else:
                self.screenshotDataDict[scrData.screenshotName] = scrData
                self.screenshotCounter3D += 1
        else:
            print("Screenshot ", scrCoreName, " with current camera settings already exists. " \
                                              "You need to rotate camera i.e. rotate picture " \
                                              "using mouse to take additional screenshot")

        # serializing all screenshots
        self.serialize_screenshot_data()

    def get_basic_simulation_data(self):
        """
        Returns an instance of BasicSimulationData
        :return: {BasicSimulationData}
        """
        tvw = self.tabViewWidget()
        bsd = tvw.basicSimulationData
        return bsd


    # called from SimpleTabView:handleCompletedStep{Regular,CML*}
    def outputScreenshots(self, general_screenshot_directory_name, mcs):

        # tvw = self.tabViewWidget()
        #         # bsd = tvw.basicSimulationData

        bsd = self.get_basic_simulation_data()

        # fills string with 0's up to self.screenshotNumberOfDigits width
        mcsFormattedNumber = string.zfill(str(mcs), self.screenshot_number_of_digits)

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

import sys
import re
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cc3d.player5.Configuration as Configuration
from cc3d.player5 import DefaultData
import cc3d
import datetime
from cc3d.player5.Utilities.WebFetcher import WebFetcher

gip = DefaultData.getIconPath

MODULENAME = '------- SimpleViewManager: '


class SimpleViewManager(QObject):

    def __init__(self, ui):
        QObject.__init__(self)
        self.visual = {
            "CellsOn": Configuration.getSetting("CellsOn"),
            "CellBordersOn": Configuration.getSetting("CellBordersOn"),
            "ClusterBordersOn": Configuration.getSetting("ClusterBordersOn"),
            "CellGlyphsOn": Configuration.getSetting("CellGlyphsOn"),
            "FPPLinksOn": Configuration.getSetting("FPPLinksOn"),
            "CC3DOutputOn": Configuration.getSetting("CC3DOutputOn"),
            "ConcentrationLimitsOn": Configuration.getSetting("ConcentrationLimitsOn"),
            "ZoomFactor": Configuration.getSetting("ZoomFactor"),
        }

        # file actions
        self.open_act = None
        self.open_lds_act = None
        self.exit_act = None
        self.twedit_act = None

        # Simulation actions
        self.run_act = None
        self.step_act = None
        self.pause_act = None
        self.stop_act = None
        self.restore_default_settings_act = None

        # visualization actions
        self.cells_act = None
        self.border_act = None
        self.cluster_border_act = None
        self.cell_glyphs_act = None
        self.fpp_links_act = None
        self.limits_act = None
        self.cc3d_output_on_act = None
        self.reset_camera_act = None
        self.zoom_in_act = None
        self.zoom_out_act = None

        # tools Actions
        self.config_act = None
        self.pif_from_vtk_act = None
        self.pif_from_simulation_act = None
        self.restart_snapshot_from_simulation_act = None

        # windows actions
        self.python_steering_panel_act = None
        self.new_graphics_window_act = None
        self.tile_act = None
        self.cascade_act = None
        self.minimize_all_graphics_windows_act = None
        self.restore_all_graphics_windows_act = None
        self.close_active_window_act = None

        # help actions
        self.quick_act = None
        self.tutor_act = None
        self.ref_man_act = None
        self.about_act = None
        self.mail_subscribe_act = None
        self.mail_unsubscribe_act = None
        self.mail_subscribe_unsubscribe_web_act = None
        self.check_update_act = None
        self.whats_this_act = None

        self.display_no_update_info = False

        self.init_actions()
        self.ui = ui

    def init_file_menu(self):
        """
        Initializes file menu
        :return:
        """
        menu = QMenu(QApplication.translate('ViewManager', '&File'), self.ui)
        menu.addAction(self.open_act)

        # LDS lattice description summary  - xml file that specifies what simulation data has been written to the disk
        menu.addAction(self.open_lds_act)
        menu.addSeparator()
        menu.addAction(self.twedit_act)
        menu.addSeparator()
        recent_simulations_menu = menu.addMenu("Recent Simulations...")
        menu.addSeparator()
        menu.addAction(self.exit_act)

        return menu, recent_simulations_menu

    def init_sim_menu(self):
        """
        Initializes simulation menu

        :return:
        """
        menu = QMenu(QApplication.translate('ViewManager', '&Simulation'), self.ui)
        menu.addAction(self.run_act)
        menu.addAction(self.step_act)
        menu.addAction(self.pause_act)
        menu.addAction(self.stop_act)

        menu.addSeparator()
        # --------------------
        menu.addAction(self.restore_default_settings_act)

        return menu

    def init_visual_menu(self):
        """
        Initializes Visualization menu
        :return:
        """
        menu = QMenu(QApplication.translate('ViewManager', '&Visualization'), self.ui)
        menu.addAction(self.cells_act)
        menu.addAction(self.border_act)
        menu.addAction(self.cluster_border_act)
        menu.addAction(self.cell_glyphs_act)
        menu.addAction(self.fpp_links_act)
        menu.addAction(self.limits_act)
        menu.addSeparator()
        menu.addAction(self.cc3d_output_on_act)
        menu.addSeparator()
        menu.addAction(self.reset_camera_act)
        menu.addAction(self.zoom_in_act)
        menu.addAction(self.zoom_out_act)

        return menu

    def init_tools_menu(self):
        """
        Initializes Tools menu
        :return:
        """
        menu = QMenu(QApplication.translate('ViewManager', '&Tools'), self.ui)
        menu.addSeparator()
        menu.addAction(self.config_act)

        menu.addAction(self.pif_from_simulation_act)
        self.pif_from_simulation_act.setEnabled(False)

        menu.addAction(self.pif_from_vtk_act)
        self.pif_from_vtk_act.setEnabled(False)

        menu.addAction(self.restart_snapshot_from_simulation_act)
        self.restart_snapshot_from_simulation_act.setEnabled(False)

        return menu

    def init_window_menu(self):
        """
        Imiplements Windows Menu
        :return:
        """
        menu = QMenu(QApplication.translate('ViewManager', '&Window'), self.ui)

        # NOTE initialization of the menu is done in the updateWindowMenu function in SimpleTabView

        return menu

    def init_help_menu(self):
        """
        Implements Help Menu
        :return:
        """

        menu = QMenu(QApplication.translate('ViewManager', '&Help'), self.ui)
        menu.addAction(self.quick_act)
        menu.addAction(self.tutor_act)
        menu.addAction(self.ref_man_act)
        menu.addSeparator()
        menu.addAction(self.mail_subscribe_act)
        menu.addAction(self.mail_unsubscribe_act)
        menu.addAction(self.mail_subscribe_unsubscribe_web_act)
        menu.addSeparator()
        menu.addAction(self.check_update_act)
        menu.addSeparator()
        menu.addAction(self.about_act)
        menu.addSeparator()
        menu.addAction(self.whats_this_act)

        return menu

    def init_file_toolbar(self):
        """
        Initializes file toolbar
        :return:
        """
        tb = QToolBar(QApplication.translate('ViewManager', 'File'), self.ui)
        # UI.Config.ToolBarIconSize
        tb.setIconSize(QSize(20, 18))
        tb.setObjectName("FileToolbar")
        tb.setToolTip(QApplication.translate('ViewManager', 'File'))

        tb.addAction(self.open_act)
        tb.addAction(self.config_act)
        tb.addAction(self.twedit_act)

        return tb

    def init_visualization_toolbar(self):
        """
        Initializes Visualization toolbar
        :return:
        """
        tb = QToolBar(QApplication.translate('Visualization', 'Visualization'), self.ui)
        # UI.Config.ToolBarIconSize
        tb.setIconSize(QSize(20, 18))
        tb.setObjectName("VisualizationToolbar")
        tb.setToolTip(QApplication.translate('ViewManager', 'Visualization'))

        tb.addAction(self.zoom_in_act)
        tb.addAction(self.zoom_out_act)

        return tb

    def init_sim_toolbar(self):
        """
        Initializes Simulation toolbat
        :return:
        """
        tb = QToolBar(QApplication.translate('ViewManager', 'Simulation'), self.ui)
        # UI.Config.ToolBarIconSize
        tb.setIconSize(QSize(20, 18))
        tb.setObjectName("SimToolbar")
        tb.setToolTip(QApplication.translate('ViewManager', 'Simulation'))

        tb.addAction(self.run_act)
        tb.addAction(self.step_act)
        tb.addAction(self.pause_act)
        tb.addAction(self.stop_act)

        return tb

    def init_window_toolbar(self):
        """
        Initializes Window toolbar
        :return:
        """
        wtb = QToolBar(QApplication.translate('ViewManager', 'Window'), self.ui)
        # UI.Config.ToolBarIconSize
        wtb.setIconSize(QSize(20, 18))
        wtb.setObjectName("WindowToolbar")
        wtb.setToolTip(QApplication.translate('ViewManager', 'Window'))

        wtb.addAction(self.new_graphics_window_act)

        return wtb

    def init_actions(self):
        """
        Initializes actions
        :return:
        """
        # list containing all file actions

        self.init_window_actions()
        self.init_file_actions()
        self.init_sim_actions()
        self.init_visual_actions()
        self.init_tools_actions()
        self.init_help_actions()

    def init_file_actions(self):
        """
        This function does the following:

        - Create Action -- act = QAction()
        - Set status tip -- act.setStatusTip()
        - Set what's this -- act.setWhatsThis()
        - Connect signals -- self.connect(act, ...)
        - Add to the action list - actList.append(act)

        :return:
        """

        self.open_act = QAction(QIcon(gip("fileopen.png")), "&Open Simulation File (.cc3d)", self)
        self.open_act.setShortcut(Qt.CTRL + Qt.Key_O)

        self.open_lds_act = QAction(QIcon(gip("screenshots_open.png")), "&Open Lattice Description Summary File...",
                                    self)

        self.exit_act = QAction(QIcon(gip("exit2.png")), "&Exit", self)

        self.twedit_act = QAction(QIcon(gip("twedit-icon.png")), "Start Twe&dit++", self)

    def init_sim_actions(self):
        """
        initializes Simulaiton actions
        :return:
        """

        gip = DefaultData.getIconPath

        self.run_act = QAction(QIcon(gip("play.png")), "&Run", self)
        self.run_act.setShortcut(Qt.CTRL + Qt.Key_M)
        self.step_act = QAction(QIcon(gip("step.png")), "&Step", self)
        self.step_act.setShortcut(Qt.CTRL + Qt.Key_E)
        self.pause_act = QAction(QIcon(gip("pause.png")), "&Pause", self)
        self.pause_act.setShortcut(Qt.CTRL + Qt.Key_D)
        self.stop_act = QAction(QIcon(gip("stop.png")), "&Stop", self)
        self.stop_act.setShortcut(Qt.CTRL + Qt.Key_X)

        self.restore_default_settings_act = QAction("Restore Default Settings", self)

    def init_visual_actions(self):
        """
        Initializes Visualization actions
        :return:
        """
        self.cells_act = QAction("&Cells", self)
        self.cells_act.setCheckable(True)
        self.cells_act.setChecked(self.visual["CellsOn"])

        self.border_act = QAction("Cell &Borders", self)
        self.border_act.setCheckable(True)
        self.border_act.setChecked(self.visual["CellBordersOn"])

        self.cluster_border_act = QAction("Cluster Borders", self)
        self.cluster_border_act.setCheckable(True)
        self.cluster_border_act.setChecked(self.visual["ClusterBordersOn"])

        self.cell_glyphs_act = QAction("Cell &Glyphs", self)
        self.cell_glyphs_act.setCheckable(True)
        self.cell_glyphs_act.setChecked(self.visual["CellGlyphsOn"])

        self.fpp_links_act = QAction("&FPP Links", self)  # callbacks for these menu items in child class SimpleTabView
        self.fpp_links_act.setCheckable(True)
        self.fpp_links_act.setChecked(self.visual["FPPLinksOn"])

        self.limits_act = QAction("Concentration &Limits", self)
        self.limits_act.setCheckable(True)
        self.limits_act.setChecked(self.visual["ConcentrationLimitsOn"])

        self.cc3d_output_on_act = QAction("&Turn On CompuCell3D Output", self)
        self.cc3d_output_on_act.setCheckable(True)
        self.cc3d_output_on_act.setChecked(self.visual["CC3DOutputOn"])

        self.reset_camera_act = QAction("Reset Camera for Graphics Window ('r')", self)

        self.zoom_in_act = QAction(QIcon(gip("zoomIn.png")), "&Zoom In", self)
        self.zoom_in_act.setShortcut(Qt.CTRL + Qt.Key_Y)
        self.zoom_out_act = QAction(QIcon(gip("zoomOut.png")), "&Zoom Out", self)

    def init_tools_actions(self):
        """
        initializes tools actions
        :return:
        """
        self.config_act = QAction(QIcon(gip("config.png")), "&Configuration...", self)

        self.config_act.setShortcut(Qt.CTRL + Qt.Key_Comma)

        self.config_act.setWhatsThis(
            """<b>Configuration</b>"""
            """<p>Set the configuration items of the simulation"""
            """ with your prefered values.</p>"""
        )

        self.pif_from_vtk_act = QAction("& Generate PIF File from VTK output ...", self)

        self.pif_from_simulation_act = QAction("& Generate PIF File from current snapshot ...", self)
        self.restart_snapshot_from_simulation_act = QAction("& Generate Restart Snapshot", self)

        self.config_act.setWhatsThis(
            """<b>Generate PIF file from current simulation snapshot </b>"""
        )

    def init_window_actions(self):
        """
        initializes Window Actions
        :return:
        """

        self.python_steering_panel_act = QAction("Steering Panel", self)
        self.python_steering_panel_act.setShortcut(self.tr("Ctrl+U"))

        self.new_graphics_window_act = QAction(QIcon(gip("kcmkwm.png")), "&New Graphics Window", self)
        self.new_graphics_window_act.setShortcut(self.tr("Ctrl+I"))

        self.tile_act = QAction("Tile", self)
        self.cascade_act = QAction("Cascade", self)

        self.minimize_all_graphics_windows_act = QAction("Minimize All Graphics Windows", self)

        self.minimize_all_graphics_windows_act.setShortcut(self.tr("Ctrl+Alt+M"))

        self.restore_all_graphics_windows_act = QAction("Restore All Graphics Windows", self)
        self.restore_all_graphics_windows_act.setShortcut(self.tr("Ctrl+Alt+N"))

        self.close_active_window_act = QAction("Close Active Window", self)
        self.close_active_window_act.setShortcut(self.tr("Ctrl+F4"))

    def init_help_actions(self):
        """
        initializes Help Actions
        :return:
        """

        self.quick_act = QAction("&Quick Start", self)
        self.quick_act.triggered.connect(self.__open_manuals_webpage)
        self.tutor_act = QAction("&Tutorials", self)
        self.tutor_act.triggered.connect(self.__open_manuals_webpage)
        self.ref_man_act = QAction(QIcon(gip("man.png")), "&Reference Manual", self)
        self.ref_man_act.triggered.connect(self.__open_manuals_webpage)
        self.about_act = QAction(QIcon(gip("cc3d_64x64_logo.png")), "&About CompuCell3D", self)
        self.about_act.triggered.connect(self.__about)
        self.mail_subscribe_act = QAction(QIcon(gip("email-at-sign-icon.png")), "Subscribe to Mailing List", self)
        self.mail_subscribe_act.triggered.connect(self.__mail_subscribe)

        self.mail_unsubscribe_act = QAction(QIcon(gip("email-at-sign-icon-unsubscribe.png")),
                                            "Unsubscribe from Mailing List", self)
        self.mail_unsubscribe_act.triggered.connect(self.__mail_unsubscribe)

        self.mail_subscribe_unsubscribe_web_act = QAction("Subscribe/Unsubscribe Mailing List - Web browser", self)
        self.mail_subscribe_unsubscribe_web_act.triggered.connect(
            self.__mail_subscribe_unsubscribe_web)

        self.check_update_act = QAction("Check for CC3D Updates", self)
        self.check_update_act.triggered.connect(self.__check_update)
        self.display_no_update_info = False

        self.whats_this_act = QAction(QIcon(gip("whatsThis.png")), "&What's This?", self)
        self.whats_this_act.setWhatsThis(
            """<b>Display context sensitive help</b>"""
            """<p>In What's This? mode, the mouse cursor shows an arrow with a question"""
            """ mark, and you can click on the interface elements to get a short"""
            """ description of what they do and how to use them. In dialogs, this"""
            """ feature can be accessed using the context help button in the"""
            """ titlebar.</p>"""
        )
        self.whats_this_act.triggered.connect(self.__whatsThis)

    def check_version(self, check_interval=-1, display_no_update_info=False):
        """
        This function checks if new CC3D version is available
        :return:None
        """

        # here we decide whether the information about no new updates is displayed or not. For automatic update checks
        # this information should not be displayed. For manual update checks we need to inform the user
        # that there are no updates

        self.display_no_update_info = display_no_update_info

        # determine if check is necessary - for now we check every week in order not to bother users with too many checks
        last_version_check_date = Configuration.getSetting('LastVersionCheckDate')

        today = datetime.date.today()
        today_date_str = today.strftime('%Y%m%d')

        old_date = datetime.date(int(last_version_check_date[:4]), int(last_version_check_date[4:6]),
                                 int(last_version_check_date[6:]))
        t_delta = today - old_date

        if t_delta.days < check_interval:
            # check for CC3D recently
            return
        else:
            print('WILL DO THE CHECK')

        version_fetcher = WebFetcher(_parent=self)
        version_fetcher.gotWebContentSignal.connect(self.process_version_check)

        version_fetcher.fetch("http://www.compucell3d.org/current_version")

    def process_version_check(self, version_str, url_str):
        """
        This function extracts current version and revision numbers from the http://www.compucell3d.org/current_version
        It informs users that new version is available and allows easy redirection to the download site
        :param version_str: content of the web page with the current version information
        :param url_str: url of the webpage with the current version information
        :return: None
        """
        if str(version_str) == '':
            print('Could not fetch "http://www.compucell3d.org/current_version webpage')
            return

        current_version = ''
        current_revision = ''
        whats_new_list = []

        current_version_regex = re.compile("(current version)([0-9\. ]*)")

        # (.*?)(<) ensures non-greedy match i.e. all the characters will be matched until first occurrence of '<'
        whats_new_regex = re.compile("(>[\S]*what is new:)(.*?)(<)")

        for line in str(version_str).split("\n"):

            search_obj = re.search(current_version_regex, line)
            search_obj_whats_new = re.search(whats_new_regex, line)

            if search_obj:
                # print 'search_obj=', search_obj
                # print search_obj.groups()
                try:
                    version_info = search_obj.groups()[1]
                    version_info = version_info.strip()
                    current_version, current_revision = version_info.split(' ')
                except:
                    pass

            if search_obj_whats_new:
                # print search_obj_whats_new.groups()
                try:
                    whats_new = search_obj_whats_new.groups()[1]
                    whats_new = whats_new.strip()
                    whats_new_list = whats_new.split(', ')
                except:
                    pass

        instance_version = cc3d.__version__
        instance_revision = cc3d.__revision__
        try:
            current_version_number = int(current_version.replace('.', ''))
        except:
            # this can happen when the page gets "decorated" by e.g. your hotel network
            # will have to come up with a better way of dealing with it
            return
        current_revision_number = int(current_revision)
        instance_version_number = int(instance_version.replace('.', ''))
        instance_revision_number = int(instance_revision)

        display_new_version_info = False

        if current_version_number > instance_version_number:
            display_new_version_info = True

        elif current_version_number == instance_version_number and current_revision_number > instance_revision_number:
            display_new_version_info = True

        today = datetime.date.today()
        today_date_str = today.strftime('%Y%m%d')

        last_version_check_date = Configuration.setSetting('LastVersionCheckDate', today_date_str)

        message = 'New version of CompuCell3D is available - %s rev. %s. Would you like to upgrade?' % (
            current_version, current_revision)

        if len(whats_new_list):
            message += '<p><b>New Features:</b></p>'
            for whats_new_item in whats_new_list:
                message += '<p> * ' + whats_new_item + '</p>'

        if display_new_version_info:

            ret = QMessageBox.information(self, 'New Version Available', message, QMessageBox.Yes | QMessageBox.No)
            if ret == QMessageBox.Yes:
                QDesktopServices.openUrl(QUrl('http://sourceforge.net/projects/cc3d/files/' + current_version))

        elif self.display_no_update_info == True:
            ret = QMessageBox.information(self, 'Software update check', 'You are running latest version of CC3D.',
                                          QMessageBox.Ok)

    def __check_update(self):
        """
        This slot checks for CC3D updates
        :return:None
        """

        self.check_version(check_interval=-1, display_no_update_info=True)

    def __open_manuals_webpage(self):
        # print 'THIS IS QUICK START GUIDE'
        QDesktopServices.openUrl(QUrl('http://www.compucell3d.org/Manuals'))

    def __mail_subscribe(self):
        QDesktopServices.openUrl(QUrl('mailto:list@iu.edu?body=SUBSCRIBE compucell3d-l'))

    def __mail_unsubscribe(self):
        QDesktopServices.openUrl(QUrl('mailto:list@iu.edu?body=UNSUBSCRIBE compucell3d-l'))

    def __mail_subscribe_unsubscribe_web(self):
        QDesktopServices.openUrl(QUrl('http://www.compucell3d.org/mailinglist'))

    def __about(self):
        version_str = '4.0.0'
        revision_str = '0'

        try:
            version_str = cc3d.__version__
            revision_str = cc3d.__revision__
        except ImportError :
            pass

        # import Configuration
        # version_str=Configuration.getVersion()
        about_text = "<h2>CompuCell3D</h2> Version: " + version_str + " Revision: " + revision_str + "<br />\
                          Copyright &copy; Biocomplexity Institute, <br />\
                          Indiana University, Bloomington, IN\
                          <p><b>CompuCell Player</b> is a visualization engine for CompuCell.</p>"
        more_info_text = "More information " \
                         "at:<br><a href=\"http://www.compucell3d.org/\">http://www.compucell3d.org/</a>"

        l_version_string = "<br><br><small><small>Support library information:<br>Python runtime version: " \
                           "%s<br>Qt runtime version: %s<br>Qt compile-time version: " \
                           "%s<br>PyQt version: %s</small></small>" % \
                           (str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "." + str(
                               sys.version_info[2]) + " - " + str(sys.version_info[3]) + " - " + str(
                               sys.version_info[4]),
                            qVersion(), QT_VERSION_STR, PYQT_VERSION_STR)

        QMessageBox.about(self, "CompuCell3D", about_text + more_info_text + l_version_string)

    def __whatsThis(self):
        """
        Private slot called in to enter Whats This mode.
        """
        QWhatsThis.enterWhatsThisMode()

    def __TBMenuTriggered(self, act):
        """
        Private method to handle the toggle of a toolbar.
        
        @param act reference to the action that was triggered (QAction)
        """

        name = str(act.data().toString())
        if name:
            tb = self.__toolbars[name][1]
            if act.isChecked():
                tb.show()
            else:
                tb.hide()

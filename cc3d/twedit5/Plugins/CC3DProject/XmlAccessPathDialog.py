from cc3d.twedit5.twedit.utils.global_imports import *

from . import ui_xmlaccesspathdlg

from cc3d.core.ParameterScanEnums import *

from collections import namedtuple

XmlAccessPathTuple = namedtuple('XmlAccessPathTuple', ['type', 'name', 'access_path'])

MAC = "qt_mac_set_native_menubar" in dir()


class TablePushButton(QPushButton):

    def __init__(self, _parent=None):
        super(TablePushButton, self).__init__(_parent)

        self.row = -1

        self.col = -1

    def setPosition(self, _row, _col):
        self.row = _row

        self.col = _col

    def getPosition(self):
        return self.row, self.col


class XmlAccessPathDialog(QDialog, ui_xmlaccesspathdlg.Ui_XMLAccessPathDlg):

    # signals

    # gotolineSignal = QtCore.pyqtSignal( ('int',))

    def __init__(self, parent=None):

        super(XmlAccessPathDialog, self).__init__(parent)

        self.fileType = 'XML'

        # self.scannableParams = {}

        self.xmlString = ''

        self.accessPath = ''

        # self.parameterScanXMLElements = {}

        # self.parameterScanDataMap = {}

        # todo - move to separate dialog

        # tuple holds xml component type and  precise access path to the CDATA or select attribute of the XML element

        self.precise_xml_access_path_tuple = XmlAccessPathTuple(None, None, None)

        # self.scannedFileName='' # name of the file being scanned - can be either XML , or Python

        self.setupUi(self)

        self.updateUi()

    def setFileType(self, _type):

        self.fileType = _type

    # def __handleActionClicked(self):

    #     senderBtn = self.sender()

    #     row, col = senderBtn.getPosition()

    #     print 'clicked row,column=', (row, col)

    #     from ParameterScanUtils import ParameterScanData

    #

    #     nameItem = self.paramTW.item(row, PARAMETER)

    #

    #     valueItem = self.paramTW.item(row, VALUE)

    #

    #     # value=float(valueItem.text())

    #     value = str(valueItem.text())

    #

    #     print 'TYPE=', TYPE

    #     typeItem = self.paramTW.item(row, TYPE)

    #     type = TYPE_DICT_REVERSE[str(typeItem.text())]

    #     print 'type=', type, '\n\n\n\n'

    #

    #     psd = ParameterScanData()

    #

    #     psd.name = str(nameItem.text())

    #     psd.type = type

    #     psd.accessPath = self.accessPath

    #

    #     # show param scan values generation dialog

    #     from ParValDlg import ParValDlg

    #     parvaldlg = ParValDlg(self)

    #     parvaldlg.initParameterScanData(_parValue=value, _parName=psd.name, _parType=psd.type,

    #                                     _parAccessPath=psd.accessPath)

    #     # parvaldlg.setAutoMinMax(value)

    #

    #     if parvaldlg.exec_():

    #         valueStr = str()

    #         try:

    #             psd.customValues = parvaldlg.getValues()

    #         except ValueError, e:

    #             QMessageBox.warning(self, "Error Parsing Parameter List",

    #                                 "Please make sure that parameter list entries have correct type")

    #             return

    #         psd.valueType = parvaldlg.getValueType()

    #         # VALUE_TYPE_DICT_REVERSE[parvaldlg.getValueType()]

    #     else:

    #         # user canceled

    #         return

    #

    #     el = psd.toXMLElem()

    #

    #     from ParameterScanUtils import XMLHandler

    #     xmlHandler = XMLHandler()

    #     xmlHandler.writeXMLElement(

    #         el.CC3DXMLElement)  # because ElementCC3D was constructed in Python we need to get C++ object (CC3DXMLElement) from it this is what writeXMLElement expects

    #     self.parameterScanDataMap[psd.stringHash()] = psd

    #

    #     # print 'xmlElem=',xmlHandler.xmlString

    # def displayXMLScannableParameters(self, _elem, _accessPath, _parameterScanFile):

    #

    #     self.accessPath = _accessPath

    #

    #     from ParameterScanUtils import ParameterScanUtils

    #

    #     psu = ParameterScanUtils()

    #     print  'xmlElem=', _elem

    #

    #     self.xmlString = '<' + _elem.name + ' '

    #     for key in _elem.attributes.keys():

    #         self.xmlString += ' ' + key + '="' + _elem.attributes[key] + '"'

    #

    #     self.xmlString += '>'

    #     self.elemLE.setText(self.xmlString)

    #

    #     self.scannableParams = psu.extractXMLScannableParameters(_elem, _parameterScanFile)

    #

    #     table = self.paramTW

    #

    #     for paramName, paramProps in self.scannableParams.iteritems():

    #         currentRow = table.rowCount()

    #         # if table.rowCount()>0 else 0

    #         print 'currentRow=', currentRow, 'paramName=', paramName

    #         table.insertRow(currentRow)

    #         paramNameItem = QTableWidgetItem(paramName)

    #         paramValueItem = QTableWidgetItem(paramProps[0])

    #

    #         paramTypeItem = QTableWidgetItem(TYPE_DICT[paramProps[1]])

    #

    #         btn = TablePushButton(table)

    #

    #         actionItem = None

    #         btn.setText('Edit...')

    #         # if paramProps[1]==0:

    #         # btn.setText('Add To Scan...')

    #         # # actionItem=QTableWidgetItem('Add To Scan')

    #         # elif paramProps[1]==1:

    #         # btn.setText('View/Edit...')

    #         # # actionItem=QTableWidgetItem('View/Edit...')

    #

    #         table.setItem(currentRow, PARAMETER, paramNameItem)

    #         table.setItem(currentRow, VALUE, paramValueItem)

    #         table.setItem(currentRow, TYPE, paramTypeItem)

    #

    #         table.setCellWidget(currentRow, ACTION, btn)

    #         btn.setPosition(currentRow, ACTION)

    #         btn.clicked.connect(self.__handleActionClicked)

    #

    def display_xml_attributes(self, _elem, _accessPath, handle_xml_access_callback=None):

        self.accessPath = _accessPath

        self.handle_xml_access_callback = handle_xml_access_callback

        # print('xmlElem=', _elem)

        self.xmlString = '<' + _elem.name + ' '

        for key in list(_elem.attributes.keys()):
            self.xmlString += ' ' + key + '="' + _elem.attributes[key] + '"'

        self.xmlString += '>'

        self.elemLE.setText(self.xmlString)

        # self.scannableParams = psu.extractXMLScannableParameters(_elem, _parameterScanFile)

        params = {}

        # # print '_elem=',_elem.name

        # # print '_elem.attributes.size()=',_elem.attributes

        if _elem.attributes.size():

            for key in list(_elem.attributes.keys()):

                try:  # checki if attribute can be converted to floating point value - if so it can be added to scannable parameters

                    print('_elem.attributes[key]=', _elem.attributes[key])

                    float(_elem.attributes[key])

                    params[key] = [_elem.attributes[key], XML_ATTR, FLOAT]

                except ValueError as e:

                    pass

        # # check if cdata is a number - if so this could be scannable parameter

        try:  # checking if attribute can be converted to floating point value - if so it can be added to scannable parameters

            float(_elem.cdata)

            params[_elem.name] = [_elem.cdata, XML_CDATA, FLOAT]

        except ValueError as e:

            pass

        table = self.paramTW

        # for paramName, paramProps in self.scannableParams.iteritems():

        for paramName, paramProps in list(params.items()):
            currentRow = table.rowCount()

            # if table.rowCount()>0 else 0

            print('currentRow=', currentRow, 'paramName=', paramName)

            table.insertRow(currentRow)

            paramNameItem = QTableWidgetItem(paramName)

            paramValueItem = QTableWidgetItem(paramProps[0])

            paramTypeItem = QTableWidgetItem(TYPE_DICT[paramProps[1]])

            btn = TablePushButton(table)

            actionItem = None

            btn.setText('Select')

            table.setItem(currentRow, PARAMETER, paramNameItem)

            table.setItem(currentRow, VALUE, paramValueItem)

            table.setItem(currentRow, TYPE, paramTypeItem)

            table.setCellWidget(currentRow, ACTION, btn)

            btn.setPosition(currentRow, ACTION)

            btn.clicked.connect(self.handle_xml_access_path)

    #

    def handle_xml_access_path(self):

        senderBtn = self.sender()

        row, col = senderBtn.getPosition()

        print('clicked row,column=', (row, col))

        nameItem = self.paramTW.item(row, PARAMETER)

        valueItem = self.paramTW.item(row, VALUE)

        # value=float(valueItem.text())

        value = str(valueItem.text())

        print('TYPE=', TYPE)

        typeItem = self.paramTW.item(row, TYPE)

        type = TYPE_DICT_REVERSE[str(typeItem.text())]

        print('type=', type, '\n\n\n\n')

        self.close()

        precise_xml_access_path = None

        if self.handle_xml_access_callback:

            if type == XML_ATTR:

                self.precise_xml_access_path_tuple = XmlAccessPathTuple(XML_ATTR, str(nameItem.text()),
                                                                        self.handle_xml_access_callback(self.accessPath,
                                                                                                        str(
                                                                                                            nameItem.text())))

                # self.precise_xml_access_path_tuple = (XML_ATTR, precise_xml_access_path)

            else:

                self.precise_xml_access_path_tuple = XmlAccessPathTuple(XML_CDATA, 'CDATA',
                                                                        self.handle_xml_access_callback(
                                                                            self.accessPath))

    def get_precise_xml_access_path_tuple(self):

        # todo - move to separate dialog

        """

        Returns tuple with xml component type and precise access path to the component of the XML element

        :return:{tuple (XMLCOMPONENT TYPE, list} precise access path to the component of the XML element

        """

        return self.precise_xml_access_path_tuple

    def displayPythonScannableParameters(self, _pythonLine, _parameterScanFile):

        print('_pythonLine=', _pythonLine)

        from ParameterScanUtils import ParameterScanUtils

        psu = ParameterScanUtils()

        foundGlobalVar = psu.checkPythonLineForGlobalVariable(_pythonLine)

        print('foundGlobalVar=', foundGlobalVar)

    def updateUi(self):

        table = self.paramTW

        table.verticalHeader().setDefaultSectionSize(20)

        table.horizontalHeader().setDefaultSectionSize(20)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

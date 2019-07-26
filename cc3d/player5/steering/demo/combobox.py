# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:50:56 2013

@author: cmarshall
"""

import sip
sip.setapi('QString', 1)
sip.setapi('QVariant', 1)
 
import pandas as pd
from PyQt5 import QtCore, QtGui,QtWidgets

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class EditorDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if not index.isValid():
            return


        column_name = self.get_col_name_from_index(index)
        print('column_name=',column_name)
        if column_name == 'Value':
            editor = QLineEdit(parent)
            # editor.setText()
            return editor
        else:
            return None
        editor = QSpinBox(parent)
        editor.setFrame(False)
        editor.setMinimum(0)
        editor.setMaximum(100)

        return editor

    # def setEditorData(self, spinBox, index):
    #     value = index.model().data(index, Qt.EditRole)
    #     # QVariant().Int
    #     spinBox.setValue(value.Int)

    def get_col_name_from_index(self,index):
        """
        returns column name from index
        :param index:
        :return:{str or NNne}
        """
        if not index.isValid():
            return

        model = index.model()
        datatable = model.datatable

        column_name = datatable.columns[index.column()]

        return column_name

    def setEditorData(self, editor, index):

        column_name = self.get_col_name_from_index(index)
        if not column_name:
            return
        if column_name == 'Value':
            value = index.model().data(index, Qt.DisplayRole)
            print('i,j=',index.column(), index.row())
            print('val=',value)
            # editor.setText(str(value.toInt()))
            editor.setText(str(value))
        else:
            return
            # editor.setValue(value.Int)

        # #  value = index.model().data(index, Qt.EditRole)
        # value = index.model().data(index, Qt.DisplayRole)
        # val_new = index.model().datatable.iloc[0,0]
        # print 'val_new=',val_new
        # print 'value=',value
        # if index.isValid() and index.column() == 0:
        #     print 'i,j=',index.column(), index.row()
        #     print'val=',value
        #     # editor.setText(str(value.toInt()))
        #     editor.setText(str(value))
        # else:
        #     editor.setValue(value.Int)
        # # value = index.model().data(index, Qt.EditRole)
        # # QVariant().Int

        #
        # #  value = index.model().data(index, Qt.EditRole)
        # value = index.model().data(index, Qt.DisplayRole)
        # val_new = index.model().datatable.iloc[0,0]
        # print 'val_new=',val_new
        # print 'value=',value
        # if index.isValid() and index.column() == 0:
        #     print 'i,j=',index.column(), index.row()
        #     print'val=',value
        #     # editor.setText(str(value.toInt()))
        #     editor.setText(str(value))
        # else:
        #     editor.setValue(value.Int)
        # # value = index.model().data(index, Qt.EditRole)
        # # QVariant().Int



    # def setModelData(self, spinBox, model, index):
    #     spinBox.interpretText()
    #     value = spinBox.value()
    #
    #     model.setData(index, value, Qt.EditRole)

    def setModelData(self, editor, model, index):

        column_name = self.get_col_name_from_index(index)
        if not column_name:
            return

        if column_name == 'Value':
            type_conv_fcn = index.model().data_type_conv_fcn(index)
            print('type_conv_fcn=',type_conv_fcn)
            value = type_conv_fcn(editor.text())
        else:
            return
            # editor.interpretText()
            # value = editor.value()

        model.setData(index, value, Qt.EditRole)


        #
        # if index.column() == 0:
        #     type_conv_fcn = index.model().data_type_conv_fcn(index)
        #     value = type_conv_fcn(editor.text())
        # else:
        #
        #     editor.interpretText()
        #     value = editor.value()
        #
        # model.setData(index, value, Qt.EditRole)


    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)



class SimDelegate(QItemDelegate):

    def __init__(self, parent=None):
        QItemDelegate.__init__(self, parent)

    def createEditor(self, parent, option, index):
        # Create editor object of QLineEdit
        if index.column() == 1:
            editor = QLineEdit(parent)
            editor.returnPressed.connect(self.commitAndCloseEditor)
            # self.connect(editor, SIGNAL("returnPressed()"), self.commitAndCloseEditor)
            return editor
        else:
            return QItemDelegate.createEditor(self, parent, option, index)

    def commitAndCloseEditor(self):
        editor = self.sender()
        if isinstance(editor, (QLineEdit)):
            # call to commitData is essential in Qt5
            self.commitData.emit(editor)
            self.closeEditor.emit(editor)

    def setEditorData(self, editor, index):
        text = index.model().data(index, Qt.DisplayRole).value()
        if index.column() == 1:
            editor.setText(text)
        else:
            QItemDelegate.setEditorData(self, editor, index)

    def setModelData(self, editor, model, index):
        # Method uses model.setData()!
        # Make sure that you implemented setData() method
        if index.column() == 1:
            model.setData(index, QVariant(editor.text()))
        else:
            QItemDelegate.setModelData(self, editor, model, index)


class ItemData(object):
    def __init__(self,name=None,val=None):
        self._name = name
        self._val = None
        if val is not None:
            self.val = val


        self._type = None
        self._min = None
        self._max = None
        self._enum = None

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self,val):
        self._val = val
        self.type = type(self.val)

    # @property
    # def type(self):
    #     return self._type
    #
    # @type.setter
    # def type(self,type):
    #     self._type = type


class TableModel(QtCore.QAbstractTableModel): 
    def __init__(self, parent=None, *args): 
        super(TableModel, self).__init__()
        self.datatable = None
        self.type_conv_fcn_data = None


        
    def update(self, dataIn):
        print('Updating Model')
        self.datatable = dataIn
        print('Datatable : {0}'.format(self.datatable))

    def update_type_conv_fcn(self,type_conv_fcn_data):
        self.type_conv_fcn_data = type_conv_fcn_data

        
    def headerData(self, p_int, orientation, role=None):

        if orientation == Qt.Horizontal and  role == Qt.DisplayRole:
            return self.datatable.columns[p_int]
        return QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable.index) 
        
    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable.columns.values) 

    def data_type_conv_fcn(self,index):

        return self.type_conv_fcn_data[index.row()]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        #print 'Data Call'
        # print index.column(), index.row()
        # if role in [QtCore.Qt.DisplayRole,QtCore.Qt.EditRole] :
        if role == QtCore.Qt.DisplayRole:
            i = index.row()
            j = index.column()
            # if i==0 and j ==0:
            #     print 'i={} j={} val={}'.format(i,j,self.datatable.iloc[i, j])
            #     print 'data_ret no format ',self.datatable.iloc[i, j]
            #     print 'self.datatable=',self.datatable
            #     print 'data_returned=','{0}'.format(self.datatable.iloc[i, j])
            #return QtCore.QVariant(str(self.datatable.iget_value(i, j)))
            # return '{0}'.format(self.datatable.iget_value(i, j))
            # return '{0}'.format(self.datatable.iloc[i, j])
            # return '{0}'
            return '{}'.format(self.datatable[self.datatable.columns[j]][i])

        else:

            return QtCore.QVariant()

    def setData(self, index, value, role=None):
        """
        This needs to be reimplemented if  allowing editing
        :param index:
        :param Any:
        :param role:
        :return:
        """


        if role != QtCore.Qt.EditRole:
            return False

        if not index.isValid():
            # return self.rootItem
            return False

        item = index.internalPointer()

        # node = item.node()
        # attributes = []
        # attributeMap = node.attributes()
        i,j = index.row(), index.column()
        print('i,j, val = ', i,j,value)
        self.datatable.iloc[i,j] = value
        # if index.column() == PROPERTY_VALUE:
        #     # print 'BEFORE SET DATA PROPERTIES'
        #
        #     # print 'New value=',value.toString()
        #     elementType = getItemType(index)
        #     if elementType == 'color':
        #         # print 'value=',dir(value)
        #         node.updatePropertyValue(str(value.toString()))
        #     else:
        #         # node.updatePropertyValue(str(value.toString())) # value passed from delegate is QVariant object
        #         # node.updatePropertyValue(value.String)  # value passed from delegate is QVariant object
        #         node.updatePropertyValue(value.value())  # value passed from delegate is QVariant object

        return True

        return False

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        # print 'flags=',QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        existingFlags = super(TableModel, self).flags(index)
        # print 'existingFlags=',existingFlags
        # existingFlags|=QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

        # if index.column() == PROPERTY_NAME:
        #     existingFlags |= QtCore.Qt.NoItemFlags
        #
        # if index.column() == PROPERTY_VALUE:
        existingFlags |= QtCore.Qt.ItemIsEditable
            # return
            # return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable|Qt.ItemIsEditable

        return existingFlags
        # return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    # def flags(self, index):
    #     return QtCore.Qt.ItemIsEnabled
            

class TableView(QtWidgets.QTableView):
    """
    A simple table to demonstrate the QComboBox delegate.
    """
    def __init__(self, *args, **kwargs):
        QtWidgets.QTableView.__init__(self, *args, **kwargs)

def get_data_frame():
    # df = pd.DataFrame({'Name':['a','b','c','d'],
    # 'First':[0.1,0.2,0.3,0.4], 'Last':[0,0,0,0], 'Value':[5.1,6.2,7.3,8.4], 'Valid':[True, True, True, False]})
    # return df

    df = pd.DataFrame.from_items([('Name',['a','b','c','d','e']),
                                  ('Value',[5.1,6.2,7.3,8.4, 'dp']),
                                  ('Min', [0.0, 0.,0.,0.,0.0]),
                                  ('Max', [100.0, 100., 100., 100.,100 ])
                                  ])
    return df


def get_types():
    type_list = [float]*4
    type_list.append(str)
    return type_list

if __name__=='__main__':




    import sys


    app = QApplication(sys.argv) # needs to be defined first


    window = QWidget()
    layout = QHBoxLayout()

    model = QStandardItemModel(4, 2)

    cdf = get_data_frame()
    model=TableModel()
    model.update(cdf)
    model.update_type_conv_fcn(get_types())

    tableView = QTableView()
    tableView.setModel(model)

    delegate = EditorDelegate()
    tableView.setItemDelegate(delegate)

    for row in range(4):
        for column in range(2):
            index = model.index(row, column, QModelIndex())
            model.setData(index, (row + 1) * (column + 1))


    layout.addWidget(tableView)
    window.setLayout(layout)
    tableView.horizontalHeader().setSectionResizeMode( QHeaderView.Stretch)
    # window.setWindowTitle("Spin Box Delegate")
    # tableView.setWindowTitle("Spin Box Delegate")
    window.resize(QSize(800,300))
    window.show()
    # tableView.show()
    sys.exit(app.exec_())



# if __name__=="__main__":
#     from sys import argv, exit
#
#     class Widget(QtWidgets.QWidget):
#         """
#         A simple test widget to contain and own the model and table.
#         """
#         def __init__(self, parent=None):
#             QtWidgets.QWidget.__init__(self, parent)
#
#
#
#             l=QtWidgets.QVBoxLayout(self)
#             cdf = self.get_data_frame()
#             self._tm=TableModel(self)
#             self._tm.update(cdf)
#             self._tv=TableView(self)
#             # delegate = SimDelegate(self._tv)
#             delegate = SpinBoxDelegate()
#             self._tv.setItemDelegate(delegate)
#             self._tv.setModel(self._tm)
#             l.addWidget(self._tv)
#
#         def get_data_frame(self):
#             df = pd.DataFrame({'Name':['a','b','c','d'],
#             'First':[2.3,5.4,3.1,7.7], 'Last':[23.4,11.2,65.3,88.8], 'Class':[1,1,2,1], 'Valid':[True, True, True, False]})
#             return df
#
#     a=QtWidgets.QApplication(argv)
#     w=Widget()
#     w.show()
#     w.raise_()
#     exit(a.exec_())
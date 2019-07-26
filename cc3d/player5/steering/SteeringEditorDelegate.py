from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from .fancy_slider import FancySlider
from .fancy_combo import FancyCombo


class SteeringEditorDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if not index.isValid():
            return

        # column_name = self.get_col_name_from_index(index)
        col_name = self.get_col_name(index)

        if col_name == 'Value':
            item = index.model().get_item(index)

            if item.widget_name == 'slider':
                editor = self.init_slider(parent, index)
            elif item.widget_name in ['pulldown', 'pull-down', 'combobox']:
                editor = self.init_combobox(parent, index)
            else:
                editor = QLineEdit(parent)

        else:
            return None
        self.current_editor = editor
        return editor

    def init_slider(self, parent, index):
        """
        initializes slider based on the current value of the index
        :param index: {index}
        :return:{QSlider instance}
        """

        item = index.model().get_item(index)

        slider = FancySlider(parent)
        slider.setOrientation(Qt.Horizontal)

        item_type = item.item_type

        if item_type == type(1):  # checking for integer type
            item.decimal_precision = 0

        slider.set_decimal_precision(item.decimal_precision)

        if item.min is None:
            slider.setMinimum(0)
        else:
            slider.setMinimum(item.min)
        if item.max is None:
            slider.setMaximum(10 * item.val)
        else:
            slider.setMaximum(item.max)

        slider.set_default_behavior()
        slider.setValue(item.val)
        # slider.setRange(0,100)
        # slider.setTickInterval(10)
        # slider.setTickInterval(2)
        # slider.setMaximum(1000)

        slider.sliderReleased.connect(self.slider_closing)
        return slider

    def slider_closing(self,*args,**kwds):
        """
        slot handling sliderReleased signal from FancySlider (QSlider). It closes editor and commits data to the
        the model
        :param args:
        :param kwds:
        :return: None
        """
        # print 'THIS IS SLIDER CLOSING'
        # print 'args=',args
        self.commitData.emit(self.current_editor)
        self.closeEditor.emit(self.current_editor)
        # self.emit(self.closeEditor)

    def init_combobox(self, parent, index):
        """
        initializes qcombobox based on the current value of the index and the user-provided provided enum options
        :param index: {index}
        :return:{QSlider instance}
        """

        item = index.model().get_item(index)

        c_box = FancyCombo(parent)
        enum_list = [str(x) for x in item.enum]
        item_pos = enum_list.index(str(item.val))
        c_box.addItems(enum_list)
        c_box.setCurrentIndex(item_pos)

        item_type = item.item_type
        c_box.currentIndexChanged.connect(self.combobox_closing)
        # item

        return c_box
    def combobox_closing(self,*args,**kwds):
        """
        slot handling currentIndexChanged signal from FancyCombo (QComboBox). It closes editor and commits data to the
        the model
        :param args:
        :param kwds:
        :return: None
        """

        # print 'THIS IS COMBOBOX CLOSING'
        # print 'args=',args
        self.commitData.emit(self.current_editor)
        self.closeEditor.emit(self.current_editor)

    def get_col_name(self, index):
        """
        returns column name
        :param index: {Index}
        :return: {str or None}
        """
        if not index.isValid():
            return None

        model = index.model()
        return model.header_data[index.column()]

    def get_item_type(self, index):
        """
        Returns type of element
        :param index: {index}
        :return: {type}
        """
        if not index.isValid():
            return None

        model = index.model()
        return model.item_data[index.row()].item_type

    def setEditorData(self, editor, index):

        column_name = self.get_col_name(index)
        if not column_name:
            return
        if column_name == 'Value':
            value = index.model().data(index, Qt.DisplayRole)
            print('i,j=', index.column(), index.row())
            print('val=', value)
            # editor.setText(str(value.toInt()))
            if isinstance(editor, QLineEdit):
                editor.setText(str(value))
            elif isinstance(editor, QSlider):
                pass  # this is placeholder the real value is set elsewhere in init_slider
                # editor.setTickPosition(50) n
            elif isinstance(editor, QComboBox):
                pass  # this is placeholder the real value is set elsewhere in init_combobox
                # editor.setValue(str(value))

            else:
                raise ValueError('Editor has unsupported type of {}'.format(type(editor)))
            # try:
            #     editor.setText(str(value))
            # except:
            #     editor.setTickPosition(50)
        else:
            return

    def setModelData(self, editor, model, index):

        column_name = self.get_col_name(index)
        if not column_name:
            return

        if column_name == 'Value':

            item = index.model().get_item(index)
            type_conv_fcn = self.get_item_type(index)
            if item.widget_name == 'slider':

                try:
                    value = type_conv_fcn(editor.value())
                except ValueError as exc:
                    QMessageBox.warning(None, 'Type Conversion Error', str(exc))
                    return
            elif item.widget_name in ['combobox', 'pull-down']:

                try:
                    value = type_conv_fcn(editor.value())
                except ValueError as exc:
                    QMessageBox.warning(None, 'Type Conversion Error', str(exc))
                    return


            else:
                try:
                    value = type_conv_fcn(editor.text())
                except ValueError as exc:
                    QMessageBox.warning(None, 'Type Conversion Error', str(exc))
                    return

            #
            # type_conv_fcn = self.get_item_type(index)
            # print 'type_conv_fcn=', type_conv_fcn
            # try:
            #     value = type_conv_fcn(editor.text())
            # except ValueError as exc:
            #     QMessageBox.warning(None,'Type Conversion Error',str(exc))
            #     return
        else:
            return
            # editor.interpretText()
            # value = editor.value()

        model.setData(index, value, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

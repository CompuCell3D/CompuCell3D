from PyQt5.QtGui import QColor
from PyQt5.QtCore import QByteArray, QPoint, QSize

from cc3d.core.Configuration.settingdict import SerializerUtil as SerializerUtilCore
from cc3d.core.Configuration.settingdict import SettingsSQL as SettingsSQLCore
from cc3d.core.Configuration.settingdict import DictWrapper as DictWrapperCore
from cc3d.core.Configuration.settingdict import ListWrapper as ListWrapperCore


class SerializerUtil(SerializerUtilCore):

    def __init__(self):
        super().__init__()

        type_2_serializer_dict = {
            'QColor': self.qcolor_2_sql,
            'QSize': self.qsize_2_sql,
            'QPoint': self.qpoint_2_sql,
            'QByteArray': self.qbytearray_2_sql
        }
        self.type_2_serializer_dict.update(type_2_serializer_dict)

        type_2_deserializer_dict = {
            'color': self.sql_2_qcolor,
            'size': self.sql_2_qsize,
            'point': self.sql_2_qpoint,
            'bytearray': self.sql_2_qbytearray
        }
        self.type_2_deserializer_dict.update(type_2_deserializer_dict)

    def qcolor_2_sql(self, val):
        """
        QColor to sql string representation

        :param val: {QColor}
        :return: {tuple} ('color',QColor representation string)
        """
        return 'color', val.name()

    def sql_2_qcolor(self, val):
        """
        sql string representation to QColor

        :param val: {str} sql string representation
        :return: {QColor}
        """
        return QColor(val)

    def qsize_2_sql(self, val):
        """
        QSize to sql string representation

        :param val: {QSize}
        :return: {tuple} ('size',QSize representation string)
        """

        return 'size', str(val.width()) + ',' + str(val.height())

    def sql_2_qsize(self, val):
        """
        sql string representation to QSize

        :param val: {str} sql string representation
        :return: {QSize}
        """
        sizeList = val.split(',')
        sizeListInt = list(map(int, sizeList))

        return QSize(sizeListInt[0], sizeListInt[1])

    def qpoint_2_sql(self, val):
        """
        QPoint to sql string representation

        :param val: {QPoint}
        :return: {tuple} ('point',QPoint representation string)
        """
        return 'point', str(val.x()) + ',' + str(val.y())

    def sql_2_qpoint(self, val):
        """
        sql string representation to QPoint

        :param val: {str} sql string representation
        :return: {QPoint}
        """
        sizeList = val.split(',')
        sizeListInt = list(map(int, sizeList))

        return QPoint(sizeListInt[0], sizeListInt[1])

    def qbytearray_2_sql(self, val):
        """
        QByteArray to sql string representation

        :param val: {QByteArray}
        :return: {tuple} ('bytearray',QByteArray representation string)
        """
        out_str = val.data()
        return 'bytearray', out_str

    def sql_2_qbytearray(self, val):
        """
        sql string representation to QByteArray

        :param val: {str} sql string representation
        :return: {QByteArray}
        """
        return QByteArray(val)

    def dict_2_sql(self, val):
        """
        Python dict to sql string representation. Dictionary may include any element for which explicit handlers exist

        :param val: {dict}
        :return: {tuple} ('dict',dict representation string)
        """
        dw = DictWrapper(val)
        return 'dict', dw.serialize()

    def list_2_sql(self, val):
        """
        Python list to sql string representation. List may include any element for which explicit handlers exist

        :param val: {list}
        :return: {tuple} ('list',list representation string)
        """
        lw = ListWrapper(val)
        return 'list', lw.serialize()


class SettingsSQL(SettingsSQLCore):
    _SerializerUtil = SerializerUtil


class DictWrapper(DictWrapperCore):
    _SerializerUtil = SerializerUtil


class ListWrapper(ListWrapperCore):
    _SerializerUtil = SerializerUtil

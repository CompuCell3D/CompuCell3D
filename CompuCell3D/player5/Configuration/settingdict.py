import sqlite3

try:
    import cPickle as pickle
except ImportError:
    import pickle
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class SerializerUtil(object):
    """
    SerializerUtil facilitates serialization/deserialization to/from SQLite database of various settings
    """

    def __init__(self):
        self.type_2_serializer_dict = {
            'QColor': self.qcolor_2_sql,
            'str': lambda val: ('str', val),
            'unicode': lambda val: ('unicode', val),
            'int': lambda val: ('int', str(val)),
            'long':lambda val: ('long', str(val)),
            'float': lambda val: ('float', str(val)),
            'complex': lambda val: ('complex', str(val)),
            'bool': lambda val: ('bool', int(val)),
            'QSize': self.qsize_2_sql,
            'QPoint': self.qpoint_2_sql,
            'QByteArray': self.qbytearray_2_sql,
            'dict': self.dict_2_sql,
            'list': self.list_2_sql,
            'tuple': self.tuple_2_sql

        }

        self.type_2_deserializer_dict = {
            'color': self.sql_2_color,
            'str': lambda val: str(val),
            'unicode': lambda val: unicode(val),
            'int': lambda val: int(val),
            'long': lambda val: long(val),
            'float': lambda val: float(val),
            'complex': lambda val: complex(val),
            'bool': lambda val: False if int(val) == 0 else True,
            'size': self.sql_2_size,
            'point': self.sql_2_point,
            'bytearray': self.sql_2_bytearray,
            'dict': self.sql_2_dict,
            'list': self.sql_2_list,
            'tuple': self.sql_2_tuple,

        }

    def qcolor_2_sql(self, val):
        """
        QColor to sql string representation
        :param val: {QColor}
        :return: {tuple} ('color',QColor representation string)
        """
        return 'color', val.name()

    def sql_2_color(self, val):
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

    def sql_2_size(self, val):
        """
        sql string representation to QSize
        :param val: {str} sql string representation
        :return: {QSize}
        """
        sizeList = val.split(',')
        sizeListInt = map(int, sizeList)

        return QSize(sizeListInt[0], sizeListInt[1])

    def qpoint_2_sql(self, val):
        """
        QPoint to sql string representation
        :param val: {QPoint}
        :return: {tuple} ('point',QPoint representation string)
        """
        return 'point', str(val.x()) + ',' + str(val.y())

    def sql_2_point(self, val):
        """
        sql string representation to QPoint
        :param val: {str} sql string representation
        :return: {QPoint}
        """
        sizeList = val.split(',')
        sizeListInt = map(int, sizeList)

        return QPoint(sizeListInt[0], sizeListInt[1])

    def qbytearray_2_sql(self, val):
        """
        QByteArray to sql string representation
        :param val: {QByteArray}
        :return: {tuple} ('bytearray',QByteArray representation string)
        """
        out_str = ''
        for i in range(val.count()):
            out_str += str(ord(val[i]))
            if i < val.count() - 1:
                out_str += ','
        return 'bytearray', out_str

    def sql_2_bytearray(self, val):
        """
        sql string representation to QByteArray
        :param val: {str} sql string representation
        :return: {QByteArray}
        """
        try:
            elemsList = map(chr, map(int, val.split(',')))
        except:
            print 'CONFIGURATIN: COULD NOT CONVERT SETTING TO QBYTEARRAY'
            elemsList = []

        ba = QByteArray()
        for i in xrange(len(elemsList)):
            ba.append(elemsList[i])

        return ba

    def generic_2_sql(self, val):
        """
        Generic type (i.e. the one for whic there are not explicit handlers) to sql string representation
        :param val: {any type not handled by explicit handlers}
        :return: {tuple} ('pickle',pickle representation string of the generic object)
        """

        return 'pickle', pickle.dumps(val)

    def sql_2_generic(self, val):
        """
        sql string representation to generic type
        :param val: {str or unicode} sql string representation
        :return: {generic type i.e. one defined int he pickle representation string}
        """
        return pickle.loads(str(val))

    # def getstate_dict(self):
    #     print 'self.keys = ', self.keys()

    def dict_2_sql(self, val):
        """
        Python dict to sql string representation. Dictionary may include any element for which
        explicit handlers exist
        :param val: {dict}
        :return: {tuple} ('dict',dict representation string)
        """
        dw = DictWrapper(val)
        return 'dict', dw.serialize()

    def sql_2_dict(self, val):
        """
        sql string representation to python dict
        :param val: {str} sql string representation
        :return: {dict}
        """
        p_load = pickle.loads(str(val))
        out_dict = {}
        for k, v in p_load.items():  # v is a tuple (type, value_repr)
            value_type = v[0]
            val_repr = v[1]
            deserializer_fcn = self.guess_deserializer_fcn(value_type)
            value = deserializer_fcn(val_repr)

            out_dict[k] = value

        return out_dict

    def list_2_sql(self, val):
        """
        Python list to sql string representation. List may include any element for which
        explicit handlers exist
        :param val: {list}
        :return: {tuple} ('list',list representation string)
        """
        lw = ListWrapper(val)
        return 'list', lw.serialize()


    def sql_2_list(self, val):
        """
        sql string representation to python list
        :param val: {str} sql string representation
        :return: {list}
        """
        l_load = pickle.loads(str(val))

        out_list = []

        for list_tuple in l_load:  # l_load is a list of tuples (type, value_repr)
            value_type = list_tuple[0]
            val_repr = list_tuple[1]

            deserializer_fcn = self.guess_deserializer_fcn(value_type)
            value = deserializer_fcn(val_repr)
            out_list.append(value)

        return out_list

    def tuple_2_sql(self, val):
        """
        Python tuple to sql string representation. List may include any element for which
        explicit handlers exist
        :param val: {tuple}
        :return: {tuple} ('tuple',tuple representation string)
        """

        type_name, serialization = self.list_2_sql(val)
        return 'tuple', serialization

    def sql_2_tuple(self, val):
        """
        sql string representation to python tuple
        :param val: {str} sql string representation
        :return: {tuple}
        """
        return tuple(self.sql_2_list(val))

    def guess_serializer_fcn(self, val):
        """
        Given the object 'val' this function guesses the function that one has to use to convert it
        to a representation that can be pickled
        :param val: {arbitrary object}
        :return: {fcn} function that facilitates conversion from object to picklable representation
        """
        try:
            return self.type_2_serializer_dict[val.__class__.__name__]
        except KeyError:
            # prevent pickle conversion - try to enforce explicit type converters
            warning_msg = 'guess_serializer_fcn: could not find converter for {}'.format(val.__class__.__name__)
            print (warning_msg)
            raise RuntimeError(warning_msg)

            # return self.generic_2_sql

    def guess_deserializer_fcn(self, stored_type):
        """
        Given type name - string value this function guesses the correct deserializer function
        that one uses to convert corresponding pickled string into actual object.
        Not: guess_deserializer_fcn is used in the contexts where one knows stored_type and its
        pickled string representation
        to a representation that can be pickled
        :param stored_type: {str} name of the stored type
        :return: {fcn} function that facilitates conversion from pickled representation to actual object
        """
        try:
            return self.type_2_deserializer_dict[stored_type]
        except KeyError:
            # prevent pickle conversion- try to enforce explicit type converters
            # raise RuntimeError('guess_deserializer_fcn: could not find converter for {}'.format(stored_type))

            return self.sql_2_generic


    def val_2_sql(self, val):
        """
        Wrapper function that converts arbitrary type 'val' to
        sql pickl representation
        :param val: {object}
        :return: {tuple} (type_name, pickled sql string representation)
        """
        try:
            serializer_fcn = self.guess_serializer_fcn(val)
        except RuntimeError:
            return None, None
        val_type, val_repr = serializer_fcn(val)

        return val_type, val_repr

    def sql_2_val(self, obj):
        """
        Converts type, str serialization tuple to actual value
        :param obj: {tuple} (type, string representation - serialization string)
        :return: actual value represented by obj serialization tuple
        """
        try:
            deserializer_fcn = self.guess_deserializer_fcn(obj[0])  # obj[0] stores type
        except RuntimeError:
            return None

        val = deserializer_fcn(obj[1])  # obj[1] stores serialization string

        return val


class DictWrapper(dict):
    """
    Wrapper class that facilitates conversion of the dictionaries into sql picklable string
    """

    def __init__(self, *args, **kwds):
        super(DictWrapper, self).__init__(*args, **kwds)

    def serialize(self):
        """
        Generates pickled string representation of the dictionary
        Note: this is not the same representation that one gets by calling pickle.dumps on a python dictionary
        Here we are avoiding direct pickling of PyQt objects, instead we are generating
        representations that are independent on the particular details of the Qt python wrapper implementations
        :return: {str} pickle serialization string of the dictionary
        """
        su = SerializerUtil()

        # state = {}
        state = self.copy()
        # su_state = {}
        for key, val in self.items():
            # if key in ['su'] : continue
            state[key] = su.val_2_sql(val)

        return pickle.dumps(state)

    def deserialize(self, s):
        """
        returns unpickled dictionary based on pickled str s. Currently not in use
        :param s: {str} string representation of the pickled dict object
        :return: {dict}
        """
        return pickle.loads(str(s))

    def __getstate__(self):
        """
        Overide of pickle-used method. Currently not in use
        :return: {dict} representation of the dict
        """
        # print 'self.keys = ', self.local_dict.keys()
        su = SerializerUtil()
        # state = {}
        state = self.copy()
        # su_state = {}
        for key, val in self.items():
            # if key in ['su'] : continue
            state[key] = su.val_2_sql(val)

        # del state['su']
        # state['su'] = su_state

        return state

    def __setstate__(self, newstate):
        """
        Overide of pickle-used method. Currently not in use
        :return: None
        """
        su = {}
        for key, val in newstate.items():
            # newstate[key] = val[1]
            self[key] = val[1]
            # newstate['su'] = None
            # self.__dict__.update(newstate)


class ListWrapper(list):
    """
    Wrapper class that facilitates conversion of the dictionaries into sql picklable string
    """

    def __init__(self, *args, **kwds):
        super(ListWrapper, self).__init__(*args, **kwds)

    def serialize(self):
        """
        Generates pickled string representation of the list
        Note: this is not the same representation that one gets by calling pickle.dumps on a python list
        Here we are avoiding direct pickling of PyQt objects, instead we are generating
        representations that are independent on the particular details of the Qt python wrapper implementations
        :return: {str} pickle serialization string of the list
        """
        su = SerializerUtil()
        out_state = []

        for val in self:
            out_state.append(su.val_2_sql(val))

        return pickle.dumps(out_state)

    def deserialize(self, s):
        """
        returns unpickled list based on pickled str s. Currently not in use
        :param s: {str} string representation of the pickled list object
        :return: {dict}
        """
        return pickle.loads(str(s))


class SettingsSQL(object):
    def __init__(self, filename=".shared.db", **kwargs):
        self.filename = filename
        self.conn = sqlite3.connect(self.filename)
        self._create_table()

        if len(kwargs) > 0:
            for key, value in kwargs.items():
                self[key] = value

        self.su = SerializerUtil()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etb):
        self.close()

    def _create_table(self):
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS settings "
                "(name TEXT PRIMARY KEY , type TEXT NOT NULL, value TEXT NOT NULL)")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_name ON settings (name)")

    def names(self):
        """
        returns a list of all setting names stored in the object
        :return: {list} list of strings
        """
        with self.conn:
            cur = self.conn.execute("SELECT name FROM settings")
            return [key[0] for key in cur.fetchall()]


    def setSetting(self, key, val):

        with self.conn:
            val_type, val_repr = self.su.val_2_sql(val)

            if val_type is not None and val_type is not None:
                self.conn.execute(
                    "INSERT OR REPLACE INTO settings VALUES (?,?,?)",
                    (key, val_type, val_repr))
            else:
                print ('NOT STORING SETTING: "{}". LIKELY DUE TO MISSING TYPE CONVERTER'.format(key))

    def setting(self, key):
        with self.conn:
            cur = self.conn.execute(
                "SELECT type, value FROM settings WHERE name = (?)", (key,))
            obj = cur.fetchone()
            if obj is None:
                raise KeyError("No such key: " + key)

            return self.su.sql_2_val(obj)
            # return pickle.loads(obj[0].encode())

    def getSetting(self, key):
        # type: (object) -> object
        # type: (object) -> object
        """
        Added for backward compatibility
        :param key:
        :return:
        """
        # print 'getting key = ', key
        return self.setting(key)

    def close(self):
        self.conn.close()



# if __name__ == "__main__":  # pragma: no cover
#     from PyQt5.QtGui import *
#     from PyQt5.QtCore import *
#     import sys
#
#     s = SettingsSQL('_settings_demo.sqlite')
#
#     l = [1,2,QColor('red'),'dupa']
#     lw = ListWrapper(l)
#
#     l_serialized = lw.serialize()
#
#
#     # l_out = pickle.dumps(lw)
#
#     print l_out
#     print lw
#
#     l_load = pickle.loads(l_out)
#
#
#     print
#     sys.exit()
#     #
#     d = {'a': 2, 'b': 3, 'c': QColor('red')}
#
#     s.setSetting('dictionary', d)
#
#     dict_s = s.setting('dictionary')
#
#     dw = DictWrapper()
#     dw.update(d)
#
#     p_serialized = dw.serialize()
#
#     p_out = pickle.dumps(dw)
#     # s.dict_2_sql(d)
#     #
#
#     p_load = pickle.loads(p_out)
#
#     print p_load
#
#     # # trying out serialization
#     # val_type, val_repr = s.su.val_2_sql(dw)
#     #
#     # print val_type
#     # print val_repr
#     # #
#     # # # dw = DictWrapper(d)
#     # # # p_out = pickle.dumps(dw)
#     # # # # s.dict_2_sql(d)
#     # # # #
#     # # #
#     # # # p_load = pickle.loads(p_out)
#     # # # print
#     sys.exit()
#
#     s = SettingsSQL('_settings.sqlite')
#
#
#     col = QColor('red')
#     size = QSize(20, 30)
#
#     ba = QByteArray();
#     ba.resize(5)
#
#     s.setSetting('bytearray', ba)
#     s.setSetting('WindowSize', size)
#     s.setSetting('ScreenshotFrequency', 8)
#     s.setSetting('MinConcentration', 8.2)
#     s.setSetting('ComplexNum', 8.2 + 3j)
#     s.setSetting('dupa', 'blada2')
#     # s.setSetting('window_data', {'size': 20, 'color': '#ffff00'})
#     s.setSetting('window_color', col)
#
#     print s.setting('bytearray')
#     print s.setting('WindowSize')
#     print s.setting('ScreenshotFrequency')
#     print s.setting('MinConcentration')
#     print s.setting('ComplexNum')
#     print s.setting('dupa')
#     print s.setting('window_color')
#
#
#     # d = SettingDict("_settings.sqlite")
#     # d['RecentFile'] = '/dupa'
#
#     # import numpy as np
#     #
#     # d = SettingDict("test.sqlite")
#     # d["thing"] = "whatever"
#     # print(d["thing"])
#     #
#     # d["wat"] = np.random.random((100,200))
#     # print(d["wat"])
#     #
#     # with SettingDict("test.sqlite") as pd:
#     #     print(pd["wat"])
#     #
#     # print(d.keys())
#     #
#     # print(SettingDict().keys())
#     #
#     # pd2 = SettingDict(a=1, b=2, c=3)
#     # for key in pd2.keys():
#     #     print(pd2[key])
#     #
#     # print(len(pd2))

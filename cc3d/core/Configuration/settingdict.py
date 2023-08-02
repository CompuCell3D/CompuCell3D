import sqlite3
import locale

try:
    import pickle as pickle
except ImportError:
    import pickle
from cc3d.core.GraphicsOffScreen.primitives import *

_enc = locale.getdefaultlocale()[1]


class SerializerUtil(object):
    """
    SerializerUtil facilitates serialization/deserialization to/from SQLite database of various settings
    """

    def __init__(self):
        self.type_2_serializer_dict = {
            'Color': lambda val: ('color', val.to_str_rgb()),
            'str': lambda val: ('str', val),
            'unicode': lambda val: ('unicode', val),
            'int': lambda val: ('int', str(val)),
            'long': lambda val: ('long', str(val)),
            'float': lambda val: ('float', str(val)),
            'complex': lambda val: ('complex', str(val)),
            'bool': lambda val: ('bool', int(val)),
            'Size2D': lambda val: ('size', str(val)),
            'Point2D': lambda val: ('point', str(val)),
            'bytearray': lambda val: ('bytearray', bytes(val)),
            'dict': self.dict_2_sql,
            'list': self.list_2_sql,
            'tuple': self.tuple_2_sql

        }

        self.type_2_deserializer_dict = {
            'color': self.sql_2_color,
            'str': lambda val: str(val),
            'unicode': lambda val: str(val),
            'int': lambda val: int(val),
            'long': lambda val: int(val),
            'float': lambda val: float(val),
            'complex': lambda val: complex(val),
            'bool': lambda val: False if int(val) == 0 else True,
            'size': self.sql_2_size,
            'point': self.sql_2_point,
            'bytearray': lambda val: bytes(val),
            'dict': self.sql_2_dict,
            'list': self.sql_2_list,
            'tuple': self.sql_2_tuple,

        }

    def sql_2_color(self, val: str):
        """
        sql string representation to Color

        :param val: {str} sql string representation
        :return: {Color}
        """
        return Color.from_str_rgb(val)

    def sql_2_size(self, val: str):
        """
        sql string representation to Size2D

        :param val: {str} sql string representation
        :return: {Size2D}
        """
        sizeList = val.split(',')
        sizeListInt = list(map(int, sizeList))

        return Size2D(width=sizeListInt[0], height=sizeListInt[1])

    def sql_2_point(self, val: str):
        """
        sql string representation to Point2D

        :param val: {str} sql string representation
        :return: {Point2D}
        """
        sizeList = val.split(',')
        sizeListInt = list(map(int, sizeList))

        return Point2D(x=sizeListInt[0], y=sizeListInt[1])

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

    def dict_2_sql(self, val):
        """
        Python dict to sql string representation. Dictionary may include any element for which explicit handlers exist

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
        p_load = pickle.loads(val)
        out_dict = {}
        for k, v in list(p_load.items()):  # v is a tuple (type, value_repr)
            value_type = v[0]
            val_repr = v[1]
            deserializer_fcn = self.guess_deserializer_fcn(value_type)
            value = deserializer_fcn(val_repr) if val_repr is not None else None

            out_dict[k] = value

        return out_dict

    def list_2_sql(self, val):
        """
        Python list to sql string representation. List may include any element for which explicit handlers exist

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
        l_load = pickle.loads(val)

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
        Python tuple to sql string representation. List may include any element for which explicit handlers exist

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
            print(warning_msg)
            raise RuntimeError(warning_msg)

            # return self.generic_2_sql

    def guess_deserializer_fcn(self, stored_type):
        """
        Given type name - string value this function guesses the correct deserializer function
        that one uses to convert corresponding pickled string into actual object.

        Note: guess_deserializer_fcn is used in the contexts where one knows stored_type and its
        pickled string representation to a representation that can be pickled

        :param stored_type: {str} name of the stored type
        :return: {fcn} function that facilitates conversion from pickled representation to actual object
        """
        try:
            return self.type_2_deserializer_dict[stored_type]
        except KeyError:
            # prevent pickle conversion- try to enforce explicit type converters
            return self.sql_2_generic

    def val_2_sql(self, val):
        """
        Wrapper function that converts arbitrary type 'val' to sql pickl representation

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

    _SerializerUtil = SerializerUtil

    def serialize(self):
        """
        Generates pickled string representation of the dictionary

        Note: this is not the same representation that one gets by calling pickle.dumps on a python dictionary

        Here we are avoiding direct pickling of objects, instead we are generating
        representations that are independent of the particular details of the implementations

        :return: {str} pickle serialization string of the dictionary
        """
        su = self._SerializerUtil()
        state = self.copy()
        for key, val in list(self.items()):
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

        su = self._SerializerUtil()

        state = self.copy()

        for key, val in list(self.items()):
            state[key] = su.val_2_sql(val)

        return state

    def __setstate__(self, newstate):
        """
        Overide of pickle-used method. Currently not in use

        :return: None
        """
        for key, val in list(newstate.items()):
            self[key] = val[1]


class ListWrapper(list):
    """
    Wrapper class that facilitates conversion of the dictionaries into sql picklable string
    """

    _SerializerUtil = SerializerUtil

    def serialize(self):
        """
        Generates pickled string representation of the list
        Note: this is not the same representation that one gets by calling pickle.dumps on a python list
        Here we are avoiding direct pickling of objects, instead we are generating
        representations that are independent on the particular details of the implementations

        :return: {str} pickle serialization string of the list
        """
        su = self._SerializerUtil()
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

    _SerializerUtil = SerializerUtil

    def __init__(self, filename=".shared.db", **kwargs):
        self.filename = filename
        self.conn = sqlite3.connect(self.filename)
        self._create_table()

        if len(kwargs) > 0:
            for key, value in list(kwargs.items()):
                self[key] = value

        self.su = self._SerializerUtil()

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
                print(('NOT STORING SETTING: "{}". LIKELY DUE TO MISSING TYPE CONVERTER'.format(key)))

    def setting(self, key):
        with self.conn:
            cur = self.conn.execute(
                "SELECT type, value FROM settings WHERE name = (?)", (key,))
            obj = cur.fetchone()
            if obj is None:
                raise KeyError("No such key: " + key)

            return self.su.sql_2_val(obj)

    def getSetting(self, key):
        """
        Added for backward compatibility

        :param key:
        :return:
        """
        # print 'getting key = ', key
        return self.setting(key)

    def close(self):
        self.conn.close()

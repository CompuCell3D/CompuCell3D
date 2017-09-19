import sqlite3

try:
    import cPickle as pickle
except ImportError:
    import pickle

from PyQt5.QtGui import *

class SerializerUtil(object):
    def __init__(self):
        self.type_2_serializer_dict = {
            'QColor': self.qcolor_2_sql,
            'str': lambda val: ('str', val),
            'int': lambda val: ('int', str(val)),
            'float': lambda val: ('float', str(val)),
            'complex': lambda val: ('complex', str(val)),
            'QSize': self.qsize_2_sql,
            'QPoint': self.qpoint_2_sql,
            'QByteArray': self.qbytearray_2_sql,

        }

    def qcolor_2_sql(self, val):

        return 'color', val.name()

    def qsize_2_sql(self, val):

        return 'size', str(val.width()) + ',' + str(val.height())

    def qpoint_2_sql(self, val):

        return 'point', str(val.x()) + ',' + str(val.y())

    def qbytearray_2_sql(self, val):

        out_str = ''
        for i in range(val.count()):
            out_str += str(ord(val[i]))
            if i < val.count() - 1:
                out_str += ','
        return 'bytearray', out_str

    def generic_2_sql(self, val):

        return 'pickle', pickle.dumps(val)

    def getstate_dict(self):
        print 'self.keys = ', self.keys()

    def dict_2_sql(self, val):

        dw = DictWrapper(val)
        print pickle.dumps(dw)

    def guess_serializer_fcn(self, val):

        try:
            return self.type_2_serializer_dict[val.__class__.__name__]
        except KeyError:
            return self.generic_2_sql

    def val_2_sql(self, val):
        serializer_fcn = self.guess_serializer_fcn(val)
        val_type, val_repr = serializer_fcn(val)

        return val_type, val_repr


class DictWrapper1(dict):
    def __init__(self,*args,**kwds):
        super(DictWrapper1,self).__init__(*args,**kwds)
        self.su = SerializerUtil()

    def __getstate__(self):
        # print 'self.keys = ', self.local_dict.keys()

        # state = {}
        state = self.__dict__.copy()
        su_state = {}
        for key,val in self.items():
            if key in ['su']: continue
            su_state[key] = self.su.val_2_sql(val)
        del state['su']
        # state['su'] = su_state

        return state

    def __setstate__(self, newstate):
        # print 'self.keys = ', self.local_dict.keys()
        su = {}
        for key, val in  newstate['su'].items():

            newstate[key] = val[1]
        # newstate['su'] = None
        self.__dict__.update(newstate)



class DictWrapper(object):
    def __init__(self, _dict):
        self.local_dict = _dict
        self.su = SerializerUtil()

    def __getstate__(self):
        print 'self.keys = ', self.local_dict.keys()

        # state = {}
        state = self.__dict__.copy()
        su_state = {}
        for key,val in self.local_dict.items():
            su_state[key] = self.su.val_2_sql(val)
        state['su'] = su_state

        return state

    def __setstate__(self, newstate):
        # print 'self.keys = ', self.local_dict.keys()
        su = {}
        for key, val in  newstate['su'].items():

            newstate['local_dict'][key] = val[1]
        newstate['su'] = None
        self.__dict__.update(newstate)


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

    def setSetting(self, key, val):

        with self.conn:

            val_type, val_repr = self.su.val_2_sql(val)

            self.conn.execute(
                "INSERT OR REPLACE INTO settings VALUES (?,?,?)",
                (key, val_type, val_repr))


    def close(self):
        self.conn.close()


class SettingDict(object):
    def __init__(self, filename=".shared.db", **kwargs):
        self.filename = filename
        self.conn = sqlite3.connect(self.filename)
        self._create_table()

        if len(kwargs) > 0:
            for key, value in kwargs.items():
                self[key] = value

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etb):
        self.close()

    def _create_table(self):
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS dict "
                "(name TEXT PRIMARY KEY, object BLOB)")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_name ON dict (name)")

    def __len__(self):
        with self.conn:
            cur = self.conn.execute("SELECT COUNT(*) FROM dict")
            return cur.fetchone()[0]

    def __getitem__(self, key):
        with self.conn:
            cur = self.conn.execute(
                "SELECT object FROM dict WHERE name = (?)", (key,))
            obj = cur.fetchone()
            if obj is None:
                raise KeyError("No such key: " + key)
            return pickle.loads(obj[0].encode())

    def __setitem__(self, key, value):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO dict VALUES (?,?)",
                (key, pickle.dumps(value)))

    def __delitem__(self, key):
        if key not in self:
            raise KeyError
        with self.conn:
            self.conn.execute("DELETE FROM dict WHERE name = (?)", (key,))

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __iter__(self):
        for key in self.keys():
            yield key

    def keys(self):
        with self.conn:
            cur = self.conn.execute("SELECT name FROM dict")
            return [key[0] for key in cur.fetchall()]

    def items(self):
        for key in self:
            yield (key, self[key])

    def values(self):
        """Return an iterator of the :class:`Permadict`'s values."""
        for key in self:
            yield self[key]

    def clear(self):
        """Remove all items from the Peramdict."""
        with self.conn:
            self.conn.execute("DELETE FROM dict")

    def get(self, key, default=None):
        """Return the value for ``key`` if it exists, otherwise return the
        ``default``.

        """
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key):
        """If ``key`` is present, remove it and return its value, else raise a
        :class:`KeyError`.

        """
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            raise

    def update(self, iterable):
        """Update the :class:`Permadict` with the key/value pairs of
        ``iterable``.

        Returns ``None``.

        """
        if isinstance(iterable, dict):
            iter_ = iterable.items()
        else:
            iter_ = iterable
        for key, value in iter_:
            self[key] = value
        return None

    def close(self):
        self.conn.close()


if __name__ == "__main__":  # pragma: no cover
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    import sys

    s = SettingsSQL('_settings_demo.sqlite')
    #
    d = {'a': 2, 'b': 3}

    dw = DictWrapper1()
    dw.update(d)
    p_out = pickle.dumps(dw)
    # s.dict_2_sql(d)
    #

    p_load = pickle.loads(p_out)
    print


    # dw = DictWrapper(d)
    # p_out = pickle.dumps(dw)
    # # s.dict_2_sql(d)
    # #
    #
    # p_load = pickle.loads(p_out)
    # print
    sys.exit()


    s = SettingsSQL('_settings.sqlite')
    col = QColor('red')
    size = QSize(20, 30)

    ba = QByteArray();
    ba.resize(5)

    s.setSetting('bytearray', ba)
    s.setSetting('WindowSize', size)
    s.setSetting('ScreenshotFrequency', 8)
    s.setSetting('MinConcentration', 8.2)
    s.setSetting('ComplexNum', 8.2 + 3j)
    s.setSetting('dupa', 'blada2')
    s.setSetting('window_data', {'size': 20, 'color': '#ffff00'})
    s.setSetting('window_color', col)


    # d = SettingDict("_settings.sqlite")
    # d['RecentFile'] = '/dupa'

    # import numpy as np
    #
    # d = SettingDict("test.sqlite")
    # d["thing"] = "whatever"
    # print(d["thing"])
    #
    # d["wat"] = np.random.random((100,200))
    # print(d["wat"])
    #
    # with SettingDict("test.sqlite") as pd:
    #     print(pd["wat"])
    #
    # print(d.keys())
    #
    # print(SettingDict().keys())
    #
    # pd2 = SettingDict(a=1, b=2, c=3)
    # for key in pd2.keys():
    #     print(pd2[key])
    #
    # print(len(pd2))

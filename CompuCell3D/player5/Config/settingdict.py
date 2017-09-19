import sqlite3
try:
    import cPickle as pickle
except ImportError:
    import pickle

from PyQt5.QtGui import *

class SettingsSQL(object):
    def __init__(self, filename=".shared.db", **kwargs):
        self.filename = filename
        self.conn = sqlite3.connect(self.filename)
        self._create_table()

        if len(kwargs) > 0:
            for key, value in kwargs.items():
                self[key] = value

        self.key_type = {
            'RecentFile':'str',
            'ScreenUpdatefrequency':'str'
        }

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


    def color_2_sql(self,val):

        return 'color', val.name()


    def generic_2_sql(self,val):

        return 'pickle', pickle.dumps(val)

    def guess_type_conversion_fcn(self, val):

        if isinstance(val,QColor):
            return self.color_2_sql
        else:
            return self.generic_2_sql

    def setSetting(self, _key, _value):
        with self.conn:
            try:
                _type = self.key_type[_key]
            except KeyError:
                _type = 'pickle'

            if _type == 'pickle':
                self.conn.execute(
                    "INSERT OR REPLACE INTO settings VALUES (?,?,?)",
                    (_key, _type, pickle.dumps(_value)))
            else:
                self.conn.execute(
                    "INSERT OR REPLACE INTO settings VALUES (?,?,?)",
                    (_key, _type, pickle.dumps(_value)))


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
    s = SettingsSQL('_settings.sqlite')

    col = QColor('red')

    s.setSetting('dupa','blada2')
    s.setSetting('window_data', {'size':20,'color':'#ffff00'})
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

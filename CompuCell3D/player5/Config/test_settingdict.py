import unittest
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from settingdict import SettingsSQL


class TestSettingdict(unittest.TestCase):
    def setUp(self):
        pass

    def test_simple_types(self):
        s = SettingsSQL('_TestSettingdict.sqlite')

        size = QSize(20, 30)
        col = QColor('red')
        ba = QByteArray()
        ba.resize(5)

        s.setSetting('bytearray', ba)
        s.setSetting('WindowSize', size)
        s.setSetting('ScreenshotFrequency', 8)
        s.setSetting('MinConcentration', 8.2)
        s.setSetting('ComplexNum', 8.2 + 3j)
        s.setSetting('dupa', 'blada2')
        s.setSetting('window_data', {'size': 20, 'color_str': '#ffff00', 'color':QColor('red')})
        s.setSetting('window_color', col)

        # testing serialization/deserialization of dictionary
        dict_s = s.setting('window_data')
        self.assertIsInstance(dict_s,dict)

        self.assertIsInstance(dict_s['size'], int)
        self.assertEqual(dict_s['size'], 20)

        self.assertIsInstance(dict_s['color_str'], str)
        self.assertEqual(dict_s['color_str'], '#ffff00')

        self.assertIsInstance(dict_s['color'], QColor)
        self.assertEqual(dict_s['color'].name(), '#ff0000')


        print dict_s

        size_s = s.setting('WindowSize')
        self.assertIsInstance(size_s, QSize)
        self.assertEqual(size_s.width(), size.width())

        int_s = s.setting('ScreenshotFrequency')
        self.assertIsInstance(int_s, int)
        self.assertEqual(int_s, 8)

        float_s = s.setting('MinConcentration')
        self.assertIsInstance(float_s, float)
        self.assertEqual(float_s, 8.2)

        complex_s = s.setting('ComplexNum')

        self.assertIsInstance(complex_s, complex)
        self.assertEqual(complex_s.real, 8.2)
        self.assertEqual(complex_s.imag, 3)

        str_s = s.setting('dupa')
        self.assertIsInstance(str_s, str)
        self.assertEqual(str_s, 'blada2')

        color_s = s.setting('window_color')
        self.assertIsInstance(color_s, QColor)
        self.assertEqual(color_s.name(), '#ff0000')

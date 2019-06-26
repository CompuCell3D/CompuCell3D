import sys
import unittest
from os.path import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from cc3d.player5.Configuration.settingdict import SettingsSQL


class TestSettingdict(unittest.TestCase):

    def setUp(self):
        pass

    def test_simple_types(self):
        s = SettingsSQL('_TestSettingdict.sqlite')

        size = QSize(20, 30)
        col = QColor('red')
        bytes_to_write = b"\x00\x00\x00\xff\x00\x00\x00\x00\xfd\x00\x00\x00\x02"
        ba = QByteArray(bytes_to_write)
        # ba = QByteArray()
        # ba.resize(5)
        s.setSetting('bytearray', ba)
        ba_check = s.setting('bytearray')

        bytes_to_write = b"\x00"
        ba = QByteArray(bytes_to_write)
        # ba = QByteArray()
        # ba.resize(5)
        s.setSetting('bytearray', ba)
        ba_check = s.setting('bytearray')

        self.assertIsInstance(ba_check, QByteArray)
        self.assertEqual(ba_check,bytes_to_write)

        s.setSetting('flag_false', False)
        s.setSetting('flag_true', True)


        s.setSetting('WindowSize', size)
        s.setSetting('ScreenshotFrequency', 8)
        s.setSetting('MinConcentration', 8.2)
        s.setSetting('ComplexNum', 8.2 + 3j)
        s.setSetting('d1', 'blade')

        s.setSetting('window_color', col)

        s.setSetting('qdate', QDate(1999,1,1))



        flag_true_s = s.setting('flag_true')
        self.assertIsInstance(flag_true_s, bool)
        self.assertEqual(flag_true_s, True)

        flag_false_s = s.setting('flag_false')
        self.assertIsInstance(flag_false_s, bool)
        self.assertEqual(flag_false_s, False)

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

        str_s = s.setting('d1')
        self.assertIsInstance(str_s, str)
        self.assertEqual(str_s, 'blade')

        color_s = s.setting('window_color')
        self.assertIsInstance(color_s, QColor)
        self.assertEqual(color_s.name(), '#ff0000')

        with self.assertRaises(KeyError):
            qdate = s.setting('qdate')


    def test_list_types(self):
        s = SettingsSQL('_TestSettingdict.sqlite')

        l = []
        s.setSetting('RecentSimulations', l)


        l = [1, 2.0, QColor('red'), 'dp']

        s.setSetting('window_data_list', l)


        l_s = s.setting('window_data_list')

        self.assertIsInstance(l_s, list)

        self.assertIsInstance(l_s[0], int)
        self.assertEqual(l_s[0], 1)

        self.assertIsInstance(l_s[1], float)
        self.assertEqual(l_s[1], 2.0)

        self.assertIsInstance(l_s[2], QColor)
        self.assertEqual(l_s[2].name(), "#ff0000")
        self.assertIsInstance(l_s[2], QColor)
        self.assertEqual(l_s[2].name(), "#ff0000")

        self.assertIsInstance(l_s[3], str)
        self.assertEqual(l_s[3], 'dp')

        print(l_s)

    def test_dict_types(self):
        s = SettingsSQL('_TestSettingdict.sqlite')

        d = {'size': 20,
             'color_str': '#ffff00',
             'color': QColor('red'),
             'flag_true': True,
             'flag_false': False
             }

        s.setSetting('window_data', d)
        # testing serialization/deserialization of dictionary
        dict_s = s.setting('window_data')
        self.assertIsInstance(dict_s, dict)

        self.assertIsInstance(dict_s['size'], int)
        self.assertEqual(dict_s['size'], 20)

        self.assertIsInstance(dict_s['color_str'], str)
        self.assertEqual(dict_s['color_str'], '#ffff00')

        self.assertIsInstance(dict_s['color'], QColor)
        self.assertEqual(dict_s['color'].name(), '#ff0000')

        self.assertIsInstance(dict_s['flag_true'], bool)
        self.assertEqual(dict_s['flag_true'], True)

        self.assertIsInstance(dict_s['flag_false'], bool)
        self.assertEqual(dict_s['flag_false'], False)

        type_color_map = {
            0: QColor('black'),
            1: QColor('blue'),
            2: QColor('green'),
            3: QColor('red'),
            4: QColor('yellow'),
            5: QColor('brown'),
            6: QColor('lightblue'),
            7: QColor('lightgreen'),
            8: QColor('purple'),
            9: QColor('orange'),
            10: QColor('turquoise'),

        }
        print(dict_s)

        s.setSetting('TypeColorMap',type_color_map)

        type_color_map_s = s.setting('TypeColorMap')

        print(type_color_map_s)

        field_params = {}

        s.setSetting('FieldParams', field_params)



    def test_custom_settings(self):
        setting_path = join(dirname(dirname(__file__)), 'CustomSetting.sqlite')
        s = SettingsSQL(setting_path)

        d = s.setting('WindowsLayout')
        print()

    def test_convert_xml_to_sql_setting(self):
        """
        This test case checks if we can copy existing xml-based settings into sql-based format
        It can be reused as a one-time utility to carry out this conversion
        :return:
        """
        if sys.platform.startswith('darwin'):
            xml_setting_path = join(dirname(dirname(__file__)), 'Configuration_settings', '_settings_osx.xml')
            sql_setting_path = join(dirname(dirname(__file__)), 'Configuration_settings', '_settings.sqlite')
        else:
            xml_setting_path = join(dirname(dirname(__file__)), 'Configuration_settings', '_settings.xml')
            sql_setting_path = join(dirname(dirname(__file__)), 'Configuration_settings', '_settings.sqlite')

        from CompuCell3D.player5.Configuration.SettingUtils import CustomSettings
        settings = CustomSettings()
        settings.readFromXML(xml_setting_path)

        sd = settings.getNameSettingDict()

        sql_settings = SettingsSQL(sql_setting_path)

        for key in list(sd.keys()):
            setting_value = sd[key].toObject()
            # print 'key=', key
            # print 'value=', setting_value
            sql_settings.setSetting(key, setting_value)

    def test_super_composite_types(self):
        s = SettingsSQL('_TestSettingdict.sqlite')

        d = {'size': 20,
             'color_data': {
                 'color_str': '#ffff00',
                 'color': QColor('red'),
             },
             'flag_true': True,
             'flag_false': False,
             'window_data_list': [1, 2, QColor('red'), 'dp']

             }

        s.setSetting('window_data', d)

        dict_s = s.setting('window_data')

        self.assertIsInstance(dict_s['size'], int)
        self.assertEqual(dict_s['size'], 20)

        color_data_s = dict_s['color_data']
        self.assertIsInstance(color_data_s, dict)

        self.assertIsInstance(color_data_s['color_str'], str)
        self.assertEqual(color_data_s['color_str'], '#ffff00')

        self.assertIsInstance(color_data_s['color'], QColor)
        self.assertEqual(color_data_s['color'].name(), '#ff0000')

        self.assertIsInstance(dict_s['flag_true'], bool)
        self.assertEqual(dict_s['flag_true'], True)

        self.assertIsInstance(dict_s['flag_false'], bool)
        self.assertEqual(dict_s['flag_false'], False)

        l_s = s.setting('window_data_list')

        self.assertIsInstance(l_s, list)

        self.assertIsInstance(l_s[0], int)
        self.assertEqual(l_s[0], 1)

        self.assertIsInstance(l_s[1], float)
        self.assertEqual(l_s[1], 2.0)

        self.assertIsInstance(l_s[2], QColor)
        self.assertEqual(l_s[2].name(), "#ff0000")
        self.assertIsInstance(l_s[2], QColor)
        self.assertEqual(l_s[2].name(), "#ff0000")

        self.assertIsInstance(l_s[3], str)
        self.assertEqual(l_s[3], 'dp')

        print(dict_s)

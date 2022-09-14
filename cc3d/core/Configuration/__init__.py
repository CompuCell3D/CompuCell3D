# NOTE
# DefaultSettings.py defines location of the global setting file
# SettingUtils.py provides most functionality as far as Setting lookup, loading writing etc...


# NOTE 2
# To add new setting edit glolbal _settings.xml and create placeholder for the new setting
# e.g.
# add
# <e Name="WindowsLayout" Type="dict">
# </e>

# Now this setting is treated as "known setting " and you can manipulat it using Configuration set/get
# setting fuctions/. For more complex settings you may need to write
# some auxiliary functions facilitating translation from Setting format to Python format - this usually applies to
# e.g. dictionaries of dictionaries

# todo - handle bad file format for settings
# todo - at the beginning read all settings and make sure there are no issues in the stored settings
# todo - see if updateFieldsParams() function is needed
# todo - see if we need syncPreferences

from .SettingUtils import *
from .SettingUtils import _global_setting_path
from .Configuration import *
from .settingdict import SettingsSQL

LATTICE_TYPES = {"Square": 1, "Hexagonal": 2}

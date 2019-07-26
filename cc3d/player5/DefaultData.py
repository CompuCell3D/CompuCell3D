import os

# ICONS' PATH
_path = os.path.abspath(os.path.dirname(__file__))
# _path = os.path.abspath(os.path.join(_path+'../../../'))
icons_dir = os.path.abspath(os.path.join(_path, 'icons'))

def getIconsDir():return icons_dir

def getIconPath(icon_name):

    return os.path.abspath(os.path.join(getIconsDir(),icon_name))

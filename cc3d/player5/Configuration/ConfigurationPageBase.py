# Don't know why I need this class

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class ConfigurationPageBase(QWidget):
    """
    Class implementing the base class for all configuration pages.
    """
    def __init__(self):
        """
        Constructor
        """
        QWidget.__init__(self)

    def selectColor(self, button, colorVar):
        """
        Public method used by the color selection buttons.
        
        @param button reference to a QButton to show the color on
        @param colorVar reference to the variable containing the color (QColor)
        @return selected color (QColor)
        """
        color = QColorDialog.getColor(colorVar)
        if color.isValid():
            size = button.iconSize()
            pm = QPixmap(size.width(), size.height())
            pm.fill(color)
            button.setIcon(QIcon(pm)) 
        else:
            color = colorVar
            
        return color

    def initColor(self, configColor, button, prefMethod):
        """
        Public method to initialize a color selection button.
        
        @param colorstr color to be set (string)
        @param button reference to a QButton to show the color on
        @param prefMethod preferences method to get the color
        @return reference to the created color (QColor)
        """
        color = prefMethod(configColor)
        size = button.size()
        pm = QPixmap(size.width(), size.height())
        pm.fill(color)
        button.setIconSize(pm.size())
        button.setIcon(QIcon(pm))
        return color

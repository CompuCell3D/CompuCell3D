from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets


class FancyCombo(QtWidgets.QComboBox):

    def __init__(self, parent=None):
        super(FancyCombo, self).__init__(parent)

    def setValue(self,val):
        print('THIS IS VALUE =', val)

    def value(self):
        return self.currentText()

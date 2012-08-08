from PyQt4.QtCore import *
from PyQt4.QtGui import *

# Can be used for advanced editing in the Model Editor.
class SimDelegate(QItemDelegate):

    def __init__(self, parent=None):
        QItemDelegate.__init__(self, parent)
        
    def createEditor(self, parent, option, index):
        # Create editor object of QLineEdit
        if index.column() == 1:
            editor = QLineEdit(parent)
            self.connect(editor, SIGNAL("returnPressed()"), self.commitAndCloseEditor)
            return editor
        else:
            return QItemDelegate.createEditor(self, parent, option, index)

    def commitAndCloseEditor(self):
        editor = self.sender()
        if isinstance(editor, (QLineEdit)):
            
            # self.emit(SIGNAL("commitData(QWidget*)"), editor)
            self.emit(SIGNAL("closeEditor(QWidget*)"), editor)

    def setEditorData(self, editor, index):
        text = index.model().data(index, Qt.DisplayRole).toString()
        if index.column() == 1:
            editor.setText(text)
        else:
            QItemDelegate.setEditorData(self, editor, index)
    
    def setModelData(self, editor, model, index):
        # Method uses model.setData()! 
        # Make sure that you implemented setData() method
        if index.column() == 1:
            model.setData(index, QVariant(editor.text()))
        else:
            QItemDelegate.setModelData(self, editor, model, index)

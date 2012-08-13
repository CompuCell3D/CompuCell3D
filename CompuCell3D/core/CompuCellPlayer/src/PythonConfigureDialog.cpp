#include "PythonConfigureDialog.h"
#include "PythonConfigureData.h"
#include <QFileDialog>

PythonConfigureDialog::PythonConfigureDialog(QWidget *parent ):QDialog(parent){
   ui.setupUi(this);
   connect( ui.browseButton, SIGNAL(clicked()), this, SLOT(getFileName()) );
}

void PythonConfigureDialog::getFileName(){
   ui.pythonFileNameField->setText(QFileDialog::getOpenFileName(this,
                    "Choose Python file to open",
                    ".",
                    "Python Scripts (*.py)"));
}

void PythonConfigureDialog::loadCurrentValues(PythonConfigureData const & _data){
    using namespace std;
    
    ui.pythonFileNameField->setText(_data.pythonFileName);
}



#ifndef PYTHONCONFIGUREDIALOGFORM_H
#define PYTHONCONFIGUREDIALOGFORM_H

#include <ui_PythonConfigureDialog.h>

class PythonConfigureData;

class PythonConfigureDialog : public QDialog
{
    Q_OBJECT

public:
  PythonConfigureDialog(QWidget *parent = 0);

   Ui_PythonConfigureDialog &getUi(){return ui;}
   void loadCurrentValues(PythonConfigureData const & _data);

private slots:
   void getFileName();

private:
    Ui_PythonConfigureDialog ui;
};

#endif

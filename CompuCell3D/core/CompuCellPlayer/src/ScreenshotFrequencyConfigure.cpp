/****************************************************************************
**
** Copyright (C) 2005-2005 Trolltech AS. All rights reserved.
**
** This file is part of the example classes of the Qt Toolkit.
**
** This file may be used under the terms of the GNU General Public
** License version 2.0 as published by the Free Software Foundation
** and appearing in the file LICENSE.GPL included in the packaging of
** this file.  Please review the following information to ensure GNU
** General Public Licensing requirements will be met:
** http://www.trolltech.com/products/qt/opensource.html
**
** If you are unsure which license is appropriate for your use, please
** review the following information:
** http://www.trolltech.com/products/qt/licensing.html or contact the
** sales department at sales@trolltech.com.
**
** This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
** WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
**
****************************************************************************/

#include <QtGui>

#include "ScreenshotFrequencyConfigure.h"
#include <limits>


ScreenshotFrequencyConfigureForm::ScreenshotFrequencyConfigureForm(QWidget *parent)
    : QDialog(parent)
{
    ui.setupUi(this);
}

void ScreenshotFrequencyConfigureForm::loadCurrentValues(int _value,int _valueScreenUpdate,bool _noOutputFlag){

   ui.freqSpinBox->setMinimum(1);
   ui.freqSpinBox->setMaximum(std::numeric_limits<int>::max()-1);
   ui.freqSpinBox->setValue(_value);

   ui.screenUpdateSpinBox->setMinimum(1);
//    ui.screenUpdateSpinBox->setMaximum(std::numeric_limits<int>::max()-1);
   ui.screenUpdateSpinBox->setMaximum(std::numeric_limits<int>::max()-1);
   ui.screenUpdateSpinBox->setValue(_valueScreenUpdate);

   ui.noOutputCheckBox->setChecked(_noOutputFlag);

}
// void CalculatorForm::on_inputSpinBox1_valueChanged(int value)
// {
//     ui.outputWidget->setText(QString::number(value + ui.inputSpinBox2->value()));
// }
// 
// void CalculatorForm::on_inputSpinBox2_valueChanged(int value)
// {
//     ui.outputWidget->setText(QString::number(value + ui.inputSpinBox1->value()));
// }

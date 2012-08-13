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

#include "VectorFieldPlotConfigure.h"

VectorFieldPlotConfigureForm::VectorFieldPlotConfigureForm(QWidget *parent)
    : QDialog(parent)
{
    ui.setupUi(this);
    connect( ui.arrowColorButton, SIGNAL(clicked()), this, SLOT(changeArrowColor()) );
}

void VectorFieldPlotConfigureForm::changeArrowColor(){
   QColor color=QColorDialog::getColor();
   QPalette palette;
   palette.setColor(ui.arrowColorButton->backgroundRole(), color);
   ui.arrowColorButton->setPalette(palette);
}

void VectorFieldPlotConfigureForm::loadCurrentValues(float min, bool minFlag, float max , bool maxFlag, int _arrowLength,unsigned int numberOfLegendBoxes, unsigned int numberAccuracy, bool _legendEnable,bool _overlayVectorCellFields, bool _scaleArrows, bool _fixedArrowColorFlag){

   ui.minLineEdit->setText(QString().setNum(min));
   ui.maxLineEdit->setText(QString().setNum(max));
    if(minFlag){
      ui.minCheckBox->setChecked(true);
    }else{
      ui.minCheckBox->setChecked(false);
      ui.minLineEdit->setEnabled(false);
    }
    
    if(maxFlag){
      ui.maxCheckBox->setChecked(true);

    }else{
      ui.maxCheckBox->setChecked(false);
      ui.maxLineEdit->setEnabled(false);
    }
    ui.boxSpinBox->setMinimum(2);
    ui.boxSpinBox->setMaximum(99);
    
    ui.accuracySpinBox->setMinimum(0);
    ui.accuracySpinBox->setMaximum(5);
   
    ui.boxSpinBox->setValue(numberOfLegendBoxes);
    ui.accuracySpinBox->setValue(numberAccuracy);

    ui.showLegendBox->setChecked(_legendEnable);
    
    if(!ui.showLegendBox->isChecked()){
      ui.boxSpinBox->setEnabled(false);
      ui.accuracySpinBox->setEnabled(false);
    }

    ui.arrowLengthSpinBox->setValue(_arrowLength);



    ui.overlayVectorCellCheckBox->setChecked(_overlayVectorCellFields);
    ui.scaleArrowsCheckBox->setChecked(_scaleArrows);
    ui.fixedArrowColorCheckBox->setChecked(_fixedArrowColorFlag);

  return;  
}


// void ColormapPlotConfigureForm::loadCurrentValues(int _value){
//    ui.freqSpinBox->setValue(_value);
// }
// void CalculatorForm::on_inputSpinBox1_valueChanged(int value)
// {
//     ui.outputWidget->setText(QString::number(value + ui.inputSpinBox2->value()));
// }
// 
// void CalculatorForm::on_inputSpinBox2_valueChanged(int value)
// {
//     ui.outputWidget->setText(QString::number(value + ui.inputSpinBox1->value()));
// }

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

#include "SimulationFileOpenDialog.h"


#include <vector>

using namespace std;

SimulationFileOpenDialogForm::SimulationFileOpenDialogForm(QWidget *parent)
    : QDialog(parent)
{
    ui.setupUi(this);
    connect( ui.xmlBrowseButton, SIGNAL(clicked()), this, SLOT(getXMLFileName()) );
    connect( ui.pythonBrowseButton, SIGNAL(clicked()), this, SLOT(getPythonFileName()) );
//    connect( ui.useXMLCheckBox, SIGNAL(clicked()), this, SLOT(xmlChecked()) );
//    connect( ui.pythonScriptCheckBox, SIGNAL(clicked()), this, SLOT(pythonChecked()) );
}


void SimulationFileOpenDialogForm::getXMLFileName(){
   ui.xmlFileLineEdit->setText(QFileDialog::getOpenFileName(this,
                    "Choose XML file to open",
                    ".",
                    "Simulation Files (*.xml)"));
}

void SimulationFileOpenDialogForm::getPythonFileName(){
   ui.pythonFileLineEdit->setText(QFileDialog::getOpenFileName(this,
                    "Choose Python file to open",
                    ".",
                    "Python scripts(*.py)"));
}
/*
void SimulationFileOpenDialogForm::xmlChecked()
{
	if (ui.useXMLCheckBox->isChecked())
	{
		ui.pythonScriptCheckBox->setChecked(false);
	}
}

void SimulationFileOpenDialogForm::pythonChecked()
{
	if (ui.pythonScriptCheckBox->isChecked())
	{
		ui.useXMLCheckBox->setChecked(false);
	}
}
*/
void SimulationFileOpenDialogForm::loadCurrentValues(const QString & _xmlFileName, bool _useXMLFileFlag , QString _pythonFileName, bool _pythonCheckBoxState){
/*
	if (!_useXMLFileFlag)
	{
		ui.xmlBrowseButton->setDisabled(true);
	}
 
	if (!_pythonCheckBoxState)
	{
		ui.pythonBrowseButton->setDisabled(true);
	}
*/
	ui.xmlFileLineEdit->setText(_xmlFileName);
	ui.pythonFileLineEdit->setText(_pythonFileName);
	ui.pythonScriptCheckBox->setChecked(_pythonCheckBoxState);
	ui.useXMLCheckBox->setChecked(_useXMLFileFlag);

	ui.pythonFileLineEdit->setEnabled(_pythonCheckBoxState);
	ui.xmlFileLineEdit->setEnabled(_useXMLFileFlag);
}

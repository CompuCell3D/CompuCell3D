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

#include "Configure3DDialog.h"


#include <vector>

using namespace std;

Configure3DDialogForm::Configure3DDialogForm(QWidget *parent)
    : QDialog(parent)
{
    ui.setupUi(this);

}


void Configure3DDialogForm::loadCurrentValues(Configure3DData const & _data){
  
  //set limits in such a way that image enlargements are done gradually
  //this partially prevents users from entering bogus values for the image size 
   ui.xSpinBox->setMinimum(0);
   ui.xSpinBox->setMaximum(_data.sizeX*2);
   ui.ySpinBox->setMinimum(0);
   ui.ySpinBox->setMaximum(_data.sizeY*2);
   ui.zSpinBox->setMinimum(0);
   ui.zSpinBox->setMaximum(_data.sizeZ*2);
   
  
  ui.xSpinBox->setValue(_data.sizeX);
  ui.ySpinBox->setValue(_data.sizeY);
  ui.zSpinBox->setValue(_data.sizeZ);
  ui.xRotSpinBox->setValue(_data.rotationX);
  ui.yRotSpinBox->setValue(_data.rotationY);
  ui.zRotSpinBox->setValue(_data.rotationZ);
  
}

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

#ifndef CELLTYPECOLORCONFIGURERFORM_H
#define CELLTYPECOLORCONFIGURERFORM_H

#include "ui_CellTypeColorConfigure.h"
#include <map>

class CellTypeColorConfigureForm : public QDialog
{
    Q_OBJECT

public:
    CellTypeColorConfigureForm(QWidget *parent = 0);
    virtual ~CellTypeColorConfigureForm(){};
    Ui_CellTypeColorDialog &getUi(){return ui;};
    void loadCurrentValues(std::map<ushort,QColor> typeColorMap);

private slots:
    void on_borderColorButton_clicked();
    void on_contourColorButton_clicked();
    void on_typeColorTable_cellClicked(int,int);
    void on_newCellTypeButton_clicked();
	void on_deleteCellTypeButton_clicked();
	void selectionChanged();

private:
   Ui_CellTypeColorDialog ui;
    
};

#endif

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

#include "CellTypeColorConfigure.h"
#include <QItemSelectionModel>
#include "ColorItem.h"
#include <iostream>
#include <list>

using namespace std;

CellTypeColorConfigureForm::CellTypeColorConfigureForm(QWidget *parent)
    : QDialog(parent)
{
    ui.setupUi(this);

	connect( ui.typeColorTable, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()) );
//	connect( ui.deleteCellTypeButton, SIGNAL(clicked()), this, SLOT(on_deleteCellTypeButton_clicked()) );
}

void CellTypeColorConfigureForm::loadCurrentValues(std::map<ushort,QColor> typeColorMap)
{
//         using namespace std;

    ui.typeColorTable->setColumnCount(2);
    ui.typeColorTable->setRowCount(typeColorMap.size());

	int i=0;
   for(map<unsigned short,QColor>::iterator mitr=typeColorMap.begin(); 
        mitr != typeColorMap.end() ;
        ++mitr)
	{
		
		QTableWidgetItem *numberItem=new QTableWidgetItem(tr("%1").arg(mitr->first));//(QString(QString().setNum(mitr->first) ));
		ui.typeColorTable->setItem(i,0,numberItem );
		QTableWidgetItem *item=new QTableWidgetItem;
		item->setBackgroundColor(mitr->second);
		ui.typeColorTable->setItem( i, 1, item);
		//qDebug() << it0->text() << "\t" << it1->text();
		++i;
	}
}



void CellTypeColorConfigureForm::on_borderColorButton_clicked()
{	
	QColor color = QColorDialog::getColor();
	QPalette palette;

	if(color.isValid())
	{
		palette.setColor(ui.borderColorLabel->backgroundRole(), color);		
	}
	else
	{
		color = ui.borderColorLabel->palette().color(ui.borderColorLabel->backgroundRole());
		palette.setColor(ui.borderColorLabel->backgroundRole(), color);		
	}

	ui.borderColorLabel->setPalette(palette);
}

void CellTypeColorConfigureForm::on_contourColorButton_clicked()
{
	QColor color = QColorDialog::getColor();
	QPalette palette;

	if(color.isValid())
	{
		palette.setColor(ui.contourColorLabel->backgroundRole(), color);		
	}
	else
	{
		color = ui.contourColorLabel->palette().color(ui.contourColorLabel->backgroundRole());
		palette.setColor(ui.contourColorLabel->backgroundRole(), color);		
	}

	ui.contourColorLabel->setPalette(palette);
}

void CellTypeColorConfigureForm::on_typeColorTable_cellClicked(int row,int col){

  if(col==0)
    return;//only color cell can be changed

  QColor color=QColorDialog::getColor();

   QTableWidgetItem *item=ui.typeColorTable->item(row,col);
   item->setBackgroundColor(color);

//    cerr<<"Type="<<ui.typeColorTable->item(row,0)->text().toStdString()<<" color="<<color.name().toStdString()<<endl;
   
}

void CellTypeColorConfigureForm::on_newCellTypeButton_clicked()
{
	ui.typeColorTable->insertRow(ui.typeColorTable->rowCount());
	int col,row;
	row=ui.typeColorTable->rowCount()-1;
	col=1;

	QTableWidgetItem *item=new QTableWidgetItem;
	item->setBackgroundColor(QColor("black"));
	ui.typeColorTable->setItem( row,col, item);
}

void CellTypeColorConfigureForm::selectionChanged()
{

	QItemSelectionModel * sm = ui.typeColorTable->selectionModel();

	if (!sm->selectedRows().isEmpty())
		ui.deleteCellTypeButton->setEnabled(true);

	if (sm->selectedRows().isEmpty())
		ui.deleteCellTypeButton->setEnabled(false);
}

void CellTypeColorConfigureForm::on_deleteCellTypeButton_clicked()
{
	QItemSelectionModel * sm = ui.typeColorTable->selectionModel();

	if (!sm->selectedRows().isEmpty() && ui.deleteCellTypeButton->isEnabled())
	{
		QList<QModelIndex> modelIndexList = sm->selectedRows();
		QList<QModelIndex>::iterator itr;

		// After removing the row with removeRow() function it renumerates the rows!!!
		// Moreover, the order of the elements in the list corresponds to the order in which the rows were selected!!! It means that to use them you need first to store the indexes and sort them out before using them.

		list<int> indexes;
		list<int>::iterator indItr;

		for (itr = modelIndexList.begin(); itr !=  modelIndexList.end(); ++itr)
		{
			indexes.push_back(itr->row()); // Removes row at the index itr->row
		}	
		indexes.sort();

		int shift = 0;

		for (indItr = indexes.begin(); indItr !=  indexes.end(); ++indItr)
		{
			ui.typeColorTable->removeRow(*indItr - shift); // Removes row at the index itr->row
			shift++;
		}	
	}
}



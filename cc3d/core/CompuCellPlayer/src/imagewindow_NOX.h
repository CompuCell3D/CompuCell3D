/****************************************************************************
**
** Copyright (C) 2004-2005 Trolltech AS. All rights reserved.
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

#ifndef IMAGEWINDOW_NOX_H
#define IMAGEWINDOW_NOX_H

#include <QMainWindow>
#include <imagewindowBase.h>

#include <QImage>

#include <QMutex>

#include <QScrollArea>

#include <ScreenshotDescription.h>
#include <UniversalGraphicsSettings.h>
#include "transactionthread.h"
#include "mainCC3D.h"
#include <map>
#include <string>
#include <fstream>


class QAction;
class QMenu;
class QTextEdit;

class Transaction;
class QAction;
class QLabel;
class QPainter;
class QPushButton;
class QToolBar;
class QRadioButton;
class QButtonGroup;
class QSpinBox;
class QCheckBox;
class QComboBox;
class QHGroupBox;
class QLineEdit;
class QLCDNumber;
class Display3D;




class ScreenshotData;
class GraphicsBase;

class GraphicsDataFields;

class Configure3DData;

class ImageWindow_NOX:public QObject, public ImageWindowBase
{
//     Q_OBJECT

public:
   ImageWindow_NOX(bool _silent=false);
   virtual ~ImageWindow_NOX() ;

protected:
//     void closeEvent(QCloseEvent *event);
    virtual void  customEvent(QEvent *event);
//     virtual void simulation();
    virtual void initializeImageWindow(bool _silent);
	 virtual bool errorHandler(QString header, QString text);


    
};

#endif

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

#ifndef IMAGEWINDOWBASE_H
#define IMAGEWINDOWBASE_H

// #include <QMainWindow>

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
#include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>

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
class PlayerSettings;



class ScreenshotData;
class GraphicsBase;

class GraphicsDataFields;

class Configure3DData;

class ImageWindowBase
{
//     Q_OBJECT

public:

    ImageWindowBase(bool _silent=false);
    virtual ~ImageWindowBase() ;
   void setXMLFile(const QString &fileName);
   void setOutputDirectory(const QString &dirName);
   void setNoOutputFlag(bool _flag);
   
   void setRunPythonFlag(bool _pythonFlag);
   void setPythonScript(const QString &fileName);
   void setScreenshotDescriptionFileName(const QString & scrDesFileName);
   virtual void startSimulation();
   void setXServerFlag(bool _xServerFlag);
   bool getXServerFlag();
	void updateSimulationFileNames();

protected:
    virtual void closeEvent(QCloseEvent *event);
    virtual void customEvent(QEvent *event);
    virtual void simulation();
    virtual void initializeImageWindow(bool _silent);
    void updatePlayerSettings(CompuCell3D::PlayerSettings & playerSettings);
	 virtual bool errorHandler(QString header, QString text);

protected:
//     bool maybeSave();
//     void loadFile(const QString &fileName);
//     bool saveFile(const QString &fileName);




    const unsigned int maxScreenshotListLength;
    const QString screenshotCoreName;
    QString screenshotDescriptionFileName;

/*    void initProjectionXYData();
    void initProjectionXZData();
    void initProjectionYZData();*/
    

//     void saveScrDesFile(const QString &fileName);
    virtual void setCurrentFile(const QString &fileName);
    QString strippedName(const QString &fullFileName);
    virtual void addTransaction(Transaction *transact);
    void initializeGraphicsPtrVec();
    void initializeGraphicsPtr(GraphicsBase * _graphicsPtr);
    QString screenshotCoreNameFromScreenshotDescription(const ScreenshotDescription & _scrDsc);

//     void drawField();
//     void drawField2D();
//     void drawField3D();
//     QString screenshotName2D();
//     QString screenshotName3D(Configure3DData & _data3);

//     void (ImageWindowBase::*draw)();

    void outputScreenshot();
    virtual void takeCurrentScreenshot(const std::string &_imageFileFullName, const std::string &_imageFileName);
    void writeSettings();
    void readSettings();
    void readScreenshotDescriptionList(std::list<ScreenshotDescription> & _screenshotDescriptionList, const std::string &fileName);
    void produceScreenshotDataList(const std::list<ScreenshotDescription> & _screenshotDescriptionList);
    QPainter *painter;

    std::list<ScreenshotData*> screenshotDataList;
    std::list<ScreenshotDescription> screenshotDescriptionList;
    
    std::vector<GraphicsBase *> graphicsPtrVec;
    
    GraphicsBase *currentGraphicsPtr;
    GraphicsBase * display3D;
    Display3D *ptr3D;

    GraphicsBase * glWidget;

    GraphicsBase * graphics2DPtr;
     


    Projection2DData *projDataPtr;

    GraphicsDataFields * graphFieldsPtr;
        
    unsigned int screenshotCounter;
    std::string imageCoreFileName;
    std::string imageFileExtension;

    ///Mutexes  
    QMutex mutex;
    QMutex mutexStartPause;
    QMutex mutexFieldDraw;
    QMutex mutexTransaction;

    QSemaphore * bufferFillUsedSemPtr;
    QSemaphore * bufferFillFreeSemPtr;
    
    TransactionThread computeThread;

    Projection2DData xyProjData;
    Projection2DData xzProjData;
    Projection2DData yzProjData;

    unsigned int screenshotFrequency;
    unsigned int screenUpdateFrequency;
    
    QString imageFormat;
    QString curFile;
    QString curFileStripped;
    QString fileXML;
    QString fileXMLStripped;

    QString tmpDirName;
	 //QString fullTmpDirName;
    QString inventoryFileName;
    QString simulationRootDir;
    std::ofstream inventoryFile;
    bool noOutputFlag;
    
   //universal settings - i.e. valid for 2D and 3D drawing
   UniversalGraphicsSettings univGraphSet;
   
   float minConcentration;
   bool minConcentrationFixed;
   float maxConcentration;
   bool maxConcentrationFixed;
   unsigned int numberOfLegendBoxes;
   unsigned int numberAccuracy;
   bool legendEnable;
   
   float minMagnitude;
   bool minMagnitudeFixed;
   float maxMagnitude;
   bool maxMagnitudeFixed;
   float arrowLength;
   unsigned int numberOfLegendBoxesVector;
   unsigned int numberAccuracyVector;
   bool legendEnableVector;
   bool overlayVectorCellFields;
   bool scaleArrows;
   bool fixedArrowColorFlag;
   
   ScreenshotData *scshData;
   ScreenshotData *scshData1;
   int numScreenNameDigits;
   bool silent;
   bool xServerFlag;
   bool runPythonFlag;
   bool useXMLFileFlag;
   bool explicitSetXMLFileFlag;
	bool gotErrorFlag;

	bool closePlayerAfterSimulationDone;

   bool saveSettingsXML;
   bool advancedSettingsXMLOn;
	
   PythonConfigureData pyConfData;

   unsigned int mcStep;
   CC3DTransaction *transactionCC3D;
   CompuCell3D::LatticeType latticeType;

	// Stops execution of the simulation thread
	bool stopThread;//TEMP 
};

#endif

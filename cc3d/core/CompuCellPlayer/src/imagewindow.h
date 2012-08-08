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

#ifndef IMAGEWINDOW_H
#define IMAGEWINDOW_H

#include <QMainWindow>
#include <imagewindowBase.h>
#include <QImage>
#include <QVTKWidget.h>
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


class QVTKWidget;
class vtkCylinderSource;
class vtkPolyDataMapper;
class vtkActor;
class vtkRenderer;
class Display3D;


class ScreenshotData;
class GraphicsBase;
class GraphicsDataFields;

class Configure3DData;
//class Smile;

class ImageWindow : public QMainWindow, public ImageWindowBase
{
    Q_OBJECT

public:
    ImageWindow(bool _silent=false);
    ~ImageWindow() ;
    virtual void startSimulation();

protected:
    virtual void closeEvent(QCloseEvent *event);
    virtual void customEvent(QEvent *event);
    virtual void initializeImageWindow(bool _silent);

private slots:
    void newFile();
    bool save();

    void open();
    bool saveScrDes();
 
	void init();
	void initStatic();

    void simulationStart();
    void simulationPause();
    void simulationStop();
    void simulationSerialize();
    void zoomIn();
    void zoomOut();
    
    ///projections and 3D
    
    void changeProjectionXY(bool);
    void alterZCoordinate(int);
    void changeProjectionXZ(bool);
    void alterYCoordinate(int);
    void changeProjectionYZ(bool);
    void alterXCoordinate(int);

    void switch3D(bool);
    
    void bordersDisplay(bool);
    void contoursDisplay(bool);
    void concentrationLimitsDisplay(bool);

    

    void frequencyUpdate(int);
    void plotTypeUpdate( const QString &);

    void configureColorMapPlot();
    void configureVectorFieldPlot();
    void configureScreenshotFrequency();
    void configureTypes3DInvisible();
    void configureCellTypeColors();
    void configure3DDisplay();
	 
	 void setClosePlayerAfterSimulationDone(bool);

    void saveSettings();
    void runPython(bool);

	void about();
	void addScreenshot();

	void dumpValues(QString location);
	void dumpUniversalGraphicsSettings();

protected:
	bool warning(QString header, QString text);
	bool warningWithMutexRelease(QString header, QString text);
	virtual bool errorHandler(QString header, QString text);

    bool maybeSave();
    void loadFile(const QString &fileName);
    bool saveFile(const QString &fileName);
    virtual void setCurrentFile(const QString &fileName);
    virtual void addTransaction(Transaction *transact);
    virtual void takeCurrentScreenshot(const std::string &_imageFileFullName,const std::string &_imageFileName);
   
    void stopSimulation();
	 void tryStopSimulation();
    QTextEdit *textEdit;
    QMenu *editMenu;
    QToolBar *fileToolBar;
    QToolBar *editToolBar;
    QAction *newAct;
    QAction *saveAct;
    QAction *saveAsAct;
    QAction *cutAct;
    QAction *copyAct;
    QAction *pasteAct;


/////////////////////////////////////////
//     const unsigned int maxScreenshotListLength;
//     const QString screenshotCoreName;
//     QString screenshotDescriptionFileName;

    void initProjectionXYData();
    void initProjectionXZData();
    void initProjectionYZData();
    
    void createActions();
    void createMenus();
    void createSteeringBar();
    void createStatusBar();

    void saveScrDesFile(const QString &fileName);
//     void setCurrentFile(const QString &fileName);
//     QString strippedName(const QString &fullFileName);
//     void addTransaction(Transaction *transact);

    void drawField();
    void drawField2D();
    void drawField3D();
    QString screenshotName2D();
    QString screenshotName3D(Configure3DData & _data3);

    void (ImageWindow::*draw)();

//     void outputScreenshot();       


//     QPainter *painter;

//     std::list<ScreenshotData*> screenshotDataList;
//     std::list<ScreenshotDescription> screenshotDescriptionList;
    
//     std::vector<GraphicsBase *> graphicsPtrVec;
    
//     GraphicsBase *currentGraphicsPtr;
//     GraphicsBase * display3D;
    QVTKWidget * display3Dqvtk;
//     Display3D *ptr3D;

//     GraphicsBase * glWidget;

//     GraphicsBase * graphics2DPtr;
     


//     Projection2DData *projDataPtr;

//     GraphicsDataFields * graphFieldsPtr;
        
//     unsigned int screenshotCounter;
//     std::string imageCoreFileName;
//     std::string imageFileExtension;

    QScrollArea *scrollView;

    ///Mutexes  
//     QMutex mutex;
//     QMutex mutexStartPause;
//     QMutex mutexFieldDraw;
//     QMutex mutexTransaction;
// 
//     QSemaphore * bufferFillUsedSemPtr;
//     QSemaphore * bufferFillFreeSemPtr;
//     
//     TransactionThread thread;
    //steering bar buttons
    QToolBar * steeringBar;
    QRadioButton *xyButton, *xzButton , *yzButton;
    QButtonGroup *crossSectionButtonsG;

    QPushButton *updateViewButton;
    
    QSpinBox *xSpinBox;
    QSpinBox *ySpinBox;
    QSpinBox *zSpinBox;
    Projection2DData xyProjData;
    Projection2DData xzProjData;
    Projection2DData yzProjData;

    QRadioButton *display3DButton;

    QPushButton * recordButton;
    

    QHGroupBox *zoomBox;
/*    unsigned int screenshotFrequency;
    unsigned int screenUpdateFrequency;*/
    
    ///treshold box
    QLabel *minMaxConcentrationLabel;

    
    QComboBox *plotTypeComboBox;
    QLabel *plotTypeComboBoxName;

    ///MC Step
    QLabel *mcStepLabel;
       
    //QLabel *imageLabel;
    QLabel *infoLabel;
    QLabel *modLabel;
/*    QString imageFormat;
    QString curFile;
    QString curFileStripped;
    QString tmpDirName;
    QString inventoryFileName;
    QString simulationRootDir;
    std::ofstream inventoryFile;*/
    
    bool modified;
    bool drawingAllowed;
    ///Popup Menus  
    
    ///File  
    QMenu *fileMenu;
    QAction *openAct;
    QAction *saveScrDesAct;
    QAction *exitAct;
    

    ///configure
    QMenu *configureMenu;
    QAction *typeColorAct;
    QAction *colorMapPlotAct;
    QAction *vectorFieldPlotAct;
    QAction *screenshotFrequencyAct;
    QAction * types3DInvisibleAct;
    QAction * configure3DDisplayAct;
	 QAction * closePlayerAfterSimulationDoneAct;
    
    QAction * saveSettingsAct;
    
    ///simulation   
    QMenu *simulationMenu;
    QAction *simulationAct;
    QAction *simulationPauseAct;
    QAction *simulationStopAct;
    QAction *simulationSerializeAct;
    

        
    ///zoom  
    QMenu *zoomMenu;
    QAction *zoomInAct;
    QAction *zoomOutAct;

    ///show  
    QMenu *showMenu;
    QAction *showBordersAct;
    QAction *showContoursAct;
    QAction *showConcentrationLimitsAct;
    

    QMenu *pythonMenu;
    QAction *runPythonAct;
    QAction *configurePythonAct;

    
    ///help    
    QMenu *helpMenu;
    QAction *aboutAct;
    QAction *aboutQtAct;
    
   ///universal settings - i.e. valid for 2D and 3D drawing
//    UniversalGraphicsSettings univGraphSet;
//    
//    float minConcentration;
//    bool minConcentrationFixed;
//    float maxConcentration;
//    bool maxConcentrationFixed;
//    unsigned int numberOfLegendBoxes;
//    unsigned int numberAccuracy;
//    bool legendEnable;
//    
//    float minMagnitude;
//    bool minMagnitudeFixed;
//    float maxMagnitude;
//    bool maxMagnitudeFixed;
//    float arrowLength;
//    unsigned int numberOfLegendBoxesVector;
//    unsigned int numberAccuracyVector;
//    bool legendEnableVector;
//    
//    ScreenshotData *scshData;
//    ScreenshotData *scshData1;
//    int numScreenNameDigits;
//    bool silent;
//    bool runPythonFlag;
//    PythonConfigureData pyConfData;
// 
//    unsigned int mcStep;
//    CC3DTransaction *transactionCC3D;
   QVTKWidget *qvtkWidget;
//	Smile *smile;

/*   vtkCylinderSource* source;
   vtkPolyDataMapper* mapper;
   vtkActor* actor;
   vtkRenderer* ren;*/
    
};


#endif

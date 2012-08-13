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

#include <QtGui>
#include <QWidget>
#include <iostream>
#include "imagewindow_NOX.h"
#include "FileUtils.h"
#include "ScreenshotData.h"
#include <sstream>

/*#include "ScreenshotFrequencyConfigure.h"
#include "ColormapPlotConfigure.h"
#include "VectorFieldPlotConfigure.h"
#include "TypesThreeDConfigure.h"
#include "Configure3DDialog.h"
#include "CellTypeColorConfigure.h"
#include "PythonConfigureDialog.h"
#include "SimulationFileOpenDialog.h"*/
#include "ColorItem.h"
#include "Graphics2D_NOX.h"
#include "Display3D_NOX.h"
// #include "Display3D.h"


// #include <QVTKWidget.h>
// #include <vtkRenderWindow.h>
// #include <vtkRenderer.h>


// #include <vtkActor.h>
// #include <vtkRenderer.h>
// #include <vtkRenderWindow.h>
// #include "vtkCylinderSource.h"
// #include <vtkPolyDataMapper.h>



#include <CompuCell3D/Simulator.h>
//#include "glwidget.h"
// #include <QRadioButton>
// #include <QButtonGroup>

using namespace std;

void ImageWindow_NOX::initializeImageWindow(bool _silent){

    mcStep=0;
    runPythonFlag=false;  
    transactionCC3D=0;
    screenUpdateFrequency=1;
    numScreenNameDigits=5;//screenshots numbers will have  that many digits
    bufferFillUsedSemPtr=0;
    bufferFillFreeSemPtr=0;

    bufferFillUsedSemPtr=new QSemaphore(1);
    bufferFillFreeSemPtr=new QSemaphore(1);

    computeThread.setTargetObject(this);
    univGraphSet.zoomFactor=1;
    readSettings();///read settings  


    
    
    curFileStripped=strippedName(curFile);
    setCurrentFile(curFile);

    

    painter=0;
    
    screenshotCounter=0;
    
    
     graphFieldsPtr=new GraphicsDataFields();

     graphics2DPtr = new Graphics2D_NOX();
     graphics2DPtr->setId(QString("2D"));
     
     projDataPtr= &((Graphics2D_NOX*)graphics2DPtr)->projData;
 
//      graphics2DPtr->setGraphicsDataFieldPtr(graphFieldsPtr);
//      graphics2DPtr->setUnivGraphSetPtr(&univGraphSet);
// 
// 
// 
//      currentGraphicsPtr=graphics2DPtr;
//      currentGraphicsPtr->setCurrentPainitgFcnPtr(currentGraphicsPtr->getPaintLattice());
//  
//      graphicsPtrVec.push_back(graphics2DPtr);
//      graphicsPtrVec.push_back(display3D);
// 
//     
//     for(unsigned int i  = 0 ; i < graphicsPtrVec.size() ; ++i){
//          graphicsPtrVec[i]->setMinMagnitudeFixed(minMagnitudeFixed);
//          graphicsPtrVec[i]->setMaxMagnitudeFixed(maxMagnitudeFixed);
//          graphicsPtrVec[i]->setMinMagnitude(minMagnitude);
//          graphicsPtrVec[i]->setMaxMagnitude(maxMagnitude);
//          graphicsPtrVec[i]->setMaxMagnitude(maxMagnitude);
//          graphicsPtrVec[i]->setNumberOfLegendBoxes(numberOfLegendBoxes);
//          graphicsPtrVec[i]->setNumberAccuracy(numberAccuracy);
//          
//          graphicsPtrVec[i]->setArrowLength(arrowLength);
//          graphicsPtrVec[i]->setNumberOfLegendBoxesVector(numberOfLegendBoxesVector);
//          graphicsPtrVec[i]->setNumberAccuracyVector(numberAccuracyVector);
//          graphicsPtrVec[i]->setLegendEnableVector(legendEnableVector);
//       }

            
    inventoryFileName="graphicsList.grl";



    

}


ImageWindow_NOX::ImageWindow_NOX(bool _silent):ImageWindowBase(_silent)
{
    initializeImageWindow(_silent);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ImageWindow_NOX::~ImageWindow_NOX(){

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow_NOX::customEvent(QEvent *event){

QEvent::Type type=((CustomEvent*)event)->type();
 cerr<<"\t\t\t\t got event of the type:"<<type<<endl;
//    if (type == (QEvent::Type)TransactionStartEvent::TransactionStart) {
//       cerr<<"got START EVENT:"<<type<<endl;
//    }

   if (type == (QEvent::Type)TransactionStartEvent::TransactionStart) {

//       cerr<<" \n\n\n GOT START EVENT \n\n\n";
		gotErrorFlag=false;
      GraphicsBase &graphics2D = *graphics2DPtr;

      initializeGraphicsPtrVec();
      


      TransactionStartEvent *transactionStartEvent = (TransactionStartEvent*)(event);

//       cerr<<"graphFieldsPtr->getSizeL()="<<graphFieldsPtr->getSizeL()<<endl;
//       cerr<<"graphFieldsPtr->getSizeM()="<<graphFieldsPtr->getSizeM()<<endl;
//       cerr<<"graphFieldsPtr->getSizeN()="<<graphFieldsPtr->getSizeN()<<endl;



//      cerr<<" \n\n\n GOT START EVENT 1 \n\n\n";
      ///initialize graphics obj

      Configure3DData data3D;
      if(transactionStartEvent->playerSettings.rotationXFlag){
         data3D.rotationX=transactionStartEvent->playerSettings.rotationX;
      }

      if(transactionStartEvent->playerSettings.rotationYFlag){
         data3D.rotationY=transactionStartEvent->playerSettings.rotationY;
      }
      if(transactionStartEvent->playerSettings.rotationZFlag){
         data3D.rotationZ=transactionStartEvent->playerSettings.rotationZ;
      }

      if(transactionStartEvent->playerSettings.sizeX3DFlag){
         data3D.sizeX=transactionStartEvent->playerSettings.sizeX3D;
      }else{
         data3D.sizeX=graphFieldsPtr->getSizeL();
      }

      if(transactionStartEvent->playerSettings.sizeY3DFlag){
         data3D.sizeY=transactionStartEvent->playerSettings.sizeY3D;
      }else{
         data3D.sizeY=graphFieldsPtr->getSizeM();
      }

      if(transactionStartEvent->playerSettings.sizeZ3DFlag){
         data3D.sizeZ=transactionStartEvent->playerSettings.sizeZ3D;
      }else{
         data3D.sizeZ=graphFieldsPtr->getSizeN();
      }


      if(transactionStartEvent->playerSettings.advancedSettingsOn){
         advancedSettingsXMLOn=transactionStartEvent->playerSettings.advancedSettingsOn;
         if(transactionStartEvent->playerSettings.saveSettingsFlag)
            saveSettingsXML=transactionStartEvent->playerSettings.saveSettings;

         updatePlayerSettings(transactionStartEvent->playerSettings);
      }

      data3D.sizeL=graphFieldsPtr->getSizeL();
      data3D.sizeM=graphFieldsPtr->getSizeM();
      data3D.sizeN=graphFieldsPtr->getSizeN();
//uncomment here
//        ((Display3D*)display3D)->setInitialConfigure3DData(data3D);




///      projDataPtr->imageLabelPtr=imageLabel;
      projDataPtr->sizeL = graphFieldsPtr->getSizeL(); //change it!
      projDataPtr->sizeM = graphFieldsPtr->getSizeM(); //change it!

      //default configuration
      projDataPtr->xMin = 0;
      projDataPtr->xMax = graphFieldsPtr->getSizeL();
      projDataPtr->yMin = 0;
      projDataPtr->yMax = graphFieldsPtr->getSizeM();
      projDataPtr->zMin = graphFieldsPtr->getSizeN()/2;
      projDataPtr->zMax = graphFieldsPtr->getSizeN()/2+1;
      projDataPtr->projection="xy";

//       cerr<<" \n\n\n GOT START EVENT 2 \n\n\n";


      ///initialize Combo box with field types
      GraphicsDataFields::floatField3DNameMapItr_t mitr;



      ///make tmp directory for storing pictures
      if(tmpDirName.isEmpty() ){
         tmpDirName=curFileStripped;
         QDate date=QDate::currentDate();
         QTime time=QTime::currentTime();
         QString dirSuffix;
   
         dirSuffix+=QString("_");
         //dirSuffix+=date.toString();
			dirSuffix+=QString().setNum(date.month());;
         dirSuffix+=QString("_");
			dirSuffix+=QString().setNum(date.day());;
			dirSuffix+=QString("_");
			dirSuffix+=QString().setNum(date.year());;
			dirSuffix+=QString("___");
         dirSuffix+=time.toString();
         tmpDirName+=dirSuffix;
      }
	  tmpDirName.replace(QString("."),QString("_"));
	  tmpDirName.replace(QString(":"),QString("_"));


      if(!noOutputFlag){
         cerr<<"NAME OF THE TEMPORARY DIR: "<<tmpDirName.toStdString()<<endl;
         QDir dir;

			//Will extract storage path based on whether XML or Python script are specified
			QString storagePath;
			QFileInfo fileInfo(curFile);
			if(useXMLFileFlag || runPythonFlag){
				if(curFile!=QString("")){
					storagePath=fileInfo.absolutePath();
				}
				else{
					cerr<<"Empty File Name curFile="<<curFile.toStdString()<<endl;	
					exit(0);
				}
			}

			
			tmpDirName=storagePath+QDir::separator()+tmpDirName;
         bool dirOK=dir.mkdir(tmpDirName);
			

         if(!dirOK){
            cerr<<"could not create directory: "<<tmpDirName.toStdString()<<endl;
            cerr<<"Make sure that directory with this name is removed"<<endl;
            exit(0);
         }
			
         ///opening graphics list file

         QString path=dir.absolutePath();
         //simulationRootDir=QString(path+QString(QDir::separator())+tmpDirName);
			simulationRootDir=tmpDirName;
         QString fullInventoryFileName(simulationRootDir+QString(QDir::separator())+inventoryFileName);
         cerr<<fullInventoryFileName.toStdString()<<endl;
   
   
   
         inventoryFile.open(fullInventoryFileName.toStdString().c_str());
   
         ///copying xml file to folder with graphics
         QString xmlFileFullName=simulationRootDir+QString(QDir::separator())+curFileStripped;
         copyFile( curFile.toStdString().c_str() , xmlFileFullName.toStdString().c_str());
      }

      ///enabling zooms
//       if(univGraphSet.zoomFactor>1){
//          zoomOutAct->setEnabled(true);
//       }
//       zoomInAct->setEnabled(true);
      ///enabling show menu


//       if(univGraphSet.bordersOn){
//          showBordersAct->setChecked(true);
//       }
//       showBordersAct->setEnabled(true);
// 
//       if(univGraphSet.contoursOn){
//          showContoursAct->setChecked(true);
//       }
// 
//       if(univGraphSet.concentrationLimitsOn){
//          showConcentrationLimitsAct->setChecked(true);
//       }
//       showConcentrationLimitsAct->setEnabled(true);
// 
//       showContoursAct->setEnabled(true);



//       drawingAllowed=true;
//       ((Display3D*)display3D)->setDrawingAllowedFlag(true);


      ///init screenshot data
      if(/*silent &&*/ ! screenshotDescriptionFileName.isEmpty()){
         readScreenshotDescriptionList(screenshotDescriptionList,screenshotDescriptionFileName.toStdString());
         produceScreenshotDataList(screenshotDescriptionList);
      }


      //setting numScreenNameDigits;
      //this is not might be not greatest method for determining it, but it works
      ostringstream numStream;
      string numString;

      numStream<<transactionStartEvent->numSteps;

      numString=numStream.str();

      numScreenNameDigits=numString.size();

      return;

   }

   if (type == (QEvent::Type)TransactionRefreshEvent::TransactionRefresh) {
       TransactionRefreshEvent *refreshEvent = (TransactionRefreshEvent *)event;
//        ((Graphics2D_NOX*)graphics2DPtr)->getPixmap().fill();
//       ((Graphics2D_NOX*)graphics2DPtr)->getPixmap().fill(Qt::red);
      ///lock mutextransaction -prevents compucell computeThread "overrunning"

       //mutexTransaction.lock();
       bufferFillUsedSemPtr->acquire();
//        cerr<<"acquired bufferFillUsedSem in imagewindow"<<endl;

       mcStep=refreshEvent->mcStep;
//        drawField();

       ///lock mutextransaction
       //mutexTransaction.unlock();
// 
//       ///screenshot
      if(!noOutputFlag){
         outputScreenshot();
      }

      bufferFillFreeSemPtr->release();
      
      

      return;
   }

   if (type == (QEvent::Type)TransactionFinishEvent::TransactionFinish) {

      TransactionFinishEvent *finishEvent = (TransactionFinishEvent *)event;
       if(graphFieldsPtr){
           graphFieldsPtr->clearAllocatedFields();
       }
       
       cerr<<finishEvent->message.toStdString()<<endl;
      if(advancedSettingsXMLOn && !saveSettingsXML){//do not save settings if user used advanced XML settings and
         
      }else{
         writeSettings();
      }
      exit(0);

   }
	if(type == (QEvent::Type)TransactionErrorEvent::TransactionError){
		TransactionErrorEvent *errorEvent = (TransactionErrorEvent *)event;
		
		if(errorHandler(errorEvent->errorCategory, errorEvent->message))
			cerr<<"Error Acknowledged"<<endl;
		else
			cerr<<"Error Disregarded"<<endl;
		cerr<<"\n\n\n\n NOX error"<<endl;
		gotErrorFlag=true;
		exit(0);

	}

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool ImageWindow_NOX::errorHandler(QString header, QString text)
{
	// mutex.tryLock() actually takes the lock if the mutex is not locked!

	mutex.tryLock();
	mutexStartPause.tryLock();


	cerr<<"ERROR: "<<header.toStdString()<<endl<<text.toStdString()<<endl;
	return true;
	
}
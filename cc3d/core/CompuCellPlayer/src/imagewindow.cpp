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
#include <QMessageBox>

#ifdef __APPLE__
#include <QMacStyle>
#endif

#include <iostream>
#include "imagewindow.h"
#include "FileUtils.h"
#include "ScreenshotData.h"
#include <sstream>

#include "ScreenshotFrequencyConfigure.h"
#include "ColormapPlotConfigure.h"
#include "VectorFieldPlotConfigure.h"
#include "TypesThreeDConfigure.h"
#include "Configure3DDialog.h"
#include "CellTypeColorConfigure.h"
#include "PythonConfigureDialog.h"
#include "SimulationFileOpenDialog.h"
#include "ColorItem.h"
#include "Display3D.h"
#include "transaction.h"

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>


#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include "vtkCylinderSource.h"
#include <vtkPolyDataMapper.h>



#include <CompuCell3D/Simulator.h>
//#include "glwidget.h"
// #include <QRadioButton>
// #include <QButtonGroup>
//#include "smile.h"

// For initStatic() to initilize static attributes and methods
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include "CompuCell3D/Field3D/NeighborFinder.h"


class TransactionFinishEvent;

using namespace std;

ImageWindow::ImageWindow(bool _silent) : ImageWindowBase(_silent)
{
//	cerr << "ImageWindow::ImageWindow stopThread ADDRESS: " << &stopThread << "\t" << stopThread << "\n";  // TESTING THREADING
    initializeImageWindow(_silent);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ImageWindow::~ImageWindow(){

}

void ImageWindow::initializeImageWindow(bool _silent)
{
    //mcStep=0; //moved to init()
    runPythonFlag=false;  
    //transactionCC3D=0;	//? moved to init()
    screenUpdateFrequency=1;
    numScreenNameDigits=5;//screenshots numbers will have  that many digits
    bufferFillUsedSemPtr=0; //?
    bufferFillFreeSemPtr=0; //?

    bufferFillUsedSemPtr=new QSemaphore(1);
    bufferFillFreeSemPtr=new QSemaphore(1);

	computeThread.setStopThread(&stopThread); // passing pointer to stopThread   
    computeThread.setTargetObject((QObject*)this);

	//cerr << "ImageWindow::initializeImageWindow stopThread ADDRESS: " << &stopThread << "\t" << stopThread << "\n"; // TESTING THREADING

    univGraphSet.zoomFactor=1;
    readSettings();///read settings  
    
    createActions();
    createMenus();
    createSteeringBar();
    createStatusBar();

	 updateSimulationFileNames();
//    if(useXMLFileFlag && fileXML!="")
//	{
//		fileXMLStripped=strippedName(fileXML);
//		setCurrentFile(fileXML);
//		setXMLFile(fileXML);
//
////       curFileStripped=strippedName(curFile);
////       setCurrentFile(curFile);
//    }
//	else if (runPythonFlag && pyConfData.pythonFileName!="")
//	{
//		curFile=pyConfData.pythonFileName;
//		curFileStripped=strippedName(pyConfData.pythonFileName);
//		setCurrentFile(pyConfData.pythonFileName);
//    }
//	else
//	{
//		curFileStripped="default_simulation.xml";
//		setCurrentFile(pyConfData.pythonFileName);
//    }

	init();
	initStatic();
	dumpValues("ImageWindow::initializeImageWindow");

}

void ImageWindow::dumpValues(QString location)
{
	qDebug() << "\n!!!!!!!!!!!!!!!!!!!!!!!!!!! DUMPING VALUES: " << location << " !!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
	cerr << "mcStep: " << mcStep << "\n";
    cerr << "transactionCC3D: " << transactionCC3D << "\n";
	cerr << "&computeThread: " << &computeThread << "\n";
	cerr << "&univGraphSet: " << &univGraphSet << "\n";
	cerr << "&latticeType: " << &latticeType << "\n";
	cerr << "this: " << this << "\n";
	cerr << "stopThread: " << stopThread << "\n";
/*

    cerr << "runPythonFlag: " << runPythonFlag << "\n";  

    cerr << "screenUpdateFrequency: " << screenUpdateFrequency << "\n";
    cerr << "numScreenNameDigits: " << numScreenNameDigits << "\n";
    cerr << "bufferFillUsedSemPtr: " << bufferFillUsedSemPtr << "\n";
    cerr << "bufferFillFreeSemPtr: " << bufferFillFreeSemPtr << "\n";

	cerr << "this: " << this << "\n";
	cerr << "univGraphSet.zoomFactor: " << univGraphSet.zoomFactor << "\n";
	cerr << "screenshotCounter: " << screenshotCounter << "\n";
*/
    //	cerr << ": " <<  << "\n"; 	Output template

	cerr << "graphicsPtrVec.size(): " << graphicsPtrVec.size() << "\n";

}

void ImageWindow::dumpUniversalGraphicsSettings()
{
	qDebug() << "\n DUMPING UniversalGraphicsSettings: !!!!!!!!!!!!!!!!!!!!!!!!!!!\n";


}
void ImageWindow::init()
{
	mcStep = 0;
	modified = false;
	painter=0;

	screenshotCounter=0;
	scrollView=0;
	scrollView = new QScrollArea();
	scrollView->setBackgroundRole(QPalette::Dark);

	transactionCC3D = 0;
	graphFieldsPtr=new GraphicsDataFields();

	graphics2DPtr = new Graphics2D(this);
	graphics2DPtr->setId(QString("2D"));
	((Graphics2D*)graphics2DPtr)->resize(0,0);
	((Graphics2D*)graphics2DPtr)->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	((Graphics2D*)graphics2DPtr)->setBackgroundRole(QPalette::Shadow);
	((Graphics2D*)graphics2DPtr)->setAlignment(Qt::AlignLeft | Qt::AlignTop);

	projDataPtr= &((Graphics2D*)graphics2DPtr)->projData;

	graphics2DPtr->setGraphicsDataFieldPtr(graphFieldsPtr);
	graphics2DPtr->setUnivGraphSetPtr(&univGraphSet);

	//      scrollView->setWidget((Graphics2D*)graphics2DPtr);
	  setCentralWidget(scrollView);
	//      setCentralWidget((Graphics2D*)graphics2DPtr) ;
	 scrollView->setWidget((Graphics2D*)graphics2DPtr);
	 ((Graphics2D*)graphics2DPtr)->setVisible(true);

	setCentralWidget(scrollView);

	display3D=new Display3D(this);
	display3D->setId(QString("3D"));
	display3Dqvtk=(QVTKWidget*)display3D;

	((Display3D*)display3D)->setVisible(false);
	((Display3D*)display3D)->resize( QSize(500, 500).expandedTo(minimumSizeHint()) );
	//     clearWState( WState_Polished );

	display3D->setGraphicsDataFieldPtr(graphFieldsPtr);
	display3D->setUnivGraphSetPtr(&univGraphSet);

	//     ((Display3D*)display3D)->initializeVTKSettings();

	currentGraphicsPtr=graphics2DPtr;
	currentGraphicsPtr->setCurrentPainitgFcnPtr(currentGraphicsPtr->getPaintLattice());

	graphicsPtrVec.clear();	//Removes elements before push.
	graphicsPtrVec.push_back(graphics2DPtr);
	graphicsPtrVec.push_back(display3D); //graphicsPtrVec.size() = 2

	initializeGraphicsPtrVec();

	inventoryFileName="graphicsList.grl";

	drawingAllowed=false;

	if (mcStepLabel)
		mcStepLabel->setText("");
}

void ImageWindow::initStatic() //Oh, nasty static variables!!!
{
//	CompuCell3D::BoundaryStrategy::destroy();
//	if(CompuCell3D::NeighborFinder::getInstance()) delete CompuCell3D::NeighborFinder::getInstance(); // Add method destroy()
//	NeighborFinder

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::saveSettings(){
   writeSettings();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::customEvent(QEvent *event){ //Has nothing to do with Qt QCustomEvent class

	QEvent::Type type=((CustomEvent*)event)->type();
// cerr<<"got event of the type:"<<type<<endl;
//    if (type == (QEvent::Type)TransactionStartEvent::TransactionStart) {
//       cerr<<"got START EVENT:"<<type<<endl;
//    }

	if (type == (QEvent::Type)TransactionStartEvent::TransactionStart) 
	{
      GraphicsBase &graphics2D = *graphics2DPtr;
		gotErrorFlag=false;
      initializeGraphicsPtrVec();

      TransactionStartEvent *transactionStartEvent = (TransactionStartEvent*)(event);

      latticeType=transactionStartEvent->latticeType;
      for(int i = 0 ; i < graphicsPtrVec.size() ; ++i)
         graphicsPtrVec[i]->setLatticeType(transactionStartEvent->latticeType);

      xSpinBox->setMinimum(0);
      xSpinBox->setMaximum(graphFieldsPtr->getSizeL()-1);
      xSpinBox->setValue(graphFieldsPtr->getSizeL()/2);
      xSpinBox->setWrapping(true);


      ySpinBox->setMinimum(0);
      ySpinBox->setMaximum(graphFieldsPtr->getSizeM()-1);
      ySpinBox->setValue(graphFieldsPtr->getSizeM()/2);
      ySpinBox->setWrapping(true);



       zSpinBox->setMinimum(0);
       zSpinBox->setMaximum(graphFieldsPtr->getSizeN()-1);
       zSpinBox->setValue(graphFieldsPtr->getSizeN()/2);
       zSpinBox->setWrapping(true);



     ///using custom settings provided by users in xml config file
      if(transactionStartEvent->playerSettings.xyProjFlag){
         zSpinBox->setValue(transactionStartEvent->playerSettings.xyProj);
      }

      if(transactionStartEvent->playerSettings.xzProjFlag){
         ySpinBox->setValue(transactionStartEvent->playerSettings.xzProj);
      }

      if(transactionStartEvent->playerSettings.yzProjFlag){
         xSpinBox->setValue(transactionStartEvent->playerSettings.yzProj);
      }

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
       ((Display3D*)display3D)->setInitialConfigure3DData(data3D);




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



      if(transactionStartEvent->playerSettings.initialProjection=="xy"){
         initProjectionXYData();
      }

      if(transactionStartEvent->playerSettings.initialProjection=="xz"){
         initProjectionXZData();
      }

      if(transactionStartEvent->playerSettings.initialProjection=="yz"){
         initProjectionYZData();
      }


      ///initialize painter and pixmap

      ((Graphics2D*)graphics2DPtr)->getPixmap().fill(Qt::red);
       ///painter=new QPainter(imageLabel->pixmap(),this);

       ///projDataPtr->painterPtr=painter;

       /// Required activate  projection  pixmap and painter must be initialized
       if(transactionStartEvent->playerSettings.initialProjection=="xy"){
         xyButton->toggle();
       }else if(transactionStartEvent->playerSettings.initialProjection=="xz"){
         xzButton->toggle();
       }else if(transactionStartEvent->playerSettings.initialProjection=="yz"){
         yzButton->toggle();
       }else{
         xyButton->toggle();
       }


      ///initialize Combo box with field types
      GraphicsDataFields::floatField3DNameMapItr_t mitr;

		// Init plotTypeComboBox!
		plotTypeComboBox->clear();

      plotTypeComboBox->addItem(QString("Cell Field"));
      ///initialize Combo box with scalar field names
      for ( mitr = graphics2D.getGraphFieldsPtr()->getFloatField3DNameMap().begin();
            mitr != graphics2D.getGraphFieldsPtr()->getFloatField3DNameMap().end() ;
            ++mitr)
      {
         plotTypeComboBox->addItem(QString(mitr->first.c_str()));
      }

      //vectorCellFloatField3DNameMapItr_t
      GraphicsDataFields::vectorFieldCellLevelNameMapItr_t mitrV;
      ///initialize Combo box with  vector field names
      for ( mitrV = graphics2D.getGraphFieldsPtr()->getVectorFieldCellLevelNameMap().begin();
            mitrV != graphics2D.getGraphFieldsPtr()->getVectorFieldCellLevelNameMap().end() ;
            ++mitrV)
      {
         plotTypeComboBox->addItem(QString(mitrV->first.c_str()));
      }

      ///make tmp directory for storing pictures
	   tmpDirName="";
		//fullTmpDirName="";
      if(tmpDirName.isEmpty() ){

			tmpDirName=curFileStripped;
         QDate date=QDate::currentDate();
         QTime time=QTime::currentTime();
         QString dirSuffix;
   
         dirSuffix+=QString("_");
         //dirSuffix+=date.toString();
			dirSuffix+=QString().setNum(date.month());;
         dirSuffix+=QString("_");
			dirSuffix+=QString().setNum(date.day());
			dirSuffix+=QString("_");
			dirSuffix+=QString().setNum(date.year());;
			dirSuffix+=QString("___");
         dirSuffix+=time.toString();
         tmpDirName+=dirSuffix;


			//dirSuffix+=QString("_");
   //      dirSuffix+=date.toString();
   //      dirSuffix+=QString("_");
   //      dirSuffix+=time.toString();
   //      tmpDirName+=dirSuffix;
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
      if(univGraphSet.zoomFactor>1){
         zoomOutAct->setEnabled(true);
      }
      zoomInAct->setEnabled(true);
      ///enabling show menu


      if(univGraphSet.bordersOn){
         showBordersAct->setChecked(true);
      }
      showBordersAct->setEnabled(true);

      if(univGraphSet.contoursOn){
         showContoursAct->setChecked(true);
      }

      if(univGraphSet.concentrationLimitsOn){
         showConcentrationLimitsAct->setChecked(true);
      }
      showConcentrationLimitsAct->setEnabled(true);

      showContoursAct->setEnabled(true);



      drawingAllowed=true;
      ((Display3D*)display3D)->setDrawingAllowedFlag(true);


      ///init screenshot data
      if(/*silent &&*/ ! screenshotDescriptionFileName.isEmpty()){
         readScreenshotDescriptionList(screenshotDescriptionList,screenshotDescriptionFileName.toStdString());
         produceScreenshotDataList(screenshotDescriptionList);
      }

      transactionCC3D->setScreenUpdateFrequency(screenUpdateFrequency);
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
       ((Graphics2D*)graphics2DPtr)->getPixmap().fill();

      ///lock mutextransaction -prevents compucell thread "overrunning"

       //mutexTransaction.lock();
       bufferFillUsedSemPtr->acquire();
//        cerr<<"acquired bufferFillUsedSem in imagewindow"<<endl;
       mcStepLabel->setText("MC Step: "+QString::number(refreshEvent->mcStep));
       mcStep=refreshEvent->mcStep;
//       cerr<<"PLAYER: getScreenUpdateFrequency="<<transactionCC3D->getScreenUpdateFrequency()<<endl;
       drawField();

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
       
       cerr<<" \n\n\n\n TRANSACTION FINISH\n\n\n "<<endl;
       
      cerr<<finishEvent->message.toStdString()<<endl;
		cerr<<"closePlayerAfterSimulationDone="<<closePlayerAfterSimulationDone<<endl;
		cerr<<"finishEvent->exitFlag="<<finishEvent->exitFlag<<endl;

		if(finishEvent->message=="FINISHED SIMULATION"){
			stopSimulation();
		}

		if(closePlayerAfterSimulationDone){
			exit(0);
		}else if(finishEvent->exitFlag){
         exit(0);
		}else{
			cerr<<"You may start new simulation or exit the Player"<<endl;
		}
   }

	if (type == (QEvent::Type)TransactionStopSimulationEvent::TransactionStopSimulation) {

      TransactionStopSimulationEvent *stopSimulationEvent = (TransactionStopSimulationEvent *)event;
		TransactionFinishEvent *finishEvent = new TransactionFinishEvent();
		finishEvent->message=QString("END CURRENT SIMULATION!!!");
                finishEvent->exitFlag=false;
		QApplication::postEvent(computeThread.getTargetObject(),finishEvent); // Adds an event queue and returns immediately
		
   }


	if(type == (QEvent::Type)TransactionErrorEvent::TransactionError){
		TransactionErrorEvent *errorEvent = (TransactionErrorEvent *)event;
		
		if(errorHandler(errorEvent->errorCategory, errorEvent->message))
			cerr<<"Error Acknowledged"<<endl;
		else
			cerr<<"Error Disregarded"<<endl;
		gotErrorFlag=true;
		simulationStop();



	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::takeCurrentScreenshot(const std::string &_imageFileFullName,const std::string &_imageFileName){
      if(display3DButton->isChecked()){
            display3D->produceImage(_imageFileFullName);
      }else{
         QImage image;
         graphics2DPtr->produceImage(image);
         image.save(QString(_imageFileFullName.c_str()),"PNG");
      }

      inventoryFile<<_imageFileName<<endl;
      cerr<<"SAVING: "<<_imageFileName<<endl;
}

void ImageWindow::drawField(){
    if(!drawingAllowed)
      return;
    (this->*draw)();   
   mutexFieldDraw.lock();
      if(!plotTypeComboBox->count()){ ///no items in combo box - return
      mutexFieldDraw.unlock();
      return;

    }
    

   //Once we know the name of the field to plot we will check what type of plot it has associated with it and set corresponding pointers to plotting functions
   //and data
   QString selectedPlotName=plotTypeComboBox->currentText();
   std::string plotType=currentGraphicsPtr->getGraphFieldsPtr()->checkPlotType(selectedPlotName.toStdString());

//    cerr<<"THE TYPE OF "<<selectedPlotName.toStdString()<<" is "<<plotType<<endl;
  
//    if( selectedPlotType == QString("Cell Field") ){
   //for definition of allowed plot types see GraphicsDataFields.h

   if( plotType == "cell_field" ){
//       cerr<<"\n\n draw Cell Field\n\n"<<endl;
     currentGraphicsPtr->setCurrentPainitgFcnPtr(currentGraphicsPtr->getPaintLattice());
     currentGraphicsPtr->drawCurrentScene();

     
     
   }
//    else if(selectedPlotType == QString("CellVelocity")){
   else if(plotType == "vector_cell_level"){   
      cerr<<"INSIDE VECTOR_CELL_LEVEL_OPTION"<<endl;
     map<std::string, GraphicsDataFields::vectorFieldCellLevel_t * >::iterator mitr=
     currentGraphicsPtr->getGraphFieldsPtr()->getVectorFieldCellLevelNameMap().find(string(selectedPlotName.toStdString().c_str()));
     
     currentGraphicsPtr->setCurrentPainitgFcnPtr(currentGraphicsPtr->getPaintCellVectorFieldLattice());
     currentGraphicsPtr->setCurrentVectorCellLevelFieldPtr(mitr->second);
     currentGraphicsPtr->drawCurrentScene();
   }
   else if (plotType == "scalar"){
      GraphicsDataFields::floatField3DNameMapItr_t mitr=
         currentGraphicsPtr->getGraphFieldsPtr()->getFloatField3DNameMap().find(string(selectedPlotName.toStdString().c_str()));

//       cerr<<"LOOKING FOR A STRING:"<<string(selectedPlotType.ascii())<<endl;
      
      if(mitr != currentGraphicsPtr->getGraphFieldsPtr()->getFloatField3DNameMap().end() ){
      
//          cerr<<"FOUND:"<<string(selectedPlotType.ascii())<<endl;   
         currentGraphicsPtr->setCurrentConcentrationFieldPtr(mitr->second);

         currentGraphicsPtr->setCurrentPainitgFcnPtr(currentGraphicsPtr->getPaintConcentrationLattice());
         currentGraphicsPtr->drawCurrentScene();
                  
                  
         if(univGraphSet.concentrationLimitsOn){
            minMaxConcentrationLabel->setText(QString("Min: "+QString::number(currentGraphicsPtr->getMinConcentrationTrue())+" Max:"+
            QString::number(currentGraphicsPtr->getMaxConcentrationTrue())));

// // /*            minMaxConcentrationLabel->setText(tr(
// //                                                    "Min: "+QString::number(currentGraphicsPtr->getMinConcentrationTrue())+
// //                                                    " Max: "+QString::number(currentGraphicsPtr->getMaxConcentrationTrue())
// //             ));*/
         }
         
      }else{
      
         currentGraphicsPtr->setCurrentConcentrationFieldPtr(& currentGraphicsPtr->getGraphFieldsPtr()->field3DConcentration);
         
         currentGraphicsPtr->setCurrentPainitgFcnPtr(currentGraphicsPtr->getPaintConcentrationLattice());
         currentGraphicsPtr->drawCurrentScene();
         
         if(univGraphSet.concentrationLimitsOn){
//             minMaxConcentrationLabel->setText(tr(
//                                                    "Min"+QString::number(currentGraphicsPtr->getMinConcentrationTrue())+
//                                                    " Max"+QString::number(currentGraphicsPtr->getMaxConcentrationTrue())
//             ));
         }         

      }
   }


   mutexFieldDraw.unlock();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///this function draws Field in 2D

void ImageWindow::drawField2D(){

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///this function draws Field in 3D
void ImageWindow::drawField3D(){
   return;
   
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ImageWindow::switch3D(bool state){

   
   if(state){
      draw = &ImageWindow::drawField3D;
      currentGraphicsPtr=display3D;
//       cerr<<"\t\t\t TURNED OFF 2D"<<endl;
      ((Graphics2D*)graphics2DPtr)->hide();



      scrollView->setBackgroundRole(QPalette::Dark);
      scrollView->takeWidget();

      scrollView->setWidget((Display3D*)display3D);
      ((Display3D*)display3D)->setVisible(true);
      ((Display3D*)display3D)->show();

      
//       cerr<<"\t\t\t TURNED ON 3D"<<endl;

   }else{
      draw = &ImageWindow::drawField2D;
      currentGraphicsPtr=graphics2DPtr;
      
      ((Display3D*)display3D)->hide();


      scrollView->setBackgroundRole(QPalette::Dark);
      setCentralWidget(scrollView);
//       cerr<<"\t\t\t TURNED OFF 3D"<<endl;
      scrollView->takeWidget();
      scrollView->setWidget((Graphics2D*)graphics2DPtr);



      ((Graphics2D*)graphics2DPtr)->show();
//       cerr<<"\t\t\t TURNED ON 2D"<<endl;
      
   }
   
   drawField();
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ImageWindow::closeEvent(QCloseEvent *event)
{
	if (warningWithMutexRelease("Exit Program", "The simulation is stopped and the program is about to close.\nDo you want to exit the program?"))
	{
		stopThread = true;
		
//		if (useXMLFileFlag)
		computeThread.wait();

		//Finish Event should be here ONLY!
		TransactionFinishEvent *finishEvent = new TransactionFinishEvent();
		finishEvent->message=QString("DONE!!!");
		QApplication::postEvent(computeThread.getTargetObject(),finishEvent); // Adds an event queue and returns immediately

		//do not save settings if user used advanced XML settings and did not explicitely request saving of the settings
        if(advancedSettingsXMLOn && !saveSettingsXML){
			event->accept();
        }else{
			 writeSettings();
			 event->accept();
        }

    } else {
		  cerr<<"Current simulation is stopped stopped. Waiting for user action!!!"<<endl;
		  stopSimulation();
        event->ignore();
    }
}

void ImageWindow::newFile()
{
    if (maybeSave()) {
//         textEdit->clear();
        setCurrentFile("");
    }
}

void ImageWindow::open()
{
	SimulationFileOpenDialogForm dialog;
	dialog.loadCurrentValues(fileXML,useXMLFileFlag,pyConfData.pythonFileName,runPythonFlag);

	if(dialog.exec())
	{
		useXMLFileFlag=dialog.getUi().useXMLCheckBox->isChecked();
		runPythonFlag=dialog.getUi().pythonScriptCheckBox->isChecked();

		fileXML=dialog.getUi().xmlFileLineEdit->text();
		pyConfData.pythonFileName=dialog.getUi().pythonFileLineEdit->text();

/*
		if (computeThread.isRunning()) 
		{
			if(warning("Quit simulation", "Do you really want to quit\n the current simulation?"))
			{
				stopThread = true;
				computeThread.wait(); // Simulation thread joins the main thread!

				//delete scrollView;
				//delete graphFieldsPtr;
				//delete graphics2DPtr;
				//delete display3D;

				//readSettings();
				//graphicsPtrVec.clear(); 		//Remove elements from "graphicsPtrVec" vector
				//mcStep = 0;

				//computeThread.initTransactions();
				//cerr << "!!!!!!!!!!!!!!!!!!!!!!!!! computeThread.sizeTransactions(): " << computeThread.sizeTransactions() << "\n";
				transactionCC3D = 0;

				init(); 						// Brute force. Redo this function
				initStatic();
				dumpValues("ImageWindow::open()");

				stopThread = false;
				//cerr << "TREAD STOPPED THE EXECUTION !!!!!!!!!!!!!!!!!!!!!";
				
				//1. Release resources
				//2. Make reinitialization
				
				// Make the main window blank and properly reinitialize the simulation. 
			}
		}
*/
		// :)
		//smile = new Smile();//this);, "images/smile.jpg");
		//setCentralWidget(smile);

		if (useXMLFileFlag && fileXML!="")
		{
			loadFile(fileXML);
			curFile=fileXML;
			fileXMLStripped=strippedName(fileXML);
			curFileStripped=strippedName(fileXML);
			setCurrentFile(fileXML);
			setXMLFile(fileXML);
//                loadFile(fileName);
//                curFile=fileName;
//                curFileStripped=strippedName(fileName);
//                setCurrentFile(fileName);

         }else if (runPythonFlag && pyConfData.pythonFileName!=""){
               curFile=pyConfData.pythonFileName;
               curFileStripped=strippedName(pyConfData.pythonFileName);
               setCurrentFile(pyConfData.pythonFileName);
               pyConfData.pythonFileName=dialog.getUi().pythonFileLineEdit->text();;
         }else{
            int ret = QMessageBox::warning(this, tr("Simulation File Missing"),
                   tr("Please specify simulation file name and check Run Python or UseXML boxes."
                      ),
                   QMessageBox::Ok /*| QMessageBox::Discard
                   | QMessageBox::Cancel,
                   QMessageBox::Save*/);
            open();
         }
		// Should stop executing the thread if it is running! 

	}
}

bool ImageWindow::save()
{
/*
    if (curFile.isEmpty()) {
        return saveAs();
    } else {
        return saveFile(curFile);
    }
*/
	return false;
}

void ImageWindow::about()
{
	


 //QSound song("icons/Like_a_Bird.wav");
 //cerr<<"song name:"<<song.fileName().toStdString()<<endl;
 //song.play();

 //cerr<<"FINISHED SONG="<<song.isFinished()<<endl;
 //QMessageBox::about(this, tr("About Player"),
 //           tr("<b>CompuCell Player</b> is a visualization engine for CompuCell "
 //              "It will play all simultions including Paris Hilton new album "
	//				));

 QMessageBox msgBox;
 msgBox.setStandardButtons(QMessageBox::Ok);
 msgBox.setText (tr("<b>CompuCell3D version 3.4.0</b>\nCopyright Biocomplexity Institute, Indiana University, Bloomington,IN\n"
					"<b>CompuCell Player</b> is a visualization engine for CompuCell "
               
					));
 //msgBox.setIcon(QMessageBox::Information);
 switch (msgBox.exec()) {
 case QMessageBox::Ok:
     //song.stop();
	  msgBox.close();
     break;
 case QMessageBox::No:
     // no was clicked
     break;
 default:
     // should never be reached
     break;
 }


}

void ImageWindow::createActions()
{
	openAct = new QAction(QIcon(":/images/open.png"),tr("&Open Simulation"), this);
	openAct->setShortcut(tr("Ctrl+O"));
	openAct->setStatusTip(tr("Open an existing simulation"));
	connect(openAct, SIGNAL(triggered()), this, SLOT(open()));

	//saveAct = new QAction(QIcon(":/images/open.png"),tr("&Save Simulation"), this);
	//saveAct->setShortcut(tr("Ctrl+S"));
	//saveAct->setStatusTip(tr("Save the simulation"));
	//connect(saveAct, SIGNAL(triggered()), this, SLOT(save()));

	saveScrDesAct = new QAction(tr("Save Screenshot Description"), this);
	saveScrDesAct->setStatusTip(tr("Save Screenshot Description text file"));
	connect(saveScrDesAct, SIGNAL(triggered()), this, SLOT(saveScrDes()));
	saveScrDesAct->setEnabled(false);
   
	exitAct = new QAction(tr("E&xit"), this);
	exitAct->setShortcut(tr("Ctrl+Q"));
	exitAct->setStatusTip(tr("Exit the application"));
	connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

	///simulation
	#ifdef __APPLE__
	simulationAct = new QAction(QIcon(":/images/22x22/Tape_play.png"),tr("Start Si&mulation"), this);
	#else
	simulationAct = new QAction(QIcon(":/images/Tape_play.png"),tr("Start Si&mulation"), this);
	#endif
	simulationAct->setShortcut(tr("Ctrl+M"));
	simulationAct->setStatusTip(tr("Start simulation"));
	connect(simulationAct, SIGNAL(triggered()), this, SLOT(simulationStart()));
	
	#ifdef __APPLE__
	simulationPauseAct = new QAction(QIcon(":/images/22x22/Tape_pause.png"),tr("Pause Simulation"), this);
	#else
	simulationPauseAct = new QAction(QIcon(":/images/Tape_pause.png"),tr("Pause Simulation"), this);
	#endif
	
	simulationPauseAct->setShortcut(tr("Ctrl+Z"));
	simulationPauseAct->setStatusTip(tr("Pause simulation"));
	simulationPauseAct->setEnabled(false);
	connect(simulationPauseAct, SIGNAL(triggered()), this, SLOT(simulationPause()));
	
	#ifdef __APPLE__
	simulationStopAct = new QAction(QIcon(":/images/22x22/Tape_stop.png"),tr("Stop Simulation"), this);
	#else
	simulationStopAct = new QAction(QIcon(":/images/Tape_stop.png"),tr("Stop Simulation"), this);
	#endif
	
	simulationStopAct->setShortcut(tr("Ctrl+X"));
	simulationStopAct->setStatusTip(tr("Stop simulation"));
	simulationStopAct->setEnabled(false);
	connect(simulationStopAct, SIGNAL(triggered()), this, SLOT(simulationStop()));



	//simulationSerializeAct = new QAction(/*QIcon(":/images/Tape_pause.png"),*/tr("Serialize simulation"), this);
	//simulationSerializeAct->setShortcut(tr("Ctrl+L"));
	//simulationSerializeAct->setStatusTip(tr("Serialize simulation"));
	//simulationSerializeAct->setEnabled(false);
	//connect(simulationSerializeAct, SIGNAL(triggered()), this, SLOT(simulationSerialize()));
		
	///zooming
	#ifdef __APPLE__
	zoomInAct = new QAction(QIcon(":/images/22x22/zoomin.png"),tr("ZoomIn"),this);
	#else
	zoomInAct = new QAction(QIcon(":/images/zoomin.png"),tr("ZoomIn"),this);
	#endif

	zoomInAct->setShortcut(tr("Ctrl++"));
	zoomInAct->setStatusTip(tr("ZoomIn view"));  
	connect(zoomInAct, SIGNAL(triggered()), this, SLOT(zoomIn()));
	zoomInAct->setEnabled(false); 
	#ifdef __APPLE__
	zoomOutAct = new QAction(QIcon(":/images/22x22/zoomout.png"),tr("ZoomOut"),this);
	#else
	zoomOutAct = new QAction(QIcon(":/images/zoomout.png"),tr("ZoomOut"),this);
   #endif
	zoomOutAct->setShortcut(tr("Ctrl+-"));
	zoomOutAct->setStatusTip(tr("ZoomOut view"));
	connect(zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOut()));
	zoomOutAct->setEnabled(false);
		
	///Show actions
	showBordersAct = new QAction(tr("Cell borders"), this);
	showBordersAct->setStatusTip(tr("Show cell borders"));
	showBordersAct->setCheckable(true);
	//Note, I have to set it before connecting to the slot. Otherwise I would get into trouble once the slot gets called from this place int the program
	showBordersAct->setChecked(univGraphSet.bordersOn);//Set initial state based on settings.
	connect(showBordersAct, SIGNAL(toggled(bool)), this, SLOT(bordersDisplay(bool)));



	showContoursAct = new QAction(tr("Concentration contours"), this);
	showContoursAct->setStatusTip(tr("Show concentration isocontour lines"));
	showContoursAct->setCheckable(true);
	//Note, I have to set it before connecting to the slot. Otherwise I would get into trouble once the slot gets called from this place int the program
	showContoursAct->setChecked(univGraphSet.contoursOn);//Set initial state based on settings.
	connect(showContoursAct, SIGNAL(toggled(bool)), this, SLOT(contoursDisplay(bool)));
	

	showConcentrationLimitsAct = new QAction(tr("Concentration limits"),this);
	showConcentrationLimitsAct->setStatusTip(tr("Show min and max concentration"));
	showConcentrationLimitsAct->setCheckable(true);
	connect(showConcentrationLimitsAct, SIGNAL(toggled(bool)), this, SLOT(concentrationLimitsDisplay(bool)));
	showConcentrationLimitsAct->setChecked(univGraphSet.concentrationLimitsOn);

	///Configure colors action
	typeColorAct = new QAction(tr("Cell type colors"), this);
	typeColorAct->setStatusTip(tr("Display and configure cell type colors"));
	connect(typeColorAct, SIGNAL(triggered()), this, SLOT(configureCellTypeColors()));

	colorMapPlotAct = new QAction(tr("Colormap plot"), this);
	colorMapPlotAct->setStatusTip(tr("Configure color map plot"));
	connect(colorMapPlotAct, SIGNAL(triggered()), this, SLOT(configureColorMapPlot()));

	vectorFieldPlotAct = new QAction(tr("Vector Field plot"),this);
	vectorFieldPlotAct->setStatusTip(tr("Configure vector field plot"));
	connect(vectorFieldPlotAct, SIGNAL(triggered()), this, SLOT(configureVectorFieldPlot()));
		            

	screenshotFrequencyAct = new QAction(tr("Screenshot frequency"),this);
	screenshotFrequencyAct->setStatusTip(tr("Configure screenshot frequency"));
	connect(screenshotFrequencyAct, SIGNAL(triggered()), this, SLOT(configureScreenshotFrequency()));
		
	types3DInvisibleAct = new QAction(tr("Cell Types invisible in 3D"),this);
	types3DInvisibleAct->setStatusTip(tr("You may list cell types that will be invisible in 3D"));
	connect(types3DInvisibleAct, SIGNAL(triggered()), this, SLOT(configureTypes3DInvisible()));

	configure3DDisplayAct = new QAction(tr("3D Display configuration"), this);
	configure3DDisplayAct->setStatusTip(tr("3D Display configuration"));
	connect(configure3DDisplayAct, SIGNAL(triggered()), this, SLOT(configure3DDisplay()));

	closePlayerAfterSimulationDoneAct=new QAction(tr("Close Player After Simulation Done"), this);
	closePlayerAfterSimulationDoneAct->setStatusTip(tr("IF you want Player to close after simulations is done. Check this menu item"));
	closePlayerAfterSimulationDoneAct->setCheckable(true);
	connect(closePlayerAfterSimulationDoneAct, SIGNAL(toggled(bool)), this, SLOT(setClosePlayerAfterSimulationDone(bool)));
	closePlayerAfterSimulationDoneAct->setChecked(closePlayerAfterSimulationDone);
	
	setClosePlayerAfterSimulationDone(closePlayerAfterSimulationDone); //Set initial state based on settings
	cerr<<"closePlayerAfterSimulationDone="<<closePlayerAfterSimulationDone<<endl;
	//exit(0);


	//save settings Action
	saveSettingsAct = new QAction(tr("Save Default Settings"),this);
	saveSettingsAct->setStatusTip(tr("Saves current settings as default settings"));
	connect(saveSettingsAct, SIGNAL(triggered()), this, SLOT(saveSettings()));
		            
	aboutAct = new QAction(tr("&About"), this);
	aboutAct->setStatusTip(tr("Show the application's About box"));
	connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));

	aboutQtAct = new QAction(tr("About &Qt"), this);
	aboutQtAct->setStatusTip(tr("Show the Qt library's About box"));
	connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
}

void ImageWindow::createMenus()
{
	fileMenu = menuBar()->addMenu(tr("&File"));
	fileMenu->addAction(openAct);
	//fileMenu->addAction(saveAct);
	fileMenu->addAction(saveScrDesAct);
	fileMenu->addSeparator();
	fileMenu->addAction(exitAct);

	simulationMenu = menuBar()->addMenu(tr("Si&mulation"));
	simulationMenu -> addAction(simulationAct);
	simulationMenu -> addAction(simulationPauseAct);
        simulationMenu -> addAction(simulationStopAct);
	//simulationMenu -> addAction(simulationSerializeAct);
	

	zoomMenu = menuBar()->addMenu(tr("Zoom"));
	zoomMenu->addAction(zoomInAct);
	zoomMenu->addAction(zoomOutAct);

	showMenu = menuBar()->addMenu(tr("Show"));
	showMenu->addAction(showBordersAct);
	showMenu->addAction(showContoursAct);

	showMenu->addAction(showConcentrationLimitsAct);
	showBordersAct->setEnabled(false);
	showContoursAct->setEnabled(false);

	configureMenu = menuBar()->addMenu(tr("Configure"));
	configureMenu->addAction(typeColorAct);
	configureMenu->addAction(colorMapPlotAct);
	configureMenu->addAction(vectorFieldPlotAct);
	configureMenu->addAction(screenshotFrequencyAct);
	configureMenu->addAction(types3DInvisibleAct);
	configureMenu->addAction(configure3DDisplayAct);
	configureMenu->addAction(closePlayerAfterSimulationDoneAct);
	configureMenu->addSeparator();
	configureMenu->addAction(saveSettingsAct);

	helpMenu = menuBar()->addMenu(tr("&Help"));
	helpMenu->addAction(aboutAct);
	helpMenu->addAction(aboutQtAct);
}

void ImageWindow::createSteeringBar()
{
	steeringBar = addToolBar(tr("Steering"));
#ifdef __APPLE__
    QMacStyle::setWidgetSizePolicy(steeringBar,QMacStyle::SizeMini);
#endif
	steeringBar->addAction(simulationAct);
	steeringBar->addAction(simulationPauseAct);
	steeringBar->addAction(simulationStopAct);
	simulationPauseAct->setEnabled(false);
	simulationStopAct->setEnabled(false);
	steeringBar->addAction(zoomInAct);
	steeringBar->addAction(zoomOutAct);
	steeringBar->addSeparator();

	display3DButton = new QRadioButton(tr("3D"));

	//    QToolTip::add(display3DButton,tr("Switch to 3D mode. To speed up rendering \n set some cell types to be invisible in the 3D mode\nClick: Configure->Cell types invisible in 3D..."));

	///xy cross section
	///xyButton=new QRadioButton(tr("xy"));
	//    QToolTip::add(xyButton,tr("View lattice along xy plane"));
	zSpinBox=new QSpinBox();
	//    QToolTip::add(zSpinBox,tr("Change \'z\' coordinate of xy plane"));

	///xz cross section
	///xzButton=new QRadioButton(tr("xz"));
	//    QToolTip::add(xzButton,tr("View lattice along xz plane"));
	ySpinBox=new QSpinBox();
	//    QToolTip::add(ySpinBox,tr("Change \'y\' coordinate of xz plane"));

	///yzButton=new QRadioButton(tr("yz"));
	//    QToolTip::add(yzButton,tr("View lattice along yz plane"));
	xSpinBox=new QSpinBox();
	//    QToolTip::add(xSpinBox,tr("Change \'x\' coordinate of yz plane"));

	steeringBar->addSeparator();

	crossSectionButtonsG = new QButtonGroup(steeringBar);
    QGroupBox *groupBox = new QGroupBox(tr("Cross Section"));
#ifdef __APPLE__
    //QMacStyle::setWidgetSizePolicy(groupBox,QMacStyle::SizeMini);
#endif
	xyButton = new QRadioButton(tr("xy"));
	xzButton = new QRadioButton(tr("xz"));
	yzButton = new QRadioButton(tr("yz"));

	//     xyButton->setChecked(true);

	QHBoxLayout *hbox = new QHBoxLayout;
	hbox->addWidget(display3DButton);
	hbox->addWidget(xyButton);
	hbox->addWidget(zSpinBox);
	hbox->addWidget(xzButton);
	hbox->addWidget(ySpinBox);
	hbox->addWidget(yzButton);
	hbox->addWidget(xSpinBox);
	//hbox->addStretch(0);
	//     hbox->addStretch(1);
	//     hbox->addSpacing(1);
    
	groupBox->setLayout(hbox);

    steeringBar ->addWidget(groupBox);

	steeringBar->addSeparator();
	plotTypeComboBoxName=new QLabel(tr("Plot Type"));
	plotTypeComboBox=new QComboBox();
	plotTypeComboBox->setMinimumContentsLength(5);
	plotTypeComboBox->setSizeAdjustPolicy(QComboBox::AdjustToContents);
	steeringBar -> addWidget(plotTypeComboBoxName);
	steeringBar -> addWidget(plotTypeComboBox);
	//    QToolTip::add(plotTypeComboBox,tr("Select plot type"));

	steeringBar->addSeparator();
	recordButton= new QPushButton(QIcon(":/images/Camera.png"),tr(""));;
	steeringBar->addWidget(recordButton);

	//    QToolTip::add(recordButton,tr("Add screenshot of current view to be recorded from now on"));

	connect(xyButton, SIGNAL(toggled(bool)), this, SLOT(changeProjectionXY(bool)) );
	connect(zSpinBox, SIGNAL(valueChanged(int)), this, SLOT(alterZCoordinate(int)) );
	connect(xzButton, SIGNAL(toggled(bool)), this, SLOT(changeProjectionXZ(bool)) );
	connect(ySpinBox, SIGNAL(valueChanged(int)), this, SLOT(alterYCoordinate(int)) );
	connect(yzButton, SIGNAL(toggled(bool)), this, SLOT(changeProjectionYZ(bool)) );
	connect(xSpinBox, SIGNAL(valueChanged(int)), this, SLOT(alterXCoordinate(int)) );
	connect(display3DButton, SIGNAL(toggled(bool)), this, SLOT(switch3D(bool)) );
	connect(plotTypeComboBox, SIGNAL(activated ( const QString &)), this, SLOT(plotTypeUpdate(const QString &)) );
	connect(recordButton, SIGNAL(clicked()), this, SLOT(addScreenshot()) );
}

void ImageWindow::createStatusBar()
{
    mcStepLabel = new QLabel(this);
    QPalette palette_back;
    palette_back.setColor(mcStepLabel->backgroundRole(), QColor("white"));
    mcStepLabel->setPalette(palette_back);
    QPalette palette_for;
    palette_for.setColor(mcStepLabel->foregroundRole(), QColor("red"));
    mcStepLabel->setPalette(palette_for);;
    mcStepLabel->setAlignment(Qt::AlignHCenter);
    mcStepLabel->setMinimumSize(mcStepLabel->sizeHint());
    
    minMaxConcentrationLabel=new QLabel(this);
    minMaxConcentrationLabel->setAlignment(Qt::AlignHCenter);
    minMaxConcentrationLabel->setMinimumSize(minMaxConcentrationLabel->sizeHint());

    statusBar()->addWidget(mcStepLabel);
    statusBar()->addWidget(minMaxConcentrationLabel);
}

bool ImageWindow::warning(QString header, QString text)
{
	// mutex.tryLock() actually takes the lock if the mutex is not locked!

	mutex.tryLock();
	mutexStartPause.tryLock();

	int ret = QMessageBox::warning(this, header, text, QMessageBox::Yes, QMessageBox::No  | QMessageBox::Default);

	// At this point mutex and mutexStartPause should be locked!

	if (ret == QMessageBox::Yes)
	{
		mutex.unlock();
		mutexStartPause.unlock();

		return true;
	}

	return false;
}


bool ImageWindow::warningWithMutexRelease(QString header, QString text)
{
	// mutex.tryLock() actually takes the lock if the mutex is not locked!

	mutex.tryLock();
	mutexStartPause.tryLock();

	int ret = QMessageBox::warning(this, header, text, QMessageBox::Yes, QMessageBox::No  | QMessageBox::Default);

	// At this point mutex and mutexStartPause should be locked!

	if (ret == QMessageBox::Yes)
	{
		mutex.unlock();
		mutexStartPause.unlock();

		return true;
	}

	mutex.unlock();
	mutexStartPause.unlock();

	return false;
}

bool ImageWindow::errorHandler(QString header, QString text)
{
	// mutex.tryLock() actually takes the lock if the mutex is not locked!

	mutex.tryLock();
	mutexStartPause.tryLock();

	int ret = QMessageBox::warning (this, header, text, QMessageBox::Ok  | QMessageBox::Default);

	// At this point mutex and mutexStartPause should be locked!

	if (ret == QMessageBox::Yes)
	{
		mutex.unlock();
		mutexStartPause.unlock();

		return true;
	}

	return false;
}


bool ImageWindow::maybeSave()
{
    return true;
}

void ImageWindow::loadFile(const QString &fileName)
{
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        QMessageBox::warning(this, tr("Application"),
                             tr("Cannot read file %1:\n%2.")
                             .arg(fileName)
                             .arg(file.errorString()));
        return;
    }

    QTextStream in(&file);
    QApplication::setOverrideCursor(Qt::WaitCursor);
//     textEdit->setPlainText(in.readAll());
    QApplication::restoreOverrideCursor();

    setCurrentFile(fileName);
    statusBar()->showMessage(tr("File loaded"), 2000);
}

bool ImageWindow::saveFile(const QString &fileName)
{
    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        QMessageBox::warning(this, tr("Application"),
                             tr("Cannot write file %1:\n%2.")
                             .arg(fileName)
                             .arg(file.errorString()));
        return false;
    }

    QTextStream out(&file);
    QApplication::setOverrideCursor(Qt::WaitCursor);
//     out << textEdit->toPlainText();
    QApplication::restoreOverrideCursor();

    setCurrentFile(fileName);
    statusBar()->showMessage(tr("File saved"), 2000);
    return true;
}

void ImageWindow::setCurrentFile(const QString &fileName)
{
    curFile = fileName;
    
//     textEdit->document()->setModified(false);
    setWindowModified(false);

    QString shownName;
    if (curFile.isEmpty()){
        shownName = "Unknown Simulation";
        curFileStripped=shownName;
    }
    else{
        shownName = strippedName(curFile);
        curFileStripped=strippedName(curFile);
    }
    
    setWindowTitle(tr("%1[*] - %2").arg(shownName).arg(tr("Application")));
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool ImageWindow::saveScrDes(){

    
    QString fileName = QFileDialog::getSaveFileName(0,"",".");
    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        QMessageBox::warning(this, tr("Application"),
                             tr("Cannot write file %1:\n%2.")
                             .arg(fileName)
                             .arg(file.errorString()));
        return false;
    }

  if (!fileName.isEmpty())
        saveScrDesFile(fileName);


    QApplication::setOverrideCursor(Qt::WaitCursor);

    QApplication::restoreOverrideCursor();

    statusBar()->showMessage(tr("File saved"), 2000);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::saveScrDesFile(const QString &fileName){

   ofstream out(fileName.toStdString().c_str());
   for (list<ScreenshotDescription>::iterator litr = screenshotDescriptionList.begin() ; litr != screenshotDescriptionList.end() ; ++litr){
      out<<*litr<<endl;
   }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::simulationStart()
{
	simulation();
	simulationPauseAct->setEnabled(true);
	simulationStopAct->setEnabled(true);
	openAct->setEnabled(false);
	simulationAct->setEnabled(false);
	stopThread = false;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::simulationPause(){
   ///need to have second mutex (StartPause) because if  mutex is unlocked, then locked by cc3d thread then if between these two instructions
   ///one hits "play" button then another thread would be created if instead of if(!mutexStartPause.locked()){ there was
   ///if(!mutex.locked()){ instruction

	simulationPauseAct->setEnabled(false);
	mutex.lock();
	mutexStartPause.lock();

	simulationAct->setEnabled(true);
	openAct->setEnabled(true);//Open simulation enabled
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::simulationStop(){
   ///need to have second mutex (StartPause) because if  mutex is unlocked, then locked by cc3d thread then if between these two instructions
   ///one hits "play" button then another thread would be created if instead of if(!mutexStartPause.locked()){ there was
   ///if(!mutex.locked()){ instruction
      if(!simulationPauseAct->isEnabled()){
	

            if (warning("Simulation Stop", "Do you want to stop current simulation?"))
            {
               stopSimulation();
            }else{
                  cerr<<"Keep going"<<endl;
            }
       }else{
				tryStopSimulation();
            //stopSimulation();
       }         
            

}


void ImageWindow::tryStopSimulation(){
		stopThread = true;
}
void ImageWindow::stopSimulation(){


	   stopThread = true;
		
//		if (useXMLFileFlag)
		computeThread.wait();
		cerr<<"AFTER WAIT"<<endl;
               simulationStopAct->setEnabled(false);
               simulationPauseAct->setEnabled(false);
         
               simulationAct->setEnabled(true);
               openAct->setEnabled(true);//open simulation enabled


		//Finish Event should be here ONLY!
		TransactionFinishEvent *finishEvent = new TransactionFinishEvent();
		finishEvent->message=QString("END CURRENT SIMULATION!!!");
                finishEvent->exitFlag=false;
		QApplication::postEvent(computeThread.getTargetObject(),finishEvent); // Adds an event queue and returns immediately

		//do not save settings if user used advanced XML settings and did not explicitely request saving of the settings

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ImageWindow::startSimulation(){
   simulationStart();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::addTransaction(Transaction *transact){
    computeThread.addTransaction(transact);
    openAct->setEnabled(false);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::simulationSerialize(){

   simulationSerializeAct->setEnabled(false);

   if(simulationPauseAct->isEnabled()){
      simulationPause();
   }
   
   transactionCC3D->getSimulator()->serialize();

   simulationSerializeAct->setEnabled(true);

   simulation();

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::zoomIn(){
   
   univGraphSet.zoomFactor++;
   if((univGraphSet.zoomFactor-1)==1){
      zoomOutAct->setEnabled(true);
   }

   QSize size = ((Graphics2D*)graphics2DPtr)->size();
   size/=(univGraphSet.zoomFactor-1);
   size*=univGraphSet.zoomFactor;   
   drawField();
   cerr<<"Zoom factor="<<univGraphSet.zoomFactor<<endl;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::zoomOut(){
   
   univGraphSet.zoomFactor--;
   if(univGraphSet.zoomFactor==1){
      zoomOutAct->setEnabled(false);
   }
   QSize size = ((Graphics2D*)graphics2DPtr)->size();
   size/=(univGraphSet.zoomFactor+1);
   size*=univGraphSet.zoomFactor;
   drawField();
   cerr<<"Zoom factor="<<univGraphSet.zoomFactor<<endl;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::initProjectionXYData(){

   projDataPtr->sizeL = graphFieldsPtr->getSizeL(); //change it!
   projDataPtr->sizeM = graphFieldsPtr->getSizeM(); //change it!

   
   projDataPtr->xMin = 0;
   projDataPtr->xMax = graphFieldsPtr->getSizeL();
   projDataPtr->yMin = 0;
   projDataPtr->yMax = graphFieldsPtr->getSizeM();
   projDataPtr->zMin = zSpinBox->value();
   projDataPtr->zMax = zSpinBox->value()+1;
   projDataPtr->projection="xy";

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::changeProjectionXY(bool _state){
//    cerr<<"\n\n\nsetting draw Ptr\n\n\n";
   draw=&ImageWindow::drawField2D;

   if(_state){

      initProjectionXYData()   ;
      drawField();
   }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::alterZCoordinate(int _value){

     xyProjData=*projDataPtr;

     xyProjData.resetBoundaries(graphFieldsPtr->getSizeL(),graphFieldsPtr->getSizeM(),graphFieldsPtr->getSizeN());

     xyProjData.zMin = _value;
     xyProjData.zMax = _value+1;
     xyProjData.projection="xy";

     if(xyButton->isChecked()){
         *projDataPtr=xyProjData;
          drawField();
     }


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::initProjectionXZData(){
   projDataPtr->sizeL = graphFieldsPtr->getSizeL(); //change it!
   projDataPtr->sizeM = graphFieldsPtr->getSizeN(); //change it!

   
   projDataPtr->xMin = 0;
   projDataPtr->xMax = graphFieldsPtr->getSizeL();
   projDataPtr->yMin = ySpinBox->value();
   projDataPtr->yMax = ySpinBox->value()+1;
   projDataPtr->zMin = 0;
   projDataPtr->zMax = graphFieldsPtr->getSizeN();
   projDataPtr->projection="xz";

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::changeProjectionXZ(bool _state){
   draw=&ImageWindow::drawField2D;


   if(_state){

      initProjectionXZData();
      
         drawField();
   }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::alterYCoordinate(int _value){


     xzProjData=*projDataPtr;

     xzProjData.resetBoundaries(graphFieldsPtr->getSizeL(),graphFieldsPtr->getSizeM(),graphFieldsPtr->getSizeN());

     xzProjData.yMin = _value;
     xzProjData.yMax = _value+1;
     xzProjData.projection="xz";

     if(xzButton->isChecked()){
         *projDataPtr=xzProjData;
         
            drawField();
     }



}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::initProjectionYZData(){

   projDataPtr->sizeL = graphFieldsPtr->getSizeM(); //change it!
   projDataPtr->sizeM = graphFieldsPtr->getSizeN(); //change it!

   
   projDataPtr->xMin = xSpinBox->value();
   projDataPtr->xMax = xSpinBox->value()+1;
   projDataPtr->yMin = 0;
   projDataPtr->yMax = graphFieldsPtr->getSizeM();
   projDataPtr->zMin = 0;
   projDataPtr->zMax = graphFieldsPtr->getSizeN();
   projDataPtr->projection="yz";


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::changeProjectionYZ(bool _state){

   draw=&ImageWindow::drawField2D;

   if(_state){

      initProjectionYZData();
      
         drawField();
   }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::alterXCoordinate(int _value){


     yzProjData=*projDataPtr;

     yzProjData.resetBoundaries(graphFieldsPtr->getSizeL(),graphFieldsPtr->getSizeM(),graphFieldsPtr->getSizeN());

     yzProjData.xMin = _value;
     yzProjData.xMax = _value+1;
     yzProjData.projection="yz";

     if(yzButton->isChecked()){
         *projDataPtr=yzProjData;
         
            drawField();
     }



}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::bordersDisplay(bool _on){
   
   if(showBordersAct->isChecked()){

      univGraphSet.bordersOn=true;
      
         drawField();
   }else{

      univGraphSet.bordersOn=false;
      
         drawField();
   }

   
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::runPython(bool){

   if(runPythonAct->isChecked()){
      runPythonFlag=true;
   }else{
      runPythonFlag=false;
   }

   
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::plotTypeUpdate( const QString &){
   
      drawField();
   
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void ImageWindow::contoursDisplay(bool _on){

   if(showContoursAct->isChecked()){

      univGraphSet.contoursOn=true;
      
         drawField();
   }else{

      univGraphSet.contoursOn=false;
      
         drawField();
   }


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::concentrationLimitsDisplay(bool _on){

   if(showConcentrationLimitsAct->isChecked()){
      
      univGraphSet.concentrationLimitsOn=true;
      
      
      
   }else{

      univGraphSet.concentrationLimitsOn=false;
      minMaxConcentrationLabel->setText("");
      minMaxConcentrationLabel->setMinimumSize(QSize());
      
   }
      
   
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::configureScreenshotFrequency(){

   ScreenshotFrequencyConfigureForm dialog;
//    ScreenshotFrequencyDialog dialog;
   dialog.loadCurrentValues(screenshotFrequency,screenUpdateFrequency,noOutputFlag);
   //shows dialog
   if (dialog.exec()) {
      //dialog vanished from the screen, read values from dialog
      screenshotFrequency=dialog.getUi().freqSpinBox->value();
      screenUpdateFrequency=dialog.getUi().screenUpdateSpinBox->value();
      noOutputFlag=dialog.getUi().noOutputCheckBox->isChecked();
      if(transactionCC3D)// before starting the simmulation transactionCC3D is null
         transactionCC3D->setScreenUpdateFrequency(screenUpdateFrequency);
   }      
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::configureColorMapPlot(){
   ColormapPlotConfigureForm dialog;


      dialog.loadCurrentValues(
                                 minConcentration , minConcentrationFixed,
                                 maxConcentration , maxConcentrationFixed,
                                 numberOfLegendBoxes,numberAccuracy,legendEnable
                                 
                              );   
    ///shows dialog
    if (dialog.exec()) {
      ///dialog vanished from the screen, read values from dialog
      minConcentration = dialog.getUi().minLineEdit->text().toFloat();
      minConcentrationFixed = dialog.getUi().minCheckBox->isChecked();
      maxConcentration = dialog.getUi().maxLineEdit->text().toFloat();
      maxConcentrationFixed = dialog.getUi().maxCheckBox->isChecked();
      numberOfLegendBoxes = dialog.getUi().boxSpinBox->value();
      numberAccuracy = dialog.getUi().accuracySpinBox->value();
      legendEnable = dialog.getUi().showLegendBox->isChecked();
      
      for(unsigned int i  = 0 ; i < graphicsPtrVec.size() ; ++i){
         graphicsPtrVec[i]->setMinConcentrationFixed(minConcentrationFixed);
         graphicsPtrVec[i]->setMaxConcentrationFixed(maxConcentrationFixed);
         graphicsPtrVec[i]->setMinConcentration(minConcentration);
         graphicsPtrVec[i]->setMaxConcentration(maxConcentration);
         graphicsPtrVec[i]->setNumberOfLegendBoxes(numberOfLegendBoxes);
         graphicsPtrVec[i]->setNumberAccuracy(numberAccuracy);
         graphicsPtrVec[i]->setLegendEnable(legendEnable);
      }

            
         drawField();
    }
    
    
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::configureVectorFieldPlot(){
   VectorFieldPlotConfigureForm dialog;


      QPalette palette;
      palette.setColor(dialog.getUi().arrowColorButton->backgroundRole(), univGraphSet.arrowPen.color());
//       palette.setColor(dialog.getUi().arrowColorButton->backgroundRole(), QColor("blue"));
      dialog.getUi().arrowColorButton->setPalette(palette);


      dialog.loadCurrentValues(
                                 minMagnitude , minMagnitudeFixed,
                                 maxMagnitude , maxMagnitudeFixed,
                                 arrowLength,
                                 numberOfLegendBoxesVector,numberAccuracyVector,legendEnableVector,overlayVectorCellFields,scaleArrows,fixedArrowColorFlag
                              );   
    ///shows dialog
    if (dialog.exec()) {
      ///dialog vanished from the screen, read values from dialog
      minMagnitude = dialog.getUi().minLineEdit->text().toFloat();
      minMagnitudeFixed = dialog.getUi().minCheckBox->isChecked();
      maxMagnitude = dialog.getUi().maxLineEdit->text().toFloat();
      maxMagnitudeFixed = dialog.getUi().maxCheckBox->isChecked();
      arrowLength = dialog.getUi().arrowLengthSpinBox->value();
      numberOfLegendBoxesVector = dialog.getUi().boxSpinBox->value();
      numberAccuracyVector = dialog.getUi().accuracySpinBox->value();
      legendEnableVector = dialog.getUi().showLegendBox->isChecked();
      overlayVectorCellFields=dialog.getUi().overlayVectorCellCheckBox->isChecked();
      scaleArrows=dialog.getUi().scaleArrowsCheckBox->isChecked();
      fixedArrowColorFlag=dialog.getUi().fixedArrowColorCheckBox->isChecked();

      QColor arrowColor = dialog.getUi().arrowColorButton->palette().color(QPalette::Button);
//       cerr<<"QColor="<<arrowColor.red()<<" "<<arrowColor.green()<<" "<<arrowColor.blue()<<endl;
      univGraphSet.arrowPen.setColor(arrowColor);



      for(unsigned int i  = 0 ; i < graphicsPtrVec.size() ; ++i){
         graphicsPtrVec[i]->setMinMagnitudeFixed(minMagnitudeFixed);
         graphicsPtrVec[i]->setMaxMagnitudeFixed(maxMagnitudeFixed);
         graphicsPtrVec[i]->setMinMagnitude(minMagnitude);
         graphicsPtrVec[i]->setMaxMagnitude(maxMagnitude);
         graphicsPtrVec[i]->setMaxMagnitude(maxMagnitude);
         graphicsPtrVec[i]->setArrowLength(arrowLength);
         graphicsPtrVec[i]->setNumberOfLegendBoxesVector(numberOfLegendBoxesVector);
         graphicsPtrVec[i]->setNumberAccuracyVector(numberAccuracyVector);
         graphicsPtrVec[i]->setLegendEnableVector(legendEnableVector);         
         graphicsPtrVec[i]->setOverlayVectorCellFields(overlayVectorCellFields);
         graphicsPtrVec[i]->setScaleArrows(scaleArrows);
         graphicsPtrVec[i]->setFixedArrowColor(fixedArrowColorFlag);
      }
      
         drawField();
    }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::configureTypes3DInvisible(){
   
   TypesThreeDConfigureForm dialog;

   dialog.loadCurrentValues(univGraphSet.types3DInvisibleVec);
      
   if (dialog.exec()) {
   
      dialog.fillTypes3DInvisibleVec(univGraphSet.types3DInvisibleVec);
         drawField();      
   }

 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::configure3DDisplay(){
   
   Configure3DDialogForm dialog;
   Display3D * disp=(Display3D*)display3D;
   Configure3DData data = disp->getConfigure3DData();
   dialog.loadCurrentValues(data);


   
   if (dialog.exec()) {

      data.sizeX=dialog.getUi().xSpinBox->value();
      data.sizeY=dialog.getUi().ySpinBox->value();
      data.sizeZ=dialog.getUi().zSpinBox->value();
      data.rotationX=dialog.getUi().xRotSpinBox->value();
      data.rotationY=dialog.getUi().yRotSpinBox->value();
      data.rotationZ=dialog.getUi().zRotSpinBox->value();

      disp->setConfigure3DData(data);
      
      
      
         drawField();      
      
      
   }

 
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::configureCellTypeColors()
{
	CellTypeColorConfigureForm dialog;

	short cellType;
	map<unsigned short,QPen>::iterator pmItr;
	map<unsigned short,QBrush>::iterator bmItr;
	UniversalGraphicsSettings::colorMapItr cItr;

	//loading current color assignments
   
	QPalette palette;
	palette.setColor(dialog.getUi().borderColorLabel->backgroundRole(), univGraphSet.borderPen.color());
	dialog.getUi().borderColorLabel->setPalette(palette);
	palette.setColor(dialog.getUi().contourColorLabel->backgroundRole(), univGraphSet.contourPen.color());
	dialog.getUi().contourColorLabel->setPalette(palette);

	dialog.loadCurrentValues(univGraphSet.typeColorMap);

	if (dialog.exec()) 
	{
		QColor borderColor = dialog.getUi().borderColorLabel->palette().color(QPalette::Background);
		QColor contourColor = dialog.getUi().contourColorLabel->palette().color(QPalette::Background);

		univGraphSet.borderPen.setColor(borderColor);
		univGraphSet.contourPen.setColor(contourColor);

		univGraphSet.borderColor=borderColor;
		univGraphSet.contourColor=contourColor;

		univGraphSet.typePenMap.clear();
		univGraphSet.typeBrushMap.clear();
		univGraphSet.typeColorMap.clear();

		for (int row=0; row < dialog.getUi().typeColorTable->rowCount(); ++row)
		{
			QColor color = dialog.getUi().typeColorTable->item (row, 1 )->backgroundColor();
			QString text = dialog.getUi().typeColorTable->item(row,0)->text();

			bool conversionOK;
			cellType=text.toShort(&conversionOK);

			if(!conversionOK)
			   continue; //in case user enters non-number value

			univGraphSet.typePenMap.insert(make_pair(cellType,QPen(color)));
			univGraphSet.typeBrushMap.insert(make_pair(cellType,QBrush(color)));
			univGraphSet.typeColorMap.insert(make_pair(cellType,color));
		}
   }


//    cerr<<"COLOR ASSIGNMENT"<<endl;
//    for(cItr=univGraphSet.typeColorMap.begin() ;  cItr != univGraphSet.typeColorMap.end() ; ++cItr){
//       cerr<<"Type="<<cItr->first<<" color="<<cItr->second.name().toStdString()<<endl;
//    }


      
         drawField();


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::frequencyUpdate(int _value){
   screenshotFrequency=_value;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindow::setClosePlayerAfterSimulationDone(bool){

	if (closePlayerAfterSimulationDoneAct->isChecked()){
		closePlayerAfterSimulationDone=true;
	}else{
		closePlayerAfterSimulationDone=false;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindow::addScreenshot(){

      if(!drawingAllowed)
         return;
         
     mutexFieldDraw.lock();

	  cerr<<"This is addScreenshot"<<endl;
      ScreenshotDescription scshDes;
       
      ScreenshotData * scshPtr=new ScreenshotData();
      QString scrName;
      Configure3DData   data3D;
      if(display3DButton->isChecked()){

         data3D = ((Display3D * )display3D)->getConfigure3DData();
         scrName=screenshotName3D(data3D);

      }else{
         scrName=screenshotName2D();
      }


      bool okToProceed=scshPtr->okToProceed(simulationRootDir,scrName);
		
      if(!okToProceed){
         if(scshPtr) delete scshPtr; scshPtr=0;
			
         mutexFieldDraw.unlock();
         return;

      }




      if(display3DButton->isChecked()){
			cerr<<" 3D after !okToProceed 1"<<endl;
         scshDes.viewDimension="3D";
         
         scshPtr->graphicsPtr=new Display3D(0);
         scshPtr->graphicsPtr->setLatticeType(latticeType);
		   scshPtr->graphicsPtr->setUnivGraphSetPtr(&univGraphSet);
//          cerr<<"SETTING 3D SCREENSHOT"<<endl;
			scshPtr->visualizationWidgetType="3D";
         //scshPtr->setVisualizationWidgetType(QString("3D"));
         
       ((Display3D*)scshPtr->graphicsPtr)->setVisible(false);
      ((Display3D*)scshPtr->graphicsPtr)->resize( QSize(502, 456).expandedTo(minimumSizeHint()) );
      
//     clearWState( WState_Polished );
      


    
         scshPtr->graphicsPtr->setUnivGraphSetPtr(&univGraphSet);
         

         scshPtr->graphicsPtr->setGraphicsDataFieldPtr(graphFieldsPtr);
        ((Display3D * )scshPtr->graphicsPtr)->setSizeLMN(graphFieldsPtr->getSizeL(),graphFieldsPtr->getSizeM(),graphFieldsPtr->getSizeN());
        
        ((Display3D * )scshPtr->graphicsPtr)->setDrawingAllowedFlag(true);
        
         
        ((Display3D * )scshPtr->graphicsPtr)->setConfigure3DData(data3D);
//          cerr<<" CONFIGURE DATA="<<data3D<<endl;
        ((Display3D*)scshPtr->graphicsPtr)->setInitialConfigure3DData(data3D);
         scshDes.data3D=data3D;

      }else{
         //scshDes
         scshDes.viewDimension="2D";
         scshDes.projData=((Graphics2D * )graphics2DPtr)->projData;
		   
			
         //scshData
         scshPtr->graphicsPtr=new Graphics2D(0);
		   
         scshPtr->graphicsPtr->setLatticeType(latticeType);
		   scshPtr->graphicsPtr->setUnivGraphSetPtr(&univGraphSet);
			
         ((Graphics2D * )scshPtr->graphicsPtr)->projData=((Graphics2D * )graphics2DPtr)->projData;
			
         scshPtr->graphicsPtr->setGraphicsDataFieldPtr(graphFieldsPtr);
		
         //scshPtr->setVisualizationWidgetType(QString("2D"));
			scshPtr->visualizationWidgetType="2D";
			

      }


      scshPtr->univGraphSet=univGraphSet;//copying current graphics settings - IMPORTANT
      

      scshPtr->coreName=scrName.toStdString();
      //scshPtr->setCoreName(scrName.toStdString().c_str());
      //scshPtr->setScreenshotIdName(scrName);

      
      QString selectedPlotType=plotTypeComboBox->currentText();


      QString selectedPlotName=plotTypeComboBox->currentText();
      std::string plotType=currentGraphicsPtr->getGraphFieldsPtr()->checkPlotType(selectedPlotName.toStdString());



//      scshPtr->univGraphSet=univGraphSet;//copying current graphics settings - IMPORTANT
//      
//
//      
//      scshPtr->setCoreName(scrName.toStdString().c_str());
//      scshPtr->setScreenshotIdName(scrName);
//
//
//      
//      QString selectedPlotType=plotTypeComboBox->currentText();
//      
////       if( selectedPlotType == QString("Cell Field") ){
//      QString selectedPlotName=plotTypeComboBox->currentText();
//      std::string plotType=currentGraphicsPtr->getGraphFieldsPtr()->checkPlotType(selectedPlotName.toStdString());

      if(plotType == "cell_field"   ){
         scshDes.plotType="cell_field";
         scshDes.plotName="Cell_Field";
         scshPtr->graphicsPtr->setCurrentPainitgFcnPtr(scshPtr->graphicsPtr->getPaintLattice());
         
      }else if (plotType == "scalar"){
            scshDes.plotType="scalar";
            scshDes.plotName=selectedPlotName;

            GraphicsDataFields::floatField3DNameMapItr_t mitr=
            scshPtr->graphicsPtr->getGraphFieldsPtr()->getFloatField3DNameMap().find(string(selectedPlotName.toStdString().c_str()));
            if(mitr != scshPtr->graphicsPtr->getGraphFieldsPtr()->getFloatField3DNameMap().end() ){
            

               scshPtr->graphicsPtr->setCurrentConcentrationFieldPtr(mitr->second);
      
               scshPtr->graphicsPtr->setCurrentPainitgFcnPtr(scshPtr->graphicsPtr->getPaintConcentrationLattice());

               scshDes.minConcentration=minConcentration;
               scshDes.maxConcentration=maxConcentration;
               scshDes.minConcentrationFixed=minConcentrationFixed;
               scshDes.maxConcentrationFixed=maxConcentrationFixed;
               scshDes.minMagnitude=minMagnitude;
               scshDes.maxMagnitude=maxMagnitude;
               scshDes.minMagnitudeFixed=minMagnitudeFixed;
               scshDes.maxMagnitudeFixed=maxMagnitudeFixed;
               
            }         

      }else if(plotType == "vector_cell_level"){
         scshDes.plotType="vector_cell_level";
         scshDes.plotName=selectedPlotName;
         map<std::string, GraphicsDataFields::vectorFieldCellLevel_t * >::iterator mitr=
         currentGraphicsPtr->getGraphFieldsPtr()->getVectorFieldCellLevelNameMap().find(string(selectedPlotName.toStdString().c_str()));
         
         scshPtr->graphicsPtr->setCurrentPainitgFcnPtr(scshPtr->graphicsPtr->getPaintCellVectorFieldLattice());
         scshPtr->graphicsPtr->setCurrentVectorCellLevelFieldPtr(mitr->second);
         

      }


         //scshDes
         scshDes.minConcentration=minConcentration;
         scshDes.maxConcentration=maxConcentration;
         scshDes.minConcentrationFixed=minConcentrationFixed;
         scshDes.maxConcentrationFixed=maxConcentrationFixed;
         scshDes.minMagnitude=minMagnitude;
         scshDes.maxMagnitude=maxMagnitude;
         scshDes.minMagnitudeFixed=minMagnitudeFixed;
         scshDes.maxMagnitudeFixed=maxMagnitudeFixed;



         initializeGraphicsPtr(scshPtr->graphicsPtr);

			scshPtr->simulationRootDir=simulationRootDir.toStdString();
			ostringstream thisDirectoryNameStream;
			thisDirectoryNameStream<<scshPtr->simulationRootDir<<QDir::separator().toAscii()<<scshPtr->coreName;
			cerr<<"IMAGEWINDOW "<<thisDirectoryNameStream.str()<<endl;
			scshPtr->thisDirectoryName=thisDirectoryNameStream.str();
			cerr<<"will check the directory="<<scshPtr->thisDirectoryName.c_str()<<endl;

			bool dirExists=scshPtr->dir.exists(QString(scshPtr->thisDirectoryName.c_str()));
			if(dirExists){
				cerr<<"Directory already exists and is in use. Ignoring request"<<endl;

			}else{
				cerr<<"will make directory: "<<scshPtr->thisDirectoryName<<endl;
				bool dirOK=scshPtr->dir.mkdir(QString(scshPtr->thisDirectoryName.c_str()));
				if(!dirOK && !dirExists){
				   cerr<<"Could not create directory: "<<scshPtr->thisDirectoryName<<endl;
					cerr<<"Make sure you have permissions to write in current directory and that you have enough disk space "<<endl;
				}
			}

			screenshotDescriptionList.push_back(scshDes);
			//          cerr<<"scshDes="<<scshDes<<endl;
			//saving scshData

			screenshotDataList.push_back(scshPtr);


			if(!saveScrDesAct->isEnabled())   
				saveScrDesAct->setEnabled(true);


			if(screenshotDataList.size()>maxScreenshotListLength){
				recordButton->setEnabled(false);
			}


			cerr<<"END OF ADD SCREENSHOT "<<endl;


			mutexFieldDraw.unlock();
			return ;


			//scshPtr->activate(simulationRootDir);

			////saving scshDes
			//screenshotDescriptionList.push_back(scshDes);

			////saving scshData
			//screenshotDataList.push_back(scshPtr);

			//if(!saveScrDesAct->isEnabled())   
			//   saveScrDesAct->setEnabled(true);
			//
			//
			//if(screenshotDataList.size()>maxScreenshotListLength){
			//   recordButton->setEnabled(false);
			//}



			mutexFieldDraw.unlock();


         //scshDes
//         scshDes.minConcentration=minConcentration;
//         scshDes.maxConcentration=maxConcentration;
//         scshDes.minConcentrationFixed=minConcentrationFixed;
//         scshDes.maxConcentrationFixed=maxConcentrationFixed;
//         scshDes.minMagnitude=minMagnitude;
//         scshDes.maxMagnitude=maxMagnitude;
//         scshDes.minMagnitudeFixed=minMagnitudeFixed;
//         scshDes.maxMagnitudeFixed=maxMagnitudeFixed;
//
//
////       {
////             scshDes.plotName=selectedPlotType;
////             
////             GraphicsDataFields::floatField3DNameMapItr_t mitr=
////             scshPtr->graphicsPtr->getGraphFieldsPtr()->getFloatField3DNameMap().find(string(selectedPlotType.toStdString().c_str()));
////       
//// 
////             
////             if(mitr != scshPtr->graphicsPtr->getGraphFieldsPtr()->getFloatField3DNameMap().end() ){
////             
//// 
////                scshPtr->graphicsPtr->setCurrentConcentrationFieldPtr(mitr->second);
////       
////                scshPtr->graphicsPtr->setCurrentPainitgFcnPtr(currentGraphicsPtr->getPaintConcentrationLattice());
////                
////                
////             }else{
////             
////                scshPtr->graphicsPtr->setCurrentConcentrationFieldPtr(& scshPtr->graphicsPtr->getGraphFieldsPtr()->field3DConcentration);
////                scshPtr->graphicsPtr->setCurrentPainitgFcnPtr(scshPtr->graphicsPtr->getPaintConcentrationLattice());
////       
////             }
////          //scshDes
////          scshDes.minConcentration=minConcentration;
////          scshDes.maxConcentration=maxConcentration;
////          scshDes.minConcentrationFixed=minConcentrationFixed;
////          scshDes.maxConcentrationFixed=maxConcentrationFixed;
////          
////          //scshData
////          initializeGraphicsPtr(scshPtr->graphicsPtr);
//// 
////          scshPtr->graphicsPtr->setCurrentPainitgFcnPtr(scshPtr->graphicsPtr->getPaintConcentrationLattice());
////       }
//
//         initializeGraphicsPtr(scshPtr->graphicsPtr);
//
//         scshPtr->activate(simulationRootDir);
//         //saving scshDes
//         screenshotDescriptionList.push_back(scshDes);
//
//         //saving scshData
//         screenshotDataList.push_back(scshPtr);
//
//         if(!saveScrDesAct->isEnabled())   
//            saveScrDesAct->setEnabled(true);
//         
//         
//         if(screenshotDataList.size()>maxScreenshotListLength){
//            recordButton->setEnabled(false);
//         }
//         
//
//
//   mutexFieldDraw.unlock();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
QString ImageWindow::screenshotName2D(){
   QString name;

   
   name+=QString("2D_");
   if(xyButton->isChecked()){
      name+=QString("xy_");
      name+=zSpinBox->text();
   }
   else if(xzButton->isChecked()){
      name+=QString("xz_");
      name+=ySpinBox->text();
   }
   else if(yzButton->isChecked()){
      name+=QString("yz_");
      name+=xSpinBox->text();
   }

   name+=QString("_");
   name+=plotTypeComboBox->currentText();
   return name;

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
QString ImageWindow::screenshotName3D(Configure3DData & _data3D){
   QString name;


   Display3D & d3D = *((Display3D*)display3D);
   name+=QString("3D_");
   name+=QString().setNum(_data3D.rotationX);
   name+=QString("_");
   name+=QString().setNum(_data3D.rotationY);
   name+=QString("_");
   name+=QString().setNum(_data3D.rotationZ);


   name+=QString("_");
   name+=plotTypeComboBox->currentText();
   return name;

}



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
#include "imagewindowBase.h"
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
#include <Python.h>
//#include "glwidget.h"
// #include <QRadioButton>
// // #include <QButtonGroup>


using namespace std;

void ImageWindowBase::initializeImageWindow(bool _silent){
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ImageWindowBase::ImageWindowBase(bool _silent):
     maxScreenshotListLength(20), silent(_silent),screenshotCoreName("screen"),xServerFlag(true),saveSettingsXML(false),advancedSettingsXMLOn(false), noOutputFlag(false),explicitSetXMLFileFlag(false),stopThread(false)//TEMP
{
	initializeImageWindow(_silent);
	//cerr << "ImageWindowBase stopThread ADDRESS: " << &stopThread << "\t" << stopThread << "\n";  // TESTING THREADING //TEMP
	Py_Initialize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ImageWindowBase::~ImageWindowBase()
{
   if(painter) delete painter; painter=0;

   if(graphFieldsPtr) delete graphFieldsPtr; graphFieldsPtr=0;


   for(list<ScreenshotData*>::iterator litr = screenshotDataList.begin() ; litr != screenshotDataList.end() ; ++litr){
      if(*litr){
         delete *litr;
         *litr=0;
      }      
   }


   if(bufferFillUsedSemPtr) delete bufferFillUsedSemPtr; bufferFillUsedSemPtr=0;
   if(bufferFillFreeSemPtr) delete bufferFillFreeSemPtr; bufferFillFreeSemPtr=0;
   

   
   mutex.lock();
   mutex.unlock();
   mutexStartPause.lock();
   mutexStartPause.unlock();


	mutexFieldDraw.lock();
   mutexFieldDraw.unlock();
   mutexTransaction.lock();
   mutexTransaction.unlock();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindowBase::updateSimulationFileNames(){
    if(useXMLFileFlag && fileXML!="")
	{
		fileXMLStripped=strippedName(fileXML);
		setCurrentFile(fileXML);
		setXMLFile(fileXML);

//       curFileStripped=strippedName(curFile);
//       setCurrentFile(curFile);
    }
	else if (runPythonFlag && pyConfData.pythonFileName!="")
	{
		curFile=pyConfData.pythonFileName;
		curFileStripped=strippedName(pyConfData.pythonFileName);
		setCurrentFile(pyConfData.pythonFileName);
    }
	else
	{
		curFileStripped="default_simulation.xml";
		setCurrentFile(pyConfData.pythonFileName);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::setXServerFlag(bool _xServerFlag){xServerFlag=_xServerFlag;}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool ImageWindowBase::getXServerFlag(){return xServerFlag;}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindowBase::initializeGraphicsPtr(GraphicsBase * _graphicsPtr)
{
      _graphicsPtr->setMinConcentrationFixed(minConcentrationFixed);
      _graphicsPtr->setMaxConcentrationFixed(maxConcentrationFixed);
      _graphicsPtr->setMinConcentration(minConcentration);
      _graphicsPtr->setMaxConcentration(maxConcentration);
      _graphicsPtr->setNumberOfLegendBoxes(numberOfLegendBoxes);
      _graphicsPtr->setNumberAccuracy(numberAccuracy);
      _graphicsPtr->setLegendEnable(legendEnable);

      _graphicsPtr->setMinMagnitudeFixed(minMagnitudeFixed);
      _graphicsPtr->setMaxMagnitudeFixed(maxMagnitudeFixed);
      _graphicsPtr->setMinMagnitude(minMagnitude);
      _graphicsPtr->setMaxMagnitude(maxMagnitude);

      _graphicsPtr->setArrowLength(arrowLength);
      _graphicsPtr->setNumberOfLegendBoxesVector(numberOfLegendBoxesVector);
      _graphicsPtr->setNumberAccuracyVector(numberAccuracyVector);
      _graphicsPtr->setLegendEnableVector(legendEnableVector);
      _graphicsPtr->setOverlayVectorCellFields(overlayVectorCellFields);
      _graphicsPtr->setScaleArrows(scaleArrows);
      _graphicsPtr->setFixedArrowColor(fixedArrowColorFlag);
      _graphicsPtr->setSilentMode(silent);
      _graphicsPtr->setXServerFlag(xServerFlag);
}

void ImageWindowBase::initializeGraphicsPtrVec(){

   for(unsigned int i  = 0 ; i < graphicsPtrVec.size() ; ++i){
      initializeGraphicsPtr(graphicsPtrVec[i]);
   }


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindowBase::setCurrentFile(const QString &fileName)
{
    curFile = fileName;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindowBase::setOutputDirectory(const QString &dirName){
   tmpDirName=dirName;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::setNoOutputFlag(bool _flag){
   noOutputFlag=_flag;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::setXMLFile(const QString &fileName){
	fileXML=fileName;
   curFile=fileName;
   curFileStripped=strippedName(fileName);
   setCurrentFile(fileName);
   cerr<<"setXMLFile="<<fileName.toStdString()<<endl;

	( fileName!="" ? useXMLFileFlag=true : useXMLFileFlag=false );

	explicitSetXMLFileFlag=true;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindowBase::setRunPythonFlag(bool _pythonFlag){
   runPythonFlag=_pythonFlag;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::setPythonScript(const QString &fileName){
   pyConfData.pythonFileName=fileName;
   runPythonFlag=true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::setScreenshotDescriptionFileName(const QString & scrDesFileName){
   screenshotDescriptionFileName=scrDesFileName;
   
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::startSimulation(){
   simulation();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::writeSettings(){
   
    QSettings settings("Biocomplexity","CompuCellPlayer");

//     settings.setPath("physics.indiana.edu", "CompuCellPlayer.info");
    settings.beginGroup("/DefaultColors");
    settings.setValue("/brush", univGraphSet.defaultBrush.color().name());
    settings.setValue("/pen", univGraphSet.defaultPen.color().name());
    settings.setValue("/border", univGraphSet.borderPen.color().name());
    settings.setValue("/bordersOn",univGraphSet.bordersOn/*QString().setNum(bordersOn) */);
    settings.setValue("/concentrationLimitsOn",univGraphSet.concentrationLimitsOn/*QString().setNum(bordersOn) */);
    
    settings.setValue("/contour", univGraphSet.contourPen.color().name());
    settings.setValue("/contoursOn",univGraphSet.contoursOn/*QString().setNum(contoursOn)*/ );
    settings.setValue("/arrowColor", univGraphSet.arrowPen.color().name());    

    if(useXMLFileFlag)
		settings.setValue("/recentFile",curFile);

    settings.setValue("/useXMLFileFlag",useXMLFileFlag);
    cerr<<"THIS IS fileXML="<<fileXML.toStdString()<<endl;
    settings.setValue("/fileXML",fileXML);
    

    QStringList penColorList;
    std::map<unsigned short,QPen>::iterator penMitr;

    for( penMitr = univGraphSet.typePenMap.begin() ; penMitr != univGraphSet.typePenMap.end() ; ++penMitr ){
        penColorList+=QString().setNum(penMitr->first);
        penColorList+=penMitr->second.color().name();
    }
    settings.setValue("/typeColorMap", penColorList);
    settings.setValue("/zoomFactor",univGraphSet.zoomFactor);
    settings.setValue("/screenshotFrequency",(int)screenshotFrequency);
    settings.setValue("/screenUpdateFrequency",(int)screenUpdateFrequency);
    settings.setValue("/noOutputFlag",noOutputFlag);

    settings.setValue("/minConcentration",minConcentration);
    settings.setValue("/minConcentrationFixed",minConcentrationFixed);
    settings.setValue("/maxConcentration",maxConcentration);
    settings.setValue("/maxConcentrationFixed",maxConcentrationFixed);

    settings.setValue("/minMagnitude",minMagnitude);
    settings.setValue("/minMagnitudeFixed",minMagnitudeFixed);
    settings.setValue("/maxMagnitude",maxMagnitude);
    settings.setValue("/maxMagnitudeFixed",maxMagnitudeFixed);
    settings.setValue("/numberOfLegendBoxes",(int)numberOfLegendBoxes);
    settings.setValue("/numberAccuracy",numberAccuracy);
    settings.setValue("/legendEnable",legendEnable);
    
    settings.setValue("/arrowLength",arrowLength);
    settings.setValue("/numberOfLegendBoxesVector",(int)numberOfLegendBoxesVector);
    settings.setValue("/numberAccuracyVector",(int)numberAccuracyVector);
    settings.setValue("/legendEnableVector",legendEnableVector);
    settings.setValue("/overlayVectorCellFields",overlayVectorCellFields);
    settings.setValue("/scaleArrows",scaleArrows);
    settings.setValue("/fixedArrowColorFlag",fixedArrowColorFlag);

    settings.setValue("/runPython",runPythonFlag);
    settings.setValue("/pythonFileName",pyConfData.pythonFileName);
    settings.setValue("/closePlayerAfterSimulationDone",closePlayerAfterSimulationDone);
    
    QStringList types3DinvisibleList;
    for(unsigned int i = 0 ; i < univGraphSet.types3DInvisibleVec.size() ; ++i){
		types3DinvisibleList+=QString().setNum(univGraphSet.types3DInvisibleVec[i]);
    }
    settings.setValue("/types3DInvisible", types3DinvisibleList);
    settings.endGroup();

	 cerr<<"write closePlayerAfterSimulationDone="<<closePlayerAfterSimulationDone<<endl;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindowBase::readSettings(){
   
    QSettings settings("Biocomplexity","CompuCellPlayer");
   
    QString defaultBrushColor;
    QString defaultPenColor;
    QString borderPenColor;
    QString contourPenColor;
    QString arrowPenColor;

    
//     settings.setPath("physics.indiana.edu", "CompuCellPlayer.info");
       
    settings.beginGroup("/DefaultColors");
    defaultBrushColor=settings.value("/brush","white").toString();
    defaultPenColor=settings.value("/pen"," white").toString();
    borderPenColor=settings.value("/border","yellow").toString();
    

    univGraphSet.bordersOn=settings.value("/bordersOn",true).toBool();
    univGraphSet.concentrationLimitsOn=settings.value("/concentrationLimitsOn",false).toBool();
    contourPenColor=settings.value("/contour","white").toString();
    univGraphSet.contoursOn=settings.value("/contoursOn",false).toBool();
    arrowPenColor=settings.value("/arrowColor","white").toString();
    curFile=settings.value("/recentFile","cellsort_2D.xml").toString();
    useXMLFileFlag=settings.value("/useXMLFileFlag",false).toBool();
    cerr<<"SETTINGS useXMLFileFlag="<<useXMLFileFlag<<endl;
    fileXML=settings.value("/fileXML","cellsort_2D.xml").toString();

    univGraphSet.defaultBrush.setColor(QColor(defaultBrushColor));
    univGraphSet.defaultPen.setColor(QColor(defaultPenColor));
    univGraphSet.borderPen.setColor(QColor(borderPenColor));
    univGraphSet.contourPen.setColor(QColor(contourPenColor));
    univGraphSet.arrowPen.setColor(QColor(arrowPenColor));

    univGraphSet.defaultColor=QColor(defaultPenColor);
    univGraphSet.borderColor=QColor(borderPenColor);
    univGraphSet.contourColor=QColor(contourPenColor);
    univGraphSet.arrowColor=QColor(arrowPenColor);
        

    QString key,value;
    QStringList defaultPenColorList;
    defaultPenColorList+="0";
    defaultPenColorList+="black";
    defaultPenColorList+="1";
    defaultPenColorList+="green";
    defaultPenColorList+="2";
    defaultPenColorList+="blue";
    defaultPenColorList+="3";
    defaultPenColorList+="red";
    defaultPenColorList+="4";
    defaultPenColorList+="darkorange";
    defaultPenColorList+="5";
    defaultPenColorList+="darksalmon";
    defaultPenColorList+="6";
    defaultPenColorList+="darkviolet";
    defaultPenColorList+="7";
    defaultPenColorList+="navy";
    defaultPenColorList+="8";
    defaultPenColorList+="cyan";
    defaultPenColorList+="9";
    defaultPenColorList+="greenyellow";
    defaultPenColorList+="10";
    defaultPenColorList+="hotpink";
    
    QStringList penColorList=settings.value("/typeColorMap").toStringList();
    if(penColorList.empty()){
      penColorList=defaultPenColorList;
    }

    for ( QStringList::Iterator it = penColorList.begin(); it != penColorList.end();  ) {
		key=*it;
		++it;
		value=*it;
		++it;
		// qDebug() << key << "\t" << value << "\n";
		univGraphSet.typePenMap.insert(std::make_pair(key.toUShort(),QPen(QColor(value))));
		univGraphSet.typeBrushMap.insert(std::make_pair(key.toUShort(),QBrush(QColor(value))));
		univGraphSet.typeColorMap.insert(std::make_pair(key.toUShort(),QColor(value)));
    }

    univGraphSet.zoomFactor=settings.value("/zoomFactor",1).toInt();
    screenshotFrequency=settings.value("/screenshotFrequency",1).toInt();
    screenUpdateFrequency=settings.value("/screenUpdateFrequency",1).toInt();
    noOutputFlag=settings.value("/noOutputFlag",false).toBool();

    minConcentration = settings.value("/minConcentration",0.0).toDouble();
    minConcentrationFixed = settings.value("/minConcentrationFixed",false).toBool();
    maxConcentration = settings.value("/maxConcentration",1.0).toDouble();
    maxConcentrationFixed = settings.value("/maxConcentrationFixed",false).toBool();

    minMagnitude = settings.value("/minMagnitude",0.0).toDouble();
    minMagnitudeFixed = settings.value("/minMagnitudeFixed",false).toBool();
    maxMagnitude = settings.value("/maxMagnitude",1.0).toDouble();
    maxMagnitudeFixed = settings.value("/maxMagnitudeFixed",false).toBool();
    numberOfLegendBoxes=settings.value("/numberOfLegendBoxes",5).toInt();
    numberAccuracy=settings.value("/numberAccuracy",3).toInt();
    legendEnable=settings.value("/legendEnable",true).toBool();
    arrowLength = settings.value("/arrowLength",3).toInt();
    numberOfLegendBoxesVector=settings.value("/numberOfLegendBoxesVector",5).toInt();
    numberAccuracyVector=settings.value("/numberAccuracyVector",3).toInt();
    legendEnableVector=settings.value("/legendEnableVector",true).toBool();
    overlayVectorCellFields=settings.value("/overlayVectorCellFields",false).toBool();
    scaleArrows=settings.value("/scaleArrows",false).toBool();
    fixedArrowColorFlag=settings.value("/fixedArrowColorFlag",false).toBool();


    runPythonFlag=settings.value("/runPython",false).toBool();
    pyConfData.pythonFileName=settings.value("/pythonFileName","defaultCompuCellScript.py").toString();
	 closePlayerAfterSimulationDone=settings.value("/closePlayerAfterSimulationDone",false).toBool();
    
	 cerr<<"read closePlayerAfterSimulationDone="<<closePlayerAfterSimulationDone<<endl;

    QStringList types3DinvisibleList=settings.value("/types3DInvisible").toStringList();
    
    univGraphSet.types3DInvisibleVec.clear();
    univGraphSet.types3DInvisibleVec.push_back(0);//by default avoid displaying medium in 3D
    for ( QStringList::Iterator it = types3DinvisibleList.begin(); it != types3DinvisibleList.end(); ++it ) {
      if((*it).toUShort() != 0)
        univGraphSet.types3DInvisibleVec.push_back((*it).toUShort());
    }

    
    settings.endGroup();

   
    
    
}

bool ImageWindowBase::errorHandler(QString header, QString text){return true;}

void ImageWindowBase::updatePlayerSettings(CompuCell3D::PlayerSettings & playerSettings){

   
    QString defaultBrushColor;
    QString defaultPenColor;
    QString borderPenColor;
    QString contourPenColor;
    QString arrowPenColor;



    if(playerSettings.borderColorNameFlag){
      borderPenColor=QString(playerSettings.borderColorName.c_str());
      univGraphSet.borderPen.setColor(QColor(borderPenColor));
      univGraphSet.borderColor=QColor(borderPenColor);
    }
    if(playerSettings.borderOnFlag){
      univGraphSet.bordersOn=playerSettings.borderOn;
    }

    if(playerSettings.concentrationLimitsOnFlag){
      univGraphSet.concentrationLimitsOn=playerSettings.concentrationLimitsOn;
    }

    if(playerSettings.contourColorNameFlag){
      contourPenColor=QString(playerSettings.contourColorName.c_str());
      univGraphSet.contourPen.setColor(QColor(contourPenColor));
      univGraphSet.contourColor=QColor(contourPenColor);
    }
    if(playerSettings.contourOnFlag){
      univGraphSet.contoursOn=playerSettings.contourOn;
    }
    if(playerSettings.arrowColorNameFlag){
      arrowPenColor=QString(playerSettings.arrowColorName.c_str());
      univGraphSet.arrowPen.setColor(QColor(arrowPenColor));
      univGraphSet.arrowColor=QColor(arrowPenColor);

    }


    if(playerSettings.typeToColorNameMapFlag){
      map<unsigned short,std::string>::iterator mitr;
      for ( mitr = playerSettings.typeToColorNameMap.begin(); mitr != playerSettings.typeToColorNameMap.end(); ++mitr){

//          univGraphSet.typePenMap.insert(std::make_pair(mitr->first,QPen(QColor(QString(mitr->second.c_str())))));
//          univGraphSet.typeBrushMap.insert(std::make_pair(mitr->first,QBrush(QColor(QString(mitr->second.c_str())))));
//          univGraphSet.typeColorMap.insert(std::make_pair(mitr->first,QColor(QString(mitr->second.c_str()))));

         univGraphSet.typePenMap[mitr->first]=QPen(QColor(QString(mitr->second.c_str())));
         univGraphSet.typeBrushMap[mitr->first]=QBrush(QColor(QString(mitr->second.c_str())));
         univGraphSet.typeColorMap[mitr->first]=QColor(QString(mitr->second.c_str()));




      }
    }      
    


    if(playerSettings.zoomFactorFlag){
      univGraphSet.zoomFactor=playerSettings.zoomFactor;
    }
    if(playerSettings.screenshotFrequencyFlag){
      screenshotFrequency=playerSettings.screenshotFrequency;
      //cerr<<"\n\n\n \t\t\t THIS IS SCREENSHOT FREQUENCY="<<screenshotFrequency<<endl;
    }

    if(playerSettings.screenUpdateFrequencyFlag){
      screenUpdateFrequency=playerSettings.screenUpdateFrequency;
    }

    if(playerSettings.noOutputFlagFlag){
      noOutputFlag=playerSettings.noOutputFlag;
    }

    if(playerSettings.minConcentrationFlag){
      minConcentration=playerSettings.minConcentration;
    }
    if(playerSettings.minConcentrationFixedFlag){
      minConcentrationFixed=playerSettings.minConcentrationFixed;
    }

    
    if(playerSettings.maxConcentrationFlag){
      maxConcentration=playerSettings.maxConcentration;
    }
    if(playerSettings.maxConcentrationFixedFlag){
      maxConcentrationFixed=playerSettings.maxConcentrationFixed;
    }
    
    if(playerSettings.minMagnitudeFlag){
      minMagnitude=playerSettings.minMagnitude;
    }
    if(playerSettings.minMagnitudeFixedFlag){
      minMagnitudeFixed=playerSettings.minMagnitudeFixed;
    }

    
    if(playerSettings.maxMagnitudeFlag){
      maxMagnitude=playerSettings.maxMagnitude;
    }
    if(playerSettings.maxMagnitudeFixedFlag){
      maxMagnitudeFixed=playerSettings.maxMagnitudeFixed;
    }
    
    if(playerSettings.numberOfLegendBoxesFlag){
      numberOfLegendBoxes=playerSettings.numberOfLegendBoxes;
    }
    if(playerSettings.numberAccuracyFlag){
      numberAccuracy=playerSettings.numberAccuracy;
    }
    if(playerSettings.legendEnableFlag){
      legendEnable=playerSettings.legendEnable;
    }
    if(playerSettings.numberOfLegendBoxesVectorFlag){
      numberOfLegendBoxesVector=playerSettings.numberOfLegendBoxesVector;
    }
    if(playerSettings.numberAccuracyVectorFlag){
      numberAccuracyVector=playerSettings.numberAccuracyVector;
    }
    if(playerSettings.legendEnableVectorFlag){
      legendEnableVector=playerSettings.legendEnableVector;
    }
    if(playerSettings.arrowLengthFlag){
      arrowLength=playerSettings.arrowLength;
    }
    if(playerSettings.overlayVectorCellFieldsFlag){
      overlayVectorCellFields=playerSettings.overlayVectorCellFields;
    }

    if(playerSettings.scaleArrowsFlag){
      scaleArrows=playerSettings.scaleArrows;
    }
    if(playerSettings.fixedArrowColorFlagFlag){
      fixedArrowColorFlag=playerSettings.fixedArrowColorFlag;
    }
    if(playerSettings.types3DInvisibleFlag){
      for(vector<unsigned short>::iterator vitr = playerSettings.types3DInvisible.begin() ; vitr != playerSettings.types3DInvisible.end() ; ++vitr){
         univGraphSet.types3DInvisibleVec.push_back(*vitr);
      }
    }

   if(playerSettings.saveSettingsFlag){
      saveSettingsXML=playerSettings.saveSettings;
   }

   //initializing graphics objects with new settings
   initializeGraphicsPtrVec();

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::closeEvent(QCloseEvent *event){
   
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::customEvent(QEvent *event){
   //cerr<<"\t\t\t\t THIS IS IMAGEWINDOWBASE"<<endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// void ImageWindowBase::drawField(){
// }
// 
// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// //this function draws Field in 2D
// 
// void ImageWindowBase::drawField2D(){
// 
// }
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// //this function draws Field in 3D
// void ImageWindowBase::drawField3D(){
//    return;
//       
//    
// }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

QString ImageWindowBase::screenshotCoreNameFromScreenshotDescription(const ScreenshotDescription & _scrDsc){
   QString scshCoreName;
   scshCoreName=_scrDsc.viewDimension+"_"+_scrDsc.plotName+"_"+_scrDsc.plotType;
   if(_scrDsc.viewDimension=="2D"){
         scshCoreName=scshCoreName+"_"+QString(_scrDsc.projData.projection.c_str())+"_";
         if(_scrDsc.projData.projection=="xy"){
            scshCoreName+=QString::number(_scrDsc.projData.zMin);
         }else if(_scrDsc.projData.projection=="xz"){
            scshCoreName+=QString::number(_scrDsc.projData.yMin);
         }else if(_scrDsc.projData.projection=="yz"){
            scshCoreName+=QString::number(_scrDsc.projData.xMin);
         }
   }else if(_scrDsc.viewDimension=="3D"){
      scshCoreName=scshCoreName+"_"+QString::number(_scrDsc.data3D.rotationX)+"_"+QString::number(_scrDsc.data3D.rotationY)+"_"+QString::number(_scrDsc.data3D.rotationZ);
   }
   return scshCoreName;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindowBase::produceScreenshotDataList(const std::list<ScreenshotDescription> & _screenshotDescriptionList){
   //This function produces list of screenshot descriptions and stores them in a list. Then every screenshotFrequency MCS
   //this list is being read and screenshots are taken.
   
	//cerr<<" \n\n INSIDE PRODUCESCREENSHOTDATA \n\n"<<endl;
   //will clear screenshotDataList just in case...
   for (list<ScreenshotData*>::iterator litr = screenshotDataList.begin() ; litr != screenshotDataList.end() ; ++litr){

      delete *litr;
      *litr=0;
   }
   screenshotDataList.clear();

   int scrCounter=0;


   
   for ( list<ScreenshotDescription>::const_iterator litr = _screenshotDescriptionList.begin() ;
         litr != _screenshotDescriptionList.end() ;
         ++litr )
   {
      ScreenshotData * scshPtr=new ScreenshotData();
      if(litr->viewDimension=="3D"){
          scshPtr->graphicsPtr=new Display3D_NOX();
			 scshPtr->graphicsPtr->setUnivGraphSetPtr(&univGraphSet);
          scshPtr->graphicsPtr->setLatticeType(latticeType);
         //scshPtr->setVisualizationWidgetType(QString("3D"));
			scshPtr->visualizationWidgetType="3D";
//          ((Display3D*)scshPtr->graphicsPtr)->setVisible(false);
//          ((Display3D*)scshPtr->graphicsPtr)->resize( QSize(502, 456).expandedTo(minimumSizeHint()) );
         scshPtr->graphicsPtr->setGraphicsDataFieldPtr(graphFieldsPtr);
        ((Display3DBase * )scshPtr->graphicsPtr)->setSizeLMN(graphFieldsPtr->getSizeL(),graphFieldsPtr->getSizeM(),graphFieldsPtr->getSizeN());
        ((Display3DBase * )scshPtr->graphicsPtr)->setDrawingAllowedFlag(true);
        ((Display3DBase * )scshPtr->graphicsPtr)->setConfigure3DData(litr->data3D);
        ((Display3DBase*)scshPtr->graphicsPtr)->setInitialConfigure3DData(litr->data3D);
      }else{
      
         //check if projData are within field boundaries
         if(! litr->projData.checkIfCompatibleWithField(graphFieldsPtr->getSizeL(),graphFieldsPtr->getSizeM(),graphFieldsPtr->getSizeN()) ){
            delete scshPtr ;
            continue;
         }
         
         scshPtr->graphicsPtr=new Graphics2D_NOX();
			scshPtr->graphicsPtr->setUnivGraphSetPtr(&univGraphSet);
         scshPtr->graphicsPtr->setLatticeType(latticeType);
         //scshPtr->setVisualizationWidgetType(QString("2D"));
			scshPtr->visualizationWidgetType="2D";
         ((Graphics2D * )scshPtr->graphicsPtr)->projData=litr->projData;
         scshPtr->graphicsPtr->setGraphicsDataFieldPtr(graphFieldsPtr);
      }
      
      scshPtr->univGraphSet=univGraphSet;//copying current graphics settings - IMPORTANT
//       initializeGraphicsPtr(scshPtr->graphicsPtr);

      QString selectedPlotType=litr->plotType;
      QString selectedPlotName=litr->plotName;
//       cerr<<" \n\n\n selectedPlotType="<<selectedPlotType.toStdString()<<" \n\n\n"<<endl;
      
      if( selectedPlotType == QString("cell_field") ){
         
         scshPtr->graphicsPtr->setCurrentPainitgFcnPtr(scshPtr->graphicsPtr->getPaintLattice());

         initializeGraphicsPtr(scshPtr->graphicsPtr); //initialize graphics settings to those in the config file

      }else if (selectedPlotType == QString("vector_cell_level")){

         

         map<std::string, GraphicsDataFields::vectorFieldCellLevel_t * >::iterator mitr=
         scshPtr->graphicsPtr->getGraphFieldsPtr()->getVectorFieldCellLevelNameMap().find(string(selectedPlotName.toStdString().c_str()));
         
         scshPtr->graphicsPtr->setCurrentPainitgFcnPtr(scshPtr->graphicsPtr->getPaintCellVectorFieldLattice());
         scshPtr->graphicsPtr->setCurrentVectorCellLevelFieldPtr(mitr->second);

         initializeGraphicsPtr(scshPtr->graphicsPtr);//initialize graphics settings to those in the config file


//          if(silent && scshPtr->getVisualizationWidgetType()==QString("2D")){//Qt 4.2 as of 07/2007 does not permit to draw text without connecting to X-server - that's why there is no legend in the silend mode
//             scshPtr->graphicsPtr->setLegendEnable(false);
//             scshPtr->graphicsPtr->setLegendEnableVector(false);
//          }


      }else if (selectedPlotType == QString("scalar")){
         

         
         GraphicsDataFields::floatField3DNameMapItr_t mitr=
         scshPtr->graphicsPtr->getGraphFieldsPtr()->getFloatField3DNameMap().find(string(selectedPlotName.toStdString().c_str()));
   
//          cerr<<"LOOKING FOR A STRING:"<<string(selectedPlotType.toStdString().c_str())<<endl;
         
         if(mitr != scshPtr->graphicsPtr->getGraphFieldsPtr()->getFloatField3DNameMap().end() ){
         
//             cerr<<"FOUND:"<<string(selectedPlotType.toStdString().c_str())<<endl;
            scshPtr->graphicsPtr->setCurrentConcentrationFieldPtr(mitr->second);
   
            scshPtr->graphicsPtr->setCurrentPainitgFcnPtr(currentGraphicsPtr->getPaintConcentrationLattice());

            initializeGraphicsPtr(scshPtr->graphicsPtr);//initialize graphics settings to those in the config file


//             if(silent && scshPtr->getVisualizationWidgetType()==QString("2D")){//Qt 4.2 as of 07/2007 does not permit to draw text without connecting to X-server - that's why there is no legend in the silend mode
//                scshPtr->graphicsPtr->setLegendEnable(false);
//                scshPtr->graphicsPtr->setLegendEnableVector(false);
//             }

                        
         }else{
            delete scshPtr ;
            continue;   
         }
      
      }else{
         cerr<<"Unknown plot type: "<<selectedPlotType.toStdString()<<" . Exiting..."<<endl;
         exit(0);
      }
      
      if(screenshotDataList.size()<=maxScreenshotListLength){

         QString currentScreenshotName=screenshotCoreName+QString().setNum(scrCounter);
         scrCounter++;

          scshPtr->univGraphSet=univGraphSet;//copying current graphics settings - IMPORTANT
         //scshPtr->setCoreName(currentScreenshotName.toStdString().c_str());
		 //scshPtr->setScreenshotIdName(currentScreenshotName);

         scshPtr->coreName=currentScreenshotName.toStdString();
		 scshPtr->simulationRootDir=simulationRootDir.toStdString();

	  ostringstream thisDirectoryNameStream;
      thisDirectoryNameStream<<scshPtr->simulationRootDir<<QDir::separator().toAscii()<<scshPtr->coreName;
	  //cerr<<"IMAGEWINDOW "<<thisDirectoryNameStream.str()<<endl;
	  scshPtr->thisDirectoryName=thisDirectoryNameStream.str();

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



         screenshotDataList.push_back(scshPtr);

         //scshPtr->activate(simulationRootDir);
      }

//      if(screenshotDataList.size()<=maxScreenshotListLength){
//      
//         QString currentScreenshotName=screenshotCoreName+QString().setNum(scrCounter);
//         scrCounter++;
//
////          scshPtr->univGraphSet=univGraphSet;//copying current graphics settings - IMPORTANT
//         currentScreenshotName=screenshotCoreNameFromScreenshotDescription(*litr);
//         scshPtr->setCoreName(currentScreenshotName.toStdString().c_str());
//         scshPtr->setScreenshotIdName(currentScreenshotName);
//         
//         screenshotDataList.push_back(scshPtr);
//         cerr<<"simulationRootDir="<<simulationRootDir.toStdString()<<endl;
//         scshPtr->activate(simulationRootDir);
//      }
   }
//    exit(0);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ImageWindowBase::takeCurrentScreenshot(const std::string &_imageFileFullName, const std::string & _imageFileName){
   
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindowBase::outputScreenshot(){

   
   screenshotCounter=mcStep;

   if(screenshotCounter % screenshotFrequency){
//        ++screenshotCounter;
      return;
   }

//    cerr<<"OUTPUTING SCREENSHOT 1"<<endl;
   ostringstream imageFileFullNameStream("");
   ostringstream imageFileNameStream("");
   imageCoreFileName=curFileStripped.toStdString();
//    cerr<<"imageCoreFileName="<<imageCoreFileName<<endl;
//    cerr<<"tmpDirName.toStdString()="<<tmpDirName.toStdString()<<endl;
   imageFileExtension="png";
   imageFileFullNameStream<<tmpDirName.toStdString()<<QString(QDir::separator()).toStdString()<<imageCoreFileName<<".";

   imageFileFullNameStream.width(numScreenNameDigits);
   imageFileFullNameStream.fill('0');
   imageFileFullNameStream<<screenshotCounter<<"."<<imageFileExtension;

   imageFileNameStream<<imageCoreFileName<<".";
   imageFileNameStream.width(numScreenNameDigits);
   imageFileNameStream.fill('0');
   imageFileNameStream<<screenshotCounter<<"."<<imageFileExtension;
	       
   takeCurrentScreenshot(imageFileFullNameStream.str(),imageFileNameStream.str());
   //cerr<<"OUTPUTING SCREENSHOT 2"<<endl;
   //for(list<ScreenshotData*>::iterator litr = screenshotDataList.begin() ; litr != screenshotDataList.end() ; ++litr){
   //   cerr<<"Screenshot"<<endl;
   //   (*litr)->outputScreenshot(screenshotCounter,numScreenNameDigits);
   //   
   //}
   for(list<ScreenshotData*>::iterator litr = screenshotDataList.begin() ; litr != screenshotDataList.end() ; ++litr){
	   
    //  (*litr)->outputScreenshot(screenshotCounter,numScreenNameDigits);

		QImage image;
		ostringstream numberStream;

		numberStream.width(numScreenNameDigits);
		numberStream.fill('0');
		numberStream<<screenshotCounter;

		QString screenshotName(QString((*litr)->coreName.c_str())+"."+QString(numberStream.str().c_str())+".png");
        QString screenshotFullName(QString((*litr)->thisDirectoryName.c_str())+QString(QDir::separator())+screenshotName);
                
                
		//cerr<<"\n\n\nTHIS IS GRAPHICS POINTER "<<(*litr)->graphicsPtr<<endl;
		//(*litr)->graphicsPtr->drawCurrentScene();
		if((*litr)->visualizationWidgetType=="3D"){
// 		  cerr<<" \n\n\n CALLING PRDUCE IMAGE FOR 3D WIDGET \n\n\n"<<endl;
		  ((Display3DBase*)((*litr)->graphicsPtr))->drawCurrentScene();
                  
		  ((Display3DBase*)((*litr)->graphicsPtr))->produceImage(screenshotFullName.toStdString());

		}else{
// 		  cerr<<" \n\n\n CALLING PRODUCE IMAGE FOR 2D WIDGET \n\n\n"<<endl;
		  ((Graphics2DBase*)((*litr)->graphicsPtr))->drawCurrentScene();
		  //cerr<<" \n\n\n AFTER DRAW SCENE \n\n\n"<<endl;
		  ((Graphics2DBase*)((*litr)->graphicsPtr))->produceImage(image);
		  //cerr<<"AFTER CALLING PRDUCE IMAGE FOR 2D WIDGET \n\n\n"<<endl;
		  image.save(QString(screenshotFullName),"PNG");

	   }


   }
   
   ++screenshotCounter;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



QString ImageWindowBase::strippedName(const QString &fullFileName)
{
    return QFileInfo(fullFileName).fileName();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageWindowBase::simulation()
{
    bool isLocked = true;
	 cerr<<"GOT HERE IN SIMULATION"<<endl;
    if (mutexStartPause.tryLock()) 
	{
		mutexStartPause.unlock();
		isLocked = false;
    }

    // dealing with situation when useXMLFileFlagis 1 in setting flag but calling python script only 
    // through command line
    if(!explicitSetXMLFileFlag)
	{
		if(runPythonFlag)
			useXMLFileFlag=false;
    }
	cerr<<"isLocked="<<isLocked<< " gotErrorFlag="<<gotErrorFlag<<endl;
	cerr<<"TRANSACTIONS IN PROCESS="<<computeThread.sizeTransactions()<<endl;
	
	if(isLocked && gotErrorFlag){
		gotErrorFlag=false;
		mutex.unlock();
		mutexStartPause.unlock();
		isLocked = false;
	}
	cerr<<"AFTER UNLOCKING isLocked="<<isLocked<< " gotErrorFlag="<<gotErrorFlag<<endl;

	if(!isLocked) ///this happens only once you start program
	{
		if(useXMLFileFlag)
		{
			// cerr<<"new transaction !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<curFile.toStdString()<<endl;
		
			transactionCC3D= new CC3DTransaction(string(curFile.toAscii()));
		}
		else
		{
			transactionCC3D= new CC3DTransaction("");
		}

		transactionCC3D->setPauseMutexPtr(&mutex);
		transactionCC3D->setFieldDrawMutexPtr(&mutexFieldDraw);
		transactionCC3D->setTransactionMutexPtr(&mutexTransaction);
		transactionCC3D->setBufferFreeSem(bufferFillFreeSemPtr);
		transactionCC3D->setBufferUsedSem(bufferFillUsedSemPtr);
		transactionCC3D->setRunPythonFlag(runPythonFlag);
		transactionCC3D->setUseXMLFileFlag(useXMLFileFlag);
		transactionCC3D->setPyDataConf(pyConfData);


		transactionCC3D->setGraphicsDataFieldPtr(graphFieldsPtr);
		transactionCC3D->setScreenUpdateFrequency(screenUpdateFrequency);

		addTransaction(transactionCC3D);

	}
	else
	{
		mutex.unlock();
		mutexStartPause.unlock();
	}

   //cerr<<"\t\t\t\t\t\tGOT SIMULATION DONE"<<endl;

}

void ImageWindowBase::addTransaction(Transaction *transact)
{
    computeThread.addTransaction(transact);

	//cerr << "!!!!!!!!!!!!!!!!!!!!!!!!! computeThread.sizeTransactions(): " << computeThread.sizeTransactions() << "\n";
}


void ImageWindowBase::readScreenshotDescriptionList(std::list<ScreenshotDescription> & _screenshotDescriptionList, const std::string &fileName){

   ifstream in(fileName.c_str());
   if(!in.good()){
      cerr<<"Could not open file "<<fileName<<endl;
      exit(1);
   }

   _screenshotDescriptionList.clear();
   
   while(!in.eof()){
      ScreenshotDescription scshDes;
      in>>scshDes;

      if(!in.fail()){
      cerr<<" new scrDes:"<<endl;
      cerr<<scshDes<<endl;

         _screenshotDescriptionList.push_back(scshDes);
      }
   }
   
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   


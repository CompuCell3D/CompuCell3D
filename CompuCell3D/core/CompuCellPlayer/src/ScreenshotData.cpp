#include "ScreenshotData.h"
#include "GraphicsBase.h"
#include <iostream>
#include <sstream>

using namespace std;

//ScreenshotData::ScreenshotData(){
//   graphicsPtr=0;
//   inventoryFilePtr=0;
//}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ScreenshotData::~ScreenshotData(){
   if(graphicsPtr)
      delete graphicsPtr;
   graphicsPtr=0;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ScreenshotData::setCoreName( std::string   _name )
{
   coreName=_name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ScreenshotData::setCoreNameQ(QString  & _name){

	coreNameQ=_name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ScreenshotData::setCoreNameC(const char * _name){

	//coreNameChar=_name;
	//coreName=string(_name);
	//coreName=string("CELL_FIELD");
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ScreenshotData::setScreenshotIdName(const QString & scrName){
	screenshotIdName = scrName.toStdString();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ScreenshotData::setVisualizationWidgetType(const QString & _type){
   visualizationWidgetType=_type.toStdString();

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool ScreenshotData::okToProceed(const QString & _simulationRootDir,const QString & _coreName){
   QString thisDirName;
   thisDirName=QString(_simulationRootDir+QDir::separator()+_coreName);

	cerr<<"_simulationRootDir="<<_simulationRootDir.toStdString()<<" _coreName="<<_coreName.toStdString()<<endl;
   bool dirExists=dir.exists(thisDirName);
	cerr<<"dirExists="<<dirExists<<endl;
   if(dirExists){
      cerr<<"Directory already exists and is in use. Ignoring request"<<endl;
   }
   
   return !dirExists;


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//void ScreenshotData::activate(const QString & _simulationRootDir){
void ScreenshotData::activate(){
   //simulationRootDir=_simulationRootDir;
   //thisDirectoryName=QString(QString(simulationRootDir.c_str())+QString(QDir::separator())+QString(coreName.c_str()));
   //ostringstream thisDirectoryNameStream;
   //thisDirectoryNameStream<<simulationRootDir<<QDir::separator().toAscii()<<coreName;
   

   bool dirExists=dir.exists(QString(thisDirectoryName.c_str()));
   if(dirExists){
      cerr<<"Directory already exists and is in use. Ignoring request"<<endl;
      return;
   }
   cerr<<"will make directory: "<<thisDirectoryName<<endl;
   bool dirOK=dir.mkdir(QString(thisDirectoryName.c_str()));
      if(!dirOK && !dirExists){
         cerr<<"Could not create directory: "<<thisDirectoryName<<endl;
         cerr<<"Make sure you have permissions to write in current directory and that you have enough disk space "<<endl;
         return;
      }
	return;
	graphicsPtr->setUnivGraphSetPtr(&univGraphSet);
   //fullInventoryFileName = QString(thisDirectoryName+QString(QDir::separator())+QString(coreName.c_str())+".grl");
   //
   //inventoryFilePtr=new ofstream();
   //inventoryFilePtr->open(fullInventoryFileName.toStdString().c_str());

   //graphicsPtr->setUnivGraphSetPtr(&univGraphSet);
   
   ////cerr<<"&univGraphSet="<<&univGraphSet<<endl;
   ////cerr<<"graphicsPtr="<<graphicsPtr<<endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ScreenshotData::outputScreenshot(unsigned int screenshotCounter,unsigned int numScreenNameDigits){
    QImage image;
    ofstream &out = *inventoryFilePtr;

    ostringstream numberStream;

    numberStream.width(numScreenNameDigits);
    numberStream.fill('0');
    numberStream<<screenshotCounter;


    QString screenshotName(QString(coreName.c_str())+"."+QString(numberStream.str().c_str())+".png");
    QString screenshotFullName(QString(thisDirectoryName.c_str())+QString(QDir::separator())+screenshotName);
    //draws scene (cell field concentration etc. depending on the initialization)
    //(graphicsPtr->*(graphicsPtr->getCurrentPainitgFcnPtr()))();
    
    graphicsPtr->drawCurrentScene();
    
    //cerr<<"GOT HERE"<<endl;
    //cerr<<"fcn address="<<graphicsPtr->getCurrentPainitgFcnPtr()<<endl;
    //cerr<<"graphicsPtr="<<graphicsPtr<<endl;

    //cerr<<"visualizationWidgetType="<<visualizationWidgetType.toStdString()<<endl;
	cerr<<"visualizationWidgetType="<<visualizationWidgetType<<endl;

    //if(visualizationWidgetType==QString("3D")){
	if(visualizationWidgetType=="3D"){
      //cerr<<" \n\n\n CALLING PRDUCE IMAGE FOR 3D WIDGET \n\n\n"<<endl;
      graphicsPtr->produceImage(screenshotFullName.toStdString());

    }else{
      //cerr<<" \n\n\n CALLING PRDUCE IMAGE FOR 2D WIDGET \n\n\n"<<endl;
      graphicsPtr->produceImage(image);
	  //cerr<<"AFTER CALLING PRDUCE IMAGE FOR 2D WIDGET \n\n\n"<<endl;
      image.save(QString(screenshotFullName),"PNG");

   }
      //adding entry to file list
    out<<screenshotName.toStdString()<<endl;
    

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





















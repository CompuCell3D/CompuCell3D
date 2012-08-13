#ifndef SCREENSHOTDATA_H
#define SCREENSHOTDATA_H

#include "UniversalGraphicsSettings.h"
#include <string>
// #include <qstring.h>
// #include <qdir.h>
#include <QDir>
#include <fstream>

class GraphicsBase;

class ScreenshotData{

   public:
	   ScreenshotData(){
			graphicsPtr=0;
			inventoryFilePtr=0;
	   
	   };
      ~ScreenshotData();
      UniversalGraphicsSettings univGraphSet;
      GraphicsBase *graphicsPtr;
      //QImage & getImage(){};
      void setCoreName(std::string  _name );
	  void setCoreNameQ(QString  & _name );
	  void setCoreNameC(const char * _name );
      //void activate(const QString & _simulationRootDir);
	  void activate();
      void outputScreenshot(unsigned int screenshotCounter,unsigned int numScreenNameDigits);
      void (GraphicsBase::*drawFcnPtr)(void);
      void setScreenshotIdName(const QString &);
      void setVisualizationWidgetType(const QString &);
      bool okToProceed(const QString & _simulationRootDir,const QString & _coreName); //will check if given directry exists

	  std::string coreName;
	  //QString screenshotIdName;
	  std::string screenshotIdName;
	  std::string visualizationWidgetType;
	  std::string simulationRootDir;
	  std::string thisDirectoryName;
	  QDir dir;
   private:
      ScreenshotData(const ScreenshotData &); /// forbid copying - avoid problems with pointer swapping etc.
      
      QString fullInventoryFileName;
      //QString thisDirectoryName;
      //QString simulationRootDir;
      //QString visualizationWidgetType;
      
      std::ofstream *inventoryFilePtr;
      
	  QString coreNameQ;
	  char * coreNameChar;
      
};



#endif

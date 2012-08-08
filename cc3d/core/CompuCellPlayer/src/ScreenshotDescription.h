#ifndef SCREENSHOTDESCRIPTION_H
#define SCREENSHOTDESCRIPTION_H
#include <iostream>
#include <QString>
#include <string>
#include "Projection2DData.h"
#include "Configure3DData.h"

class ScreenshotDescription{
   public:
      ScreenshotDescription():
      viewDimension("none"),
      plotType("none"),
      plotName("none"),
      minConcentrationFixed(false),
      maxConcentrationFixed(false),
      minConcentration(0),
      maxConcentration(1),
      minMagnitudeFixed(false),
      maxMagnitudeFixed(false),
      minMagnitude(0),
      maxMagnitude(1)

      {}   
      QString viewDimension;
      QString plotType;
      QString plotName;
      
      bool minConcentrationFixed;
      bool maxConcentrationFixed;
      float minConcentration;
      float maxConcentration;
      bool minMagnitudeFixed;
      bool maxMagnitudeFixed;
      float minMagnitude;
      float maxMagnitude;

      Projection2DData projData;
      Configure3DData data3D;


};

inline std::ostream & operator<<(std::ostream & out,const ScreenshotDescription & scrDes){
   using namespace std;

   out<<scrDes.viewDimension.toStdString()<<" ";
   out<<scrDes.plotType.toStdString()<<" ";
   out<<scrDes.plotName.toStdString()<<" ";
   out<<(bool)scrDes.minConcentrationFixed<<" ";
   out<<(bool)scrDes.maxConcentrationFixed<<" ";
   out<<scrDes.minConcentration<<" ";
   out<<scrDes.maxConcentration<<" ";
   out<<(bool)scrDes.minMagnitudeFixed<<" ";
   out<<(bool)scrDes.maxMagnitudeFixed<<" ";
   out<<scrDes.minMagnitude<<" ";
   out<<scrDes.maxMagnitude<<" ";
   out<<scrDes.projData<<" ";
   out<<scrDes.data3D<<" ";
   return out;

}

inline std::istream & operator>>(std::istream & in,ScreenshotDescription & scrDes){
   using namespace std;

   string viewDimension;
   string plotType;
   string plotName;

   in>>viewDimension;
   in>>plotType;
   in>>plotName;
   scrDes.viewDimension=viewDimension.c_str();
   scrDes.plotType=plotType.c_str();
   scrDes.plotName=plotName.c_str();

   in>>scrDes.minConcentrationFixed;
   in>>scrDes.maxConcentrationFixed;
   in>>scrDes.minConcentration;
   in>>scrDes.maxConcentration;
   in>>scrDes.minMagnitudeFixed;
   in>>scrDes.maxMagnitudeFixed;
   in>>scrDes.minMagnitude;
   in>>scrDes.maxMagnitude;

   in>>scrDes.projData;
   in>>scrDes.data3D;
   return in;
}


#endif


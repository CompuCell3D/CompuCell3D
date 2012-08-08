#ifndef PLAYERSETTINGS_H
#define PLAYERSETTINGS_H

#include <string>
#include <map>
#include <vector>
#include <CompuCell3D/ParseData.h>

#include "PlayerSettingsDLLSpecifier.h"

namespace CompuCell3D{
class PLAYERSETTINGS_EXPORT  PlayerSettings : public ParseData{

   public:
      PlayerSettings():
      ParseData("PlayerSettings"),
      xyProj(0),
      xzProj(0),
      yzProj(0),
      rotationX(0),
      rotationY(0),
      rotationZ(0),
      sizeX3D(0),
      sizeY3D(0),
      sizeZ3D(0),
      xyProjFlag(false),
      xzProjFlag(false),
      yzProjFlag(false),
      rotationXFlag(false),
      rotationYFlag(false),
      rotationZFlag(false),
      sizeX3DFlag(false),
      sizeY3DFlag(false),
      sizeZ3DFlag(false),
      initialProjection("xy"),
      advancedSettingsOn(false),
      saveSettings(false),
      saveSettingsFlag(false),
      minConcentration(0),
      minConcentrationFlag(false),
      minConcentrationFixed(false),
      minConcentrationFixedFlag(false),
      maxConcentration(1.0),
      maxConcentrationFlag(false),
      maxConcentrationFixed(false),
      maxConcentrationFixedFlag(false),
      numberOfLegendBoxes(5),
      numberOfLegendBoxesFlag(false),
      numberAccuracy(2),
      numberAccuracyFlag(false),
      legendEnable(false),
      legendEnableFlag(false),
      minMagnitude(0),
      minMagnitudeFlag(false),
      minMagnitudeFixed(false),
      minMagnitudeFixedFlag(false),
      maxMagnitude(1.0),
      maxMagnitudeFlag(false),
      maxMagnitudeFixed(false),
      maxMagnitudeFixedFlag(false),
      arrowLength(3.0),
      arrowLengthFlag(false),
      numberOfLegendBoxesVector(5),
      numberOfLegendBoxesVectorFlag(false),
      numberAccuracyVector(2),
      numberAccuracyVectorFlag(false),
      legendEnableVector(false),
      legendEnableVectorFlag(false),
      overlayVectorCellFields(false),
      overlayVectorCellFieldsFlag(false),
      scaleArrows(false),
      scaleArrowsFlag(false),
      fixedArrowColorFlag(false),      
      fixedArrowColorFlagFlag(false),

      arrowColorName("white"),
      arrowColorNameFlag(false),
      typeToColorNameMapFlag(false),
      contourColorName("white"),
      contourColorNameFlag(false),
      contourOn(false),
      contourOnFlag(false),

      borderColorName("yellow"),
      borderColorNameFlag(false),
      borderOn(false),
      borderOnFlag(false),
      concentrationLimitsOn(false),
      concentrationLimitsOnFlag(false),
      zoomFactor(1),
      zoomFactorFlag(false),
      screenshotFrequency(1),
      screenshotFrequencyFlag(false),
      screenUpdateFrequency(1),
      screenUpdateFrequencyFlag(false),
      noOutputFlag(false),
      noOutputFlagFlag(false),
      types3DInvisibleFlag(false)

      {
		clippingRange[0]=0.0;
		clippingRange[1]=0.0;
		focalPoint[0]=0.0;
		focalPoint[1]=0.0;
		focalPoint[2]=0.0;
		position[0]=0.0;
		position[1]=0.0;
		position[2]=0.0;
		viewUp[0]=0.0;
		viewUp[1]=0.0;
		viewUp[2]=0.0;
	  }
      ~PlayerSettings(){}
      
      unsigned int xyProj,xzProj,yzProj;

      bool xyProjFlag,xzProjFlag,yzProjFlag;

      int rotationX,rotationY, rotationZ;
      
      bool rotationXFlag,rotationYFlag,rotationZFlag;
      
      unsigned int sizeX3D,sizeY3D,sizeZ3D;

      bool sizeX3DFlag,sizeY3DFlag,sizeZ3DFlag;

	  //3D camera Data
	  float clippingRange[2];//Min Max
	  float focalPoint[3];//x,y,z
	  float position[3];//x,y,z
	  float viewUp[3]; //x,y,z

      std::string initialProjection;
      //this section deals with advanced settings - those that are usually set from the Player directly
      bool advancedSettingsOn;
      bool saveSettings;
      bool saveSettingsFlag;
      //flags tell which settings have been modified from xml level. Only those setting will be modified int the Player
      float minConcentration;
      bool minConcentrationFlag;
      bool minConcentrationFixed;
      bool minConcentrationFixedFlag;
      float maxConcentration;
      bool maxConcentrationFlag;
      bool maxConcentrationFixed;
      bool maxConcentrationFixedFlag;
      unsigned int numberOfLegendBoxes;
      bool numberOfLegendBoxesFlag;
      unsigned int numberAccuracy;
      bool numberAccuracyFlag;
      bool legendEnable;
      bool legendEnableFlag;

      float minMagnitude;
      bool minMagnitudeFlag;
      bool minMagnitudeFixed;
      bool minMagnitudeFixedFlag;
      float maxMagnitude;
      bool maxMagnitudeFlag;
      bool maxMagnitudeFixed;
      bool maxMagnitudeFixedFlag;
      float arrowLength;
      bool arrowLengthFlag;
      unsigned int numberOfLegendBoxesVector;
      bool numberOfLegendBoxesVectorFlag;
      unsigned int numberAccuracyVector;
      bool numberAccuracyVectorFlag;
      bool legendEnableVector;
      bool legendEnableVectorFlag;
      bool overlayVectorCellFields;
      bool overlayVectorCellFieldsFlag;
      bool scaleArrows;
      bool scaleArrowsFlag;
      bool fixedArrowColorFlag;      
      bool fixedArrowColorFlagFlag;

      std::string arrowColorName;
      bool arrowColorNameFlag;
      std::map<unsigned short,std::string> typeToColorNameMap;
      bool typeToColorNameMapFlag;
      std::string contourColorName;
      bool contourColorNameFlag;
      bool contourOn;
      bool contourOnFlag;

      std::string borderColorName;
      bool borderColorNameFlag;
      bool borderOn;
      bool borderOnFlag;
      bool concentrationLimitsOn;
      bool concentrationLimitsOnFlag;

      unsigned int zoomFactor;
      bool zoomFactorFlag;
      unsigned int screenshotFrequency;
      bool screenshotFrequencyFlag;
      unsigned int screenUpdateFrequency;
      bool screenUpdateFrequencyFlag;
      bool noOutputFlag;
      bool noOutputFlagFlag;

      std::vector<unsigned short> types3DInvisible;
      bool types3DInvisibleFlag;

};
};
#endif




#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// // // #include <PublicUtilities/StringUtils.h>
// // // #include <iostream>
// // // #include <algorithm>

using namespace std;

#include "PlayerSettingsPlugin.h"
using namespace CompuCell3D;


std::string PlayerSettingsPlugin::toString(){
  return "PlayerSettings";
}

std::string PlayerSettingsPlugin::steerableName(){
  return toString();
}

PlayerSettingsPlugin::PlayerSettingsPlugin():xmlData(0), playerSettingsPtr(0) {
	playerSettingsPtr=&playerSettings;
}

PlayerSettingsPlugin::~PlayerSettingsPlugin() {}


void PlayerSettingsPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData){
	xmlData=_xmlData;
}

void PlayerSettingsPlugin::extraInit(Simulator *simulator) {
	update(xmlData,true);

   Potts3D *potts=simulator->getPotts();
   Dim3D fieldDim = potts->getCellFieldG()->getDim();

     cerr<<"\n\n\\t\t\tPlayer Settings \n\n";
      cerr<<"playerSettingsPtr->xyProjFlag="<<playerSettingsPtr->xyProjFlag<<endl;
      cerr<<"playerSettingsPtr->xyProj="<<playerSettingsPtr->xyProj<<endl;
      cerr<<"This is field dim="<<fieldDim<<endl;
   ///Making sure that player settings are reasonable
   if(playerSettingsPtr->xyProjFlag){
//       cerr<<"playerSettingsPtr->xyProjFlag="<<playerSettingsPtr->xyProjFlag<<endl;
      ASSERT_OR_THROW("Value of XYProj has to be within limits for z dimension of the field",
         playerSettingsPtr->xyProj>=0 && playerSettingsPtr->xyProj<fieldDim.z
      );
//       cerr<<"playerSettingsPtr->xyProj="<<playerSettingsPtr->xyProj<<endl;
   }

   if(playerSettingsPtr->xzProjFlag){
      ASSERT_OR_THROW("Value of XZProj has to be within limits for y dimension of the field",
         playerSettingsPtr->xzProj>=0 && playerSettingsPtr->xyProj<fieldDim.y
      );
   }

   if(playerSettingsPtr->yzProjFlag){
      ASSERT_OR_THROW("Value of YZProj has to be within limits for x dimension of the field",
         playerSettingsPtr->yzProj>=0 && playerSettingsPtr->yzProj<fieldDim.x
      );
   }

   if(playerSettingsPtr->rotationXFlag){
      ASSERT_OR_THROW("Value of XRot must be between -180 and 180",
         playerSettingsPtr->rotationX>=-180 && playerSettingsPtr->rotationX<=180
      );
   }

   if(playerSettingsPtr->rotationYFlag){
      ASSERT_OR_THROW("Value of YRot must be between -180 and 180",
         playerSettingsPtr->rotationY>=-180 && playerSettingsPtr->rotationY<=180
      );
   }

      if(playerSettingsPtr->rotationZFlag){
      ASSERT_OR_THROW("Value of ZRot must be between -180 and 180",
         playerSettingsPtr->rotationZ>=-180 && playerSettingsPtr->rotationZ<=180
      );
   }

//    cerr<<"\n\n\\t\t\tPlayer Settings \n\n"; 
   
      
}

void PlayerSettingsPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	
	cerr<<"_xmlData="<<_xmlData<<endl;

	CC3DXMLElement *proj2DElement;
	proj2DElement=_xmlData->getFirstElement("Project2D");
	if(proj2DElement){

      ///reading 2D settings - projections
		if(proj2DElement->findAttribute("XYProj")){
         playerSettings.xyProj=proj2DElement->getAttributeAsUInt("XYProj");
         playerSettings.xyProjFlag=true;
		}

		if(proj2DElement->findAttribute("XZProj")){
         playerSettings.xzProj=proj2DElement->getAttributeAsUInt("XZProj");
         playerSettings.xzProjFlag=true;
		}

		if(proj2DElement->findAttribute("YZProj")){
         playerSettings.yzProj=proj2DElement->getAttributeAsUInt("YZProj");
         playerSettings.yzProjFlag=true;
		}
	}


	CC3DXMLElement *rot3DElement;
	rot3DElement=_xmlData->getFirstElement("Rotate3D");
	if(rot3DElement){

      ///reading 3D settings - rotations
		if(rot3DElement->findAttribute("XRot")){
         playerSettings.rotationX=rot3DElement->getAttributeAsUInt("XRot");
         playerSettings.rotationXFlag=true;
		}

		if(rot3DElement->findAttribute("YRot")){
         playerSettings.rotationY=rot3DElement->getAttributeAsUInt("YRot");
         playerSettings.rotationYFlag=true;
		}

		if(rot3DElement->findAttribute("ZRot")){
         playerSettings.rotationZ=rot3DElement->getAttributeAsUInt("ZRot");
         playerSettings.rotationZFlag=true;
		}

	}

	CC3DXMLElement *resize3DElement;
	resize3DElement=_xmlData->getFirstElement("Resize3D");
	if(resize3DElement){
      ///reading  3D widget size parameters used by player - the bigger the smaller object will be
      
		if(resize3DElement->findAttribute("X")){
         playerSettings.sizeX3D=resize3DElement->getAttributeAsUInt("X");
         playerSettings.sizeX3DFlag=true;
		}

		if(resize3DElement->findAttribute("Y")){
         playerSettings.sizeY3D=resize3DElement->getAttributeAsUInt("Y");
         playerSettings.sizeY3DFlag=true;
		}

		if(resize3DElement->findAttribute("Z")){
         playerSettings.sizeZ3D=resize3DElement->getAttributeAsUInt("Z");
         playerSettings.sizeZ3DFlag=true;
		}

	}
	CC3DXMLElement *view3DElement=_xmlData->getFirstElement("View3D");
	if (view3DElement){
		CC3DXMLElement *clippingRangeElement=view3DElement->getFirstElement("ClippingRange");
		if (clippingRangeElement){
			playerSettings.clippingRange[0]=clippingRangeElement->getAttributeAsDouble("Min");
			playerSettings.clippingRange[1]=clippingRangeElement->getAttributeAsDouble("Max");
		}

		CC3DXMLElement *focalPointElement=view3DElement->getFirstElement("focalPoint");
		if (focalPointElement){
			playerSettings.focalPoint[0]=focalPointElement->getAttributeAsDouble("x");
			playerSettings.focalPoint[1]=focalPointElement->getAttributeAsDouble("y");
			playerSettings.focalPoint[1]=focalPointElement->getAttributeAsDouble("z");
		}

		CC3DXMLElement *positionElement=view3DElement->getFirstElement("position");
		if (positionElement){
			playerSettings.position[0]=positionElement->getAttributeAsDouble("x");
			playerSettings.position[1]=positionElement->getAttributeAsDouble("y");
			playerSettings.position[1]=positionElement->getAttributeAsDouble("z");
		}

		CC3DXMLElement *viewUpElement=view3DElement->getFirstElement("viewUp");
		if (viewUpElement){
			playerSettings.viewUp[0]=viewUpElement->getAttributeAsDouble("x");
			playerSettings.viewUp[1]=viewUpElement->getAttributeAsDouble("y");
			playerSettings.viewUp[1]=viewUpElement->getAttributeAsDouble("z");
		}

	}

	CC3DXMLElement *initProjXMLElement=_xmlData->getFirstElement("InitialProjection");
	if(initProjXMLElement){
		if(initProjXMLElement->findAttribute("Projection")){
         playerSettings.initialProjection=initProjXMLElement->getAttribute("Projection");
         //changing to lowercase 
         changeToLower(playerSettings.initialProjection);
         ASSERT_OR_THROW("InitialProjection has have a value of xy or xz or yz",
         playerSettings.initialProjection=="xy" || playerSettings.initialProjection=="xz" || playerSettings.initialProjection=="yz"
			);
		}

	}

	CC3DXMLElement *concXMLElement;
	concXMLElement=_xmlData->getFirstElement("Concentration");
	if(concXMLElement){
      ///reading  3D widget size parameters used by player - the bigger the smaller object will be
      
		playerSettings.advancedSettingsOn=true;

		if(concXMLElement->findAttribute("Min")){
         playerSettings.minConcentration=concXMLElement->getAttributeAsDouble("Min");
         playerSettings.minConcentrationFlag=true;

         playerSettings.minConcentrationFixed=true;
         playerSettings.minConcentrationFixedFlag=true;

		}

		if(concXMLElement->findAttribute("Max")){
         playerSettings.maxConcentration=concXMLElement->getAttributeAsDouble("Max");
         playerSettings.maxConcentrationFlag=true;
         playerSettings.maxConcentrationFixed=true;
         playerSettings.maxConcentrationFixedFlag=true;

		}


		if(concXMLElement->findAttribute("MinFixed")){
         playerSettings.minConcentrationFixed=concXMLElement->getAttributeAsBool("MinFixed");
         playerSettings.minConcentrationFixedFlag=true;
		}

		if(concXMLElement->findAttribute("MaxFixed")){
         playerSettings.maxConcentrationFixed=concXMLElement->getAttributeAsBool("MaxFixed");
         playerSettings.maxConcentrationFixedFlag=true;
		}

		if(concXMLElement->findAttribute("LegendEnable")){
         playerSettings.legendEnable=concXMLElement->getAttributeAsBool("LegendEnable");
         playerSettings.legendEnableFlag=true;
		}

		if(concXMLElement->findAttribute("NumberOfLegendBoxes")){
         playerSettings.numberOfLegendBoxes=concXMLElement->getAttributeAsUInt("NumberOfLegendBoxes");
         playerSettings.numberOfLegendBoxesFlag=true;
		}

		if(concXMLElement->findAttribute("NumberAccuracy")){
         playerSettings.numberAccuracy=concXMLElement->getAttributeAsUInt("NumberAccuracy");
         playerSettings.numberAccuracyFlag=true;
		}

		if(concXMLElement->findAttribute("ConcentrationLimitsOn")){
         playerSettings.concentrationLimitsOn=concXMLElement->getAttributeAsBool("ConcentrationLimitsOn");
         playerSettings.concentrationLimitsOnFlag=true;
		}


	}


	CC3DXMLElement *magnitudeXMLElement;
	magnitudeXMLElement=_xmlData->getFirstElement("Magnitude");
	if(magnitudeXMLElement){
      ///reading  3D widget size parameters used by player - the bigger the smaller object will be
      
		playerSettings.advancedSettingsOn=true;

		if(magnitudeXMLElement->findAttribute("Min")){
         playerSettings.minMagnitude=magnitudeXMLElement->getAttributeAsDouble("Min");
         playerSettings.minMagnitudeFlag=true;

         playerSettings.minMagnitudeFixed=true;
         playerSettings.minMagnitudeFixedFlag=true;			
		}

		if(magnitudeXMLElement->findAttribute("Max")){
         playerSettings.maxMagnitude=magnitudeXMLElement->getAttributeAsDouble("Max");
         playerSettings.maxMagnitudeFlag=true;

         playerSettings.maxMagnitudeFixed=true;
         playerSettings.maxMagnitudeFixedFlag=true;			
		}


		if(magnitudeXMLElement->findAttribute("MinFixed")){
         playerSettings.minMagnitudeFixed=magnitudeXMLElement->getAttributeAsBool("MinFixed");
         playerSettings.minMagnitudeFixedFlag=true;
		}

		if(magnitudeXMLElement->findAttribute("MaxFixed")){
         playerSettings.maxMagnitudeFixed=magnitudeXMLElement->getAttributeAsBool("MaxFixed");
         playerSettings.maxMagnitudeFixedFlag=true;
		}




		if(magnitudeXMLElement->findAttribute("LegendEnable")){
         playerSettings.legendEnableVector=magnitudeXMLElement->getAttributeAsBool("LegendEnable");
         playerSettings.legendEnableVectorFlag=true;			
		}


		if(magnitudeXMLElement->findAttribute("NumberOfLegendBoxes")){
			playerSettings.numberOfLegendBoxesVector=magnitudeXMLElement->getAttributeAsUInt("NumberOfLegendBoxes");
         playerSettings.numberOfLegendBoxesVectorFlag=true;
		}

		if(magnitudeXMLElement->findAttribute("NumberAccuracy")){
         playerSettings.numberAccuracyVector=magnitudeXMLElement->getAttributeAsUInt("NumberAccuracy");
         playerSettings.numberAccuracyVectorFlag=true;
		}

		if(magnitudeXMLElement->findAttribute("OverlayVectorAndCellFields")){
         playerSettings.overlayVectorCellFields=magnitudeXMLElement->getAttributeAsBool("OverlayVectorAndCellFields");
         playerSettings.overlayVectorCellFieldsFlag=true;
		}

		if(magnitudeXMLElement->findAttribute("ScaleArrows")){
         playerSettings.scaleArrows=magnitudeXMLElement->getAttributeAsBool("ScaleArrows");
         playerSettings.scaleArrowsFlag=true;
		}

		if(magnitudeXMLElement->findAttribute("FixedArrowColorFlag")){
         playerSettings.fixedArrowColorFlag=magnitudeXMLElement->getAttributeAsBool("FixedArrowColorFlag");
         playerSettings.fixedArrowColorFlagFlag=true;
		}

		if(magnitudeXMLElement->findAttribute("ArrowColor")){
         playerSettings.arrowColorName=magnitudeXMLElement->getAttribute("ArrowColor");
         changeToLower(playerSettings.arrowColorName);
         playerSettings.arrowColorNameFlag=true;
		}

	}

	CC3DXMLElement *borderXMLElement=_xmlData->getFirstElement("Border");
	if(borderXMLElement){
		
		playerSettings.advancedSettingsOn=true;      

		if(borderXMLElement->findAttribute("Color")){
         playerSettings.borderColorName=borderXMLElement->getAttribute("Color");
         //changing to lowercase 
         changeToLower(playerSettings.borderColorName);
         playerSettings.borderColorNameFlag=true;
		}
		if(borderXMLElement->findAttribute("BorderOn")){
         playerSettings.borderOn=borderXMLElement->getAttributeAsBool("BorderOn");
         playerSettings.borderOnFlag=true;
		}
	}

	CC3DXMLElement *contourXMLElement=_xmlData->getFirstElement("Contour");
	if(contourXMLElement){
		
		playerSettings.advancedSettingsOn=true;      

		if(contourXMLElement->findAttribute("Color")){

         playerSettings.contourColorName=contourXMLElement->getAttribute("Color");
         
         //changing to lowercase 
         changeToLower(playerSettings.contourColorName);
         playerSettings.contourColorNameFlag=true;
		}

		if(contourXMLElement->findAttribute("ContourOn")){
         playerSettings.contourOn=contourXMLElement->getAttributeAsBool("ContourOn");
         playerSettings.contourOnFlag=true;
		}
	}

	CC3DXMLElementList cellColorVecXML=_xmlData->getElements("Cell");
	for(int i = 0 ; i < cellColorVecXML.size() ; ++i ){
		playerSettings.advancedSettingsOn=true;

      std::string color;
      unsigned short type;

      type=(unsigned short)cellColorVecXML[i]->getAttributeAsUInt("Type");
      color=cellColorVecXML[i]->getAttributeAsUInt("Color");

      playerSettings.typeToColorNameMap[type]=color;
      playerSettings.typeToColorNameMapFlag=true;

	}

	CC3DXMLElement *visContrXMLElement=_xmlData->getFirstElement("VisualControl");
	if(visContrXMLElement){
      
		playerSettings.advancedSettingsOn=true;
		
		if(visContrXMLElement->findAttribute("ZoomFactor")){
         playerSettings.zoomFactor=visContrXMLElement->getAttributeAsUInt("ZoomFactor");
         playerSettings.zoomFactorFlag=true;
		}

		if(visContrXMLElement->findAttribute("ScreenshotFrequency")){
         playerSettings.screenshotFrequency=visContrXMLElement->getAttributeAsUInt("ScreenshotFrequency");
         playerSettings.screenshotFrequencyFlag=true;
		}

		if(visContrXMLElement->findAttribute("ScreenUpdateFrequency")){
         playerSettings.screenUpdateFrequency=visContrXMLElement->getAttributeAsUInt("ScreenUpdateFrequency");
         playerSettings.screenUpdateFrequencyFlag=true;
		}

		if(visContrXMLElement->findAttribute("NoOutput")){
         playerSettings.noOutputFlag=visContrXMLElement->getAttributeAsBool("NoOutput");
         playerSettings.noOutputFlagFlag=true;
		}
	}


	CC3DXMLElement *types3DInvisElement=_xmlData->getFirstElement("TypesInvisibleIn3D");
	if(types3DInvisElement){
		
      playerSettings.advancedSettingsOn=true;

      std::string typesInvisibleIn3DString=types3DInvisElement->getAttribute("Types");
      vector<string> typesInvisiblein3DVector; 
      parseStringIntoList(typesInvisibleIn3DString,typesInvisiblein3DVector,",");

      for(unsigned int i = 0 ; i < typesInvisiblein3DVector.size() ; ++i){
         playerSettings.types3DInvisible.push_back((unsigned short)BasicString::parseUInteger(typesInvisiblein3DVector[i]));
      }

      playerSettings.types3DInvisibleFlag=true;

	}

	CC3DXMLElement *settingsXMLElement=_xmlData->getFirstElement("Settings");
	if(settingsXMLElement){
		if(settingsXMLElement->findAttribute("SaveSettings")){
         playerSettings.saveSettings=settingsXMLElement->getAttributeAsBool("SaveSettings");
         playerSettings.saveSettingsFlag=true;

		}
	}

}





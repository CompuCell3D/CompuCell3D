

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <PublicUtilities/StringUtils.h>
// // // #include <algorithm>
// // // #include <string>
#include <limits>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>




#include "NeighborStickPlugin.h"




NeighborStickPlugin::NeighborStickPlugin() : thresh(0),xmlData(0) {
}

NeighborStickPlugin::~NeighborStickPlugin() {
  
}

void NeighborStickPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;
  potts=simulator->getPotts();
  simulator->getPotts()->registerEnergyFunctionWithName(this,"NeighborStick");
  simulator->registerSteerableObject(this);

   bool pluginAlreadyRegisteredFlag;
   NeighborTrackerPlugin * neighborTrackerPluginPtr=(NeighborTrackerPlugin*)(Simulator::pluginManager.get("NeighborTracker",&pluginAlreadyRegisteredFlag));
   if (!pluginAlreadyRegisteredFlag){
      neighborTrackerPluginPtr->init(simulator);
      ASSERT_OR_THROW("NeighborTracker plugin not initialized!", neighborTrackerPluginPtr);
      neighborTrackerAccessorPtr=neighborTrackerPluginPtr->getNeighborTrackerAccessorPtr();
      ASSERT_OR_THROW("neighborAccessorPtr  not initialized!", neighborTrackerAccessorPtr);
   }
}

void NeighborStickPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}

void NeighborStickPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
   
   automaton = potts->getAutomaton();
   set<unsigned char> cellTypesSet;

	
	thresh=_xmlData->getFirstElement("Threshold")->getDouble();
	typeNamesString=_xmlData->getFirstElement("Types")->getText();
	parseStringIntoList(typeNamesString , typeNames , ",");
	cerr<<"NEIGHBOR STICK typeNamesString="<<typeNamesString<<endl;

   std::vector<std::string> temp;

   
   for(int i = 0; i < typeNames.size(); i++) {
      temp.push_back(typeNames[i]); 
		cerr<<"typeNames[i]="<<typeNames[i]<<endl;
   }
   //typeNames.clear();
   //typeNames=temp;
   //typeNames.pop_back(); //delete empyt element
   for(int i = 0; i < typeNames.size(); i++) {
      idNames.push_back(automaton->getTypeId(typeNames[i])); 

		cerr<<"adding type ID = "<<automaton->getTypeId(typeNames[i])<<endl;
   }   



}

double NeighborStickPlugin::changeEnergy(const Point3D &pt,
                                  const CellG *newCell,
                                  const CellG *oldCell) {
//    cerr<<"ChangeEnergy"<<endl;
   
   
  double energy = 0;
  unsigned int token = 0;
  double distance = 0;
  Point3D n;
  int totalArea = 0;
  CellG *nCell=0;
  WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
  Neighbor neighbor;
  
  std::set<NeighborSurfaceData> * neighborData;
  std::set<NeighborSurfaceData >::iterator sitr;
  
//   cerr << "Threshold: " << thresh << endl;
//   cerr << "  SIZE: " << typeNames.size() << endl;
//   for(int i = 0; i < typeNames.size(); i++) {
//      cerr << "Name: " << typeNames[i] << " ID: " << (int)automaton->getTypeId(typeNames[i]) << " and again: " << idNames[i] << endl;
//   }
  
  
  
     

   if(oldCell) {
      neighborData = &(neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr)->cellNeighbors);
      
      for(sitr=neighborData->begin() ; sitr != neighborData->end() ; ++sitr){
         
          //cerr << "Type: " << (int)oldCell->type << " ID: " << oldCell->id << endl;
         nCell= sitr->neighborAddress;
         if(nCell){
            int nType = (int)nCell->type;
//             cerr << "Neighbor Type: " << nType << endl;
            for(int i = 0; i < idNames.size() ; i++) {
               if(nType==idNames[i]) {
                  //cerr << "SAME TYPE step: " << i << endl;
                  //cerr << "Common Surface Area: " << (int)sitr->commonSurfaceArea << endl;
                  
                  totalArea +=(int)sitr->commonSurfaceArea;
                  break;
               }
//               else {
//                  continue;
////                   cerr << "Wrong Type" << endl;
//               }
            }
         }
      
      }
   }
	//cerr<<"totalArea="<<totalArea<<" thresh="<<thresh<<endl;
   if((totalArea < thresh) & (totalArea != 0)){
		return (std::numeric_limits<float>::max)()/2.0;
   }
   else {
      return 0;
   }
}




std::string NeighborStickPlugin::toString(){

   return "NeighborStick";

}

std::string NeighborStickPlugin::steerableName(){

   return toString();

}



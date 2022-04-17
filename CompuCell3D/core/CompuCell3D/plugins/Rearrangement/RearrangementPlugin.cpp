


#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>

// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
#include <NeighborTracker/NeighborTrackerPlugin.h>
// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>

using namespace CompuCell3D;


using namespace std;


#include "RearrangementPlugin.h"

RearrangementPlugin::RearrangementPlugin() : 
potts(0),
   fRearrangement(0.0),
   lambdaRearrangement(0.0),
   percentageLossThreshold(1.0),
   defaultPenalty(0.0),
   cellFieldG(0)

{}

RearrangementPlugin::~RearrangementPlugin() {}

void RearrangementPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
  potts = simulator->getPotts();
  potts->registerEnergyFunctionWithName(this,toString());
  simulator->registerSteerableObject(this);

  boundaryStrategy=BoundaryStrategy::getInstance();
  maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);// 1st nearest neighbor
  cellFieldG=(WatchableField3D<CellG *> *)potts->getCellFieldG();

  bool pluginAlreadyRegisteredFlag;
  NeighborTrackerPlugin * nTracker =(NeighborTrackerPlugin *) Simulator::pluginManager.get("NeighborTracker",&pluginAlreadyRegisteredFlag); //this will load NeighborTracker plugin if it si not already loaded 
   if(!pluginAlreadyRegisteredFlag)
      nTracker->init(simulator,0);
  neighborTrackerAccessorPtr = nTracker->getNeighborTrackerAccessorPtr();


}




void RearrangementPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	
	fRearrangement=_xmlData->getFirstElement("FRearrangement")->getDouble();
   lambdaRearrangement=_xmlData->getFirstElement("LambdaRearrangement")->getDouble();
   percentageLossThreshold=_xmlData->getFirstElement("PercentageLossThreshold")->getDouble();
   defaultPenalty=_xmlData->getFirstElement("DefaultPenalty")->getDouble();
}

pair<CellG*,CellG*> RearrangementPlugin::preparePair(CellG* cell1, CellG* cell2){

   if(cell1<cell2)
      return make_pair(cell1,cell2);
   else
      return make_pair(cell2,cell1);


}


double RearrangementPlugin::changeEnergy(const Point3D &pt,
				  const CellG *newCell,
				  const CellG *oldCell) {

  float energy=0.0;
  CellG *nCell=0;
  Neighbor neighbor;
  multiset<std::pair<CellG*,CellG*> > oldSet;
  multiset<std::pair<CellG*,CellG*> > newSet;
  set<NeighborSurfaceData> * nsdSetPtr;
  set<NeighborSurfaceData>::iterator nsdSitr;

   for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
      neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
      if(!neighbor.distance){
      //if distance is 0 then the neighbor returned is invalid
      continue;
      }
      nCell = cellFieldG->get(neighbor.pt);
         if(newCell!=nCell && (newCell!=0 && nCell!=0)){
            newSet.insert(preparePair(nCell,const_cast<CellG*>(newCell)));
         }

         if(oldCell!=nCell && (oldCell!=0 && nCell!=0) ){
            oldSet.insert(preparePair(nCell,const_cast<CellG*>(oldCell)));
         }

//       if (newCell == nCell) newDiff--;
//       else newDiff++;
//    
//       if (oldCell == nCell) oldDiff++;
//       else oldDiff--;
   }

   pair<CellG*,CellG*> lastPair;
   lastPair=preparePair(0,0);
   int diff=0;
   short commonSufraceContactArea;
   float percentageSurfaceLoss;
    
   for (multiset<std::pair<CellG*,CellG*> >::iterator sitr=oldSet.begin() ; sitr!= oldSet.end() ; ++sitr){
      if(lastPair!=*sitr){
         lastPair=*sitr;
         //here I am penalizing for lost cell-cell contact
         diff=newSet.count(*sitr)-oldSet.count(*sitr);

         

//          cerr<<"diff="<<diff<<endl;
//          if(diff==-4 || diff==4){
//             cerr<<"sitr="<<sitr->first<<" , "<<sitr->second<<endl;
//             cerr<<" newSet.count(*sitr)="<<newSet.count(*sitr)<<" oldSet.count(*sitr)="<<oldSet.count(*sitr)<<endl;
//             cerr<<"oldCell="<<oldCell<<" newCell="<<newCell<<endl;
//             for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
//                neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
//                if(!neighbor.distance){
//                //if distance is 0 then the neighbor returned is invalid
//                continue;
//                }
//                nCell = cellFieldG->get(neighbor.pt);
//                cerr<<"nCell="<<nCell<<endl;         
//             }
// 
// 
// 
//          }
         if(diff<0){

            nsdSetPtr=&(neighborTrackerAccessorPtr->get(sitr->first->extraAttribPtr)->cellNeighbors);
            nsdSitr=nsdSetPtr->find(NeighborSurfaceData(sitr->second));
            if(nsdSitr != nsdSetPtr->end()){
               commonSufraceContactArea=nsdSitr->commonSurfaceArea;
               percentageSurfaceLoss=-diff/float(commonSufraceContactArea);
//                cerr<<"commonSufraceContactArea="<<commonSufraceContactArea<<" percentageSurfaceLoss="<<percentageSurfaceLoss<<endl;
//                if(percentageSurfaceLoss>0.95){
//                   cerr<<"LOSING CONTACT"<<endl;
//                   cerr<<"commonSufraceContactArea="<<commonSufraceContactArea<<" percentageSurfaceLoss="<<percentageSurfaceLoss<<endl;

//                }
            }else{
               cerr<<" THIS IS THE ERROR: COULD NOT FIND REQUESTED NEIGHBOR"<<endl;
               exit(0);
            }            
            if (percentageSurfaceLoss>=percentageLossThreshold){
               energy+=defaultPenalty;
//                cerr<<"Energy="<<energy<<endl;
            }else{
               energy+=exp(-percentageSurfaceLoss)*lambdaRearrangement;
            }
//             energy+=-diff*lambdaRearrangement;
         }
      }
   }
  
//   cerr<<"energy="<<energy<<endl;
  if(newCell){
//       energy+=newCell->surface/(float)newCell->volume*fRearrangement;
//       cerr<<"newCell->surface/newCell->volume*fRearrangement="<<newCell->surface/(float)newCell->volume*fRearrangement<<endl;
  }
  if(oldCell){
//       energy+=oldCell->surface/(float)oldCell->volume*fRearrangement;
//       cerr<<"oldCell->surface/oldCell->volume*fRearrangement="<<oldCell->surface/(float)oldCell->volume*fRearrangement<<endl;
  }

  return energy;

}


std::string RearrangementPlugin::toString(){
   return "Rearrangement";
}

std::string RearrangementPlugin::steerableName(){
   return toString();
}
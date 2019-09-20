#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include "ContactOrientationPlugin.h"


ContactOrientationPlugin::ContactOrientationPlugin():
pUtils(0),
lockPtr(0),
xmlData(0) ,
angularTermDefined(false),
cellFieldG(0),
boundaryStrategy(0),
automaton(0),
angularTermFcnPtr(&ContactOrientationPlugin::singleTermFormula)
{}

ContactOrientationPlugin::~ContactOrientationPlugin() {
    //pUtils->destroyLock(lockPtr);
    //delete lockPtr;
    //lockPtr=0;
}

void ContactOrientationPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData=_xmlData;
    sim=simulator;
    potts=simulator->getPotts();
    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    fieldDim=cellFieldG->getDim();
    pUtils=sim->getParallelUtils();
    //lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
    //pUtils->initLock(lockPtr); 
    
   update(xmlData,true);
   

    potts->getCellFactoryGroupPtr()->registerClass(&contactOrientationDataAccessor);
    potts->registerEnergyFunctionWithName(this,"ContactOrientation");
        
    
    
    simulator->registerSteerableObject(this);
}

void ContactOrientationPlugin::extraInit(Simulator *simulator){
    
}


double ContactOrientationPlugin::singleTermFormula(double _alpha,double _theta){
	return _alpha*fabs(cos(_theta));
}


double ContactOrientationPlugin::angularTermFunction(double _alpha,double _theta){

		int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
		ExpressionEvaluator & ev=eed[currentWorkNodeNumber];
		double angularTerm=0.0;


		ev[0]=_alpha;
		ev[1]=_theta;	
		angularTerm=ev.eval();


		return angularTerm;
}

double ContactOrientationPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {	

	
    double energy = 0;
    
    unsigned int token = 0;
    double distance = 0;
    Point3D n;

    CellG *nCell=0;
    WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
    Neighbor neighbor;
    
   //precalculating COMs before and after flip
   Coordinates3D<double> centroidOldAfter;
   Coordinates3D<double> centroidNewAfter;
       
   Vector3 comOldAfter;
   Vector3 comNewAfter;
   Vector3 comOldBefore;
   Vector3 comNewBefore;


   if(oldCell){
      comOldBefore=Vector3(oldCell->xCOM,oldCell->yCOM,oldCell->zCOM);
       
      if(oldCell->volume>1){
         centroidOldAfter=precalculateCentroid(pt, oldCell, -1,fieldDim,boundaryStrategy);
         comOldAfter=Vector3(centroidOldAfter.X()/(float)(oldCell->volume-1), centroidOldAfter.Y()/(float)(oldCell->volume-1),centroidOldAfter.Z()/(float)(oldCell->volume-1));             
      }else{
         comOldAfter=Vector3(oldCell->xCOM,oldCell->yCOM,oldCell->zCOM);
      }

   }

   if(newCell){
      comNewBefore=Vector3(newCell->xCOM,newCell->yCOM,newCell->zCOM);
      centroidNewAfter=precalculateCentroid(pt, newCell, 1,fieldDim,boundaryStrategy);
      comNewAfter=Vector3(centroidNewAfter.X()/(float)(newCell->volume+1),centroidNewAfter.Y()/(float)(newCell->volume+1),centroidNewAfter.Z()/(float)(newCell->volume+1));
   }

    
    
    
    for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
        neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
        if(!neighbor.distance){
            //if distance is 0 then the neighbor returned is invalid
            continue;
        }

        nCell = fieldG->get(neighbor.pt);
        Vector3 ptVec(pt.x,pt.y,pt.z);
        
        if(nCell!=oldCell){
            
            
            
            if((nCell != 0) && (oldCell != 0)) {
                double termN=0.0;
                double termOld=0.0;
                    
                Vector3 comNBefore(nCell->xCOM,nCell->yCOM,nCell->zCOM);
                Vector3 oldCellPolVector=getOriantationVector(oldCell);
                Vector3 nCellPolVector=getOriantationVector(nCell);
                double alphaOld=getAlpha(oldCell);
                double alphaN=getAlpha(nCell);                        
                Vector3 oldDistVec=ptVec-comOldBefore;
                Vector3 nDistVec=ptVec-comNBefore;
                
                double thetaOld=oldDistVec.Angle(oldCellPolVector);
                double thetaN=nDistVec.Angle(nCellPolVector);                        
                    
                if (oldCell->volume>1){
                    //termOld=alphaOld*fabs(cos(thetaOld));
					//termOld=alphaOld*cos(thetaOld);
					// termOld=singleTermFormula(alphaOld,thetaOld);
                    termOld=(this->*angularTermFcnPtr)(alphaOld,thetaOld);
                }else{
                    termOld=0.0;
                }                
                
                //termN=alphaN*fabs(cos(thetaN));
                //termN=alphaN*cos(thetaN);
				// termN=singleTermFormula(alphaN,thetaN);
                termN=(this->*angularTermFcnPtr)(alphaN,thetaN);

                if((nCell->clusterId) != (oldCell->clusterId)) {
                    
                    energy-=termOld+termN;    
        
                }
            }else{
     //           double nonMediumTerm=0.0;
				
				

     //           if (oldCell){
     //               double termOld=0.0;
     //               Vector3 oldCellPolVector=getOriantationVector(oldCell);                   
     //               double alphaOld=getAlpha(oldCell);                    
     //               Vector3 oldDistVec=ptVec-comOldBefore;
     //               double thetaOld=oldDistVec.Angle(oldCellPolVector);
     //                                       
     //               if (oldCell->volume>1){
     //                   //termOld=alphaOld*fabs(cos(thetaOld));
					//	//termOld=alphaOld*cos(thetaOld);
					//	termOld=singleTermFormula(alphaOld,thetaOld);
     //               }else{
     //                   termOld=0.0;
     //               }             
     //               
     //               nonMediumTerm=termOld;
     //           }else{
     //               
     //               double termN=0.0;    
     //               Vector3 comNBefore(nCell->xCOM,nCell->yCOM,nCell->zCOM);
     //               Vector3 nCellPolVector=getOriantationVector(nCell);
     //               double alphaN=getAlpha(nCell);                        
     //               Vector3 nDistVec=ptVec-comNBefore;                    

     //               double thetaN=nDistVec.Angle(nCellPolVector);                                                
     //               
     //               //termN=alphaN*fabs(cos(thetaN)); 
					////termN=alphaN*cos(thetaN); 
					//termN=singleTermFormula(alphaN,thetaN); 
     //               
     //               nonMediumTerm=termN;               
     //           }    
     //           energy -= nonMediumTerm;
           }
                
        }
		
        //watch for case nCell=oldCell - use oldCellCOMAfter for calculations
        if(nCell!=newCell){

            if((newCell != 0) && (nCell != 0)) {


                double termN=0.0;
                double termNew=0.0;
                Vector3 comNAfter;
				//notice, that if oldCell is about to disappear - it has Vol=1 before spin flip
				//then it will never be picked as a after-flip neighbor of the new cell and consequently 
				// statement : comNAfter=comOldAfter; will never be called so there is no issue what COM of oldCell after it disappears should be
				if (nCell==oldCell){
					comNAfter=comOldAfter;
				}else{
					comNAfter=Vector3(nCell->xCOM,nCell->yCOM,nCell->zCOM);
				}
                
                Vector3 newCellPolVector=getOriantationVector(newCell);
                Vector3 nCellPolVector=getOriantationVector(nCell);
                double alphaNew=getAlpha(newCell);
                double alphaN=getAlpha(nCell);                        

                Vector3 newDistVec=ptVec-comNewAfter;
                Vector3 nDistVec=ptVec-comNAfter;
                
                double thetaNew=newDistVec.Angle(newCellPolVector);
                double thetaN=nDistVec.Angle(nCellPolVector);                        
                
				//termNew=alphaNew*fabs(cos(thetaNew));
				//termNew=alphaNew*cos(thetaNew);
				// termNew=singleTermFormula(alphaNew,thetaNew);
                termNew=(this->*angularTermFcnPtr)(alphaNew,thetaNew);
                
                //termN=alphaN*fabs(cos(thetaN));
                //termN=alphaN*cos(thetaN);
				// termN=singleTermFormula(alphaN,thetaN);
                termN=(this->*angularTermFcnPtr)(alphaN,thetaN);

                if((nCell->clusterId) != (newCell->clusterId)) {
                    
                    energy+=termNew+termN;    
       
                }

            }
            else{
     //           double nonMediumTerm=0.0;
					//			

     //           if (newCell){
     //               double termNew=0.0;
     //               Vector3 newCellPolVector=getOriantationVector(newCell);                   
     //               double alphaNew=getAlpha(newCell);                    
     //               Vector3 newDistVec=ptVec-comNewAfter;
     //               double thetaNew=newDistVec.Angle(newCellPolVector);

					////termNew=alphaNew*fabs(cos(thetaNew));                                                                
					////termNew=alphaNew*cos(thetaNew);                                                                
					//termNew=singleTermFormula(alphaNew,thetaNew);
     //               nonMediumTerm=termNew;

     //           }else{
     //               
     //               double termN=0.0;    

					//Vector3 comNAfter;
					//if (nCell==oldCell){
					//	comNAfter=comOldAfter;
					//}else{
					//	comNAfter=Vector3(nCell->xCOM,nCell->yCOM,nCell->zCOM);
					//}

     //               Vector3 nCellPolVector=getOriantationVector(nCell);
     //               double alphaN=getAlpha(nCell);                        
     //               Vector3 nDistVec=ptVec-comNAfter;                    

     //               double thetaN=nDistVec.Angle(nCellPolVector);                                                
     //               
     //               //termN=alphaN*fabs(cos(thetaN)); 
     //               //termN=alphaN*cos(thetaN); 
					//termN=singleTermFormula(alphaN,thetaN); 

     //               nonMediumTerm=termN;               
     //           }    
     //           energy += nonMediumTerm;               
            }
        }
    }


    //cerr<<"pt="<<pt<<" energy="<<energy<<endl;
    //cerr<<"energy="<<energy<<endl;
    return energy;
}            


void ContactOrientationPlugin::handleEvent(CC3DEvent & _event){
	if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){
    
        update(xmlData);
	}
}


void ContactOrientationPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

    angularTermDefined=false;
    
    
	if (_xmlData->findElement("AngularTerm")){
		unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
		eed.allocateSize(maxNumberOfWorkNodes);
		vector<string> variableNames;
		variableNames.push_back("Alpha");
		variableNames.push_back("Theta");

		eed.addVariables(variableNames.begin(),variableNames.end());
		eed.update(_xmlData->getFirstElement("AngularTerm"));			
		angularTermDefined=true;
        angularTermFcnPtr=&ContactOrientationPlugin::angularTermFunction;
	}else{
		angularTermDefined=false;
        angularTermFcnPtr=&ContactOrientationPlugin::singleTermFormula;
	}
    
    

    //Here I initialize max neighbor index for direct acces to the list of neighbors 
    boundaryStrategy=BoundaryStrategy::getInstance();
    maxNeighborIndex=0;

    
    
    if(_xmlData->getFirstElement("Depth")){
        maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
            //cerr<<"got here will do depth"<<endl;
    }else{
        //cerr<<"got here will do neighbor order"<<endl;
        if(_xmlData->getFirstElement("NeighborOrder")){

            maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());	
        }else{
            maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

        }

    }

    cerr<<"Contact maxNeighborIndex="<<maxNeighborIndex<<endl;


	return ;
    set<unsigned char> cellTypesSet;
    contactEnergies.clear();
    
    //if(potts->getDisplayUnitsFlag()){
    //    Unit contactEnergyUnit=potts->getEnergyUnit()/powerUnit(potts->getLengthUnit(),2);
    //    CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
    //    if (!unitsElem){ //add Units element
    //            unitsElem=_xmlData->attachElement("Units");
    //    }

    //    if(unitsElem->getFirstElement("EnergyUnit")){
    //            unitsElem->getFirstElement("EnergyUnit")->updateElementValue(contactEnergyUnit.toString());
    //    }else{
    //            CC3DXMLElement * energyUnitElem = unitsElem->attachElement("EnergyUnit",contactEnergyUnit.toString());
    //    }
    //}
    
    CC3DXMLElementList energyVec=_xmlData->getElements("Energy");
    
    for (int i = 0 ; i<energyVec.size(); ++i){
            setContactEnergy(energyVec[i]->getAttribute("Type1"), energyVec[i]->getAttribute("Type2"), energyVec[i]->getDouble());
            //inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
            cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type1")));
            cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type2")));
    }
    
    //Now that we know all the types used in the simulation we will find size of the contactEnergyArray
    vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector
    
    int size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
    size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated
    
    int index ;
    contactEnergyArray.clear();
    contactEnergyArray.assign(size,vector<double>(size,0.0));
    
    for(int i = 0 ; i < size ; ++i)
        for(int j = 0 ; j < size ; ++j){
            index = getIndex(cellTypesVector[i],cellTypesVector[j]);
            contactEnergyArray[i][j] = contactEnergies[index];
        }
        
    cerr<<"size="<<size<<endl;
    for(int i = 0 ; i < size ; ++i)
        for(int j = 0 ; j < size ; ++j){
                cerr<<"contact["<<i<<"]["<<j<<"]="<<contactEnergyArray[i][j]<<endl;
        }

    
}

double ContactOrientationPlugin::contactEnergy(const CellG *cell1, const CellG *cell2) {
	return contactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];
}

void ContactOrientationPlugin::setContactEnergy(const string typeName1,const string typeName2, const double energy){
    char type1 = automaton->getTypeId(typeName1);
    char type2 = automaton->getTypeId(typeName2);

    int index = getIndex(type1, type2);

    contactEnergies_t::iterator it = contactEnergies.find(index);
    ASSERT_OR_THROW(string("Contact energy for ") + typeName1 + " " + typeName2 +
            " already set!", it == contactEnergies.end());

    contactEnergies[index] = energy;
}


int ContactOrientationPlugin::getIndex(const int type1, const int type2) const {
	if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
	else return ((type2 + 1) | ((type1 + 1) << 16));
}


void ContactOrientationPlugin::setOriantationVector(CellG * _cell,double _x, double _y, double _z){

    contactOrientationDataAccessor.get(_cell->extraAttribPtr)->oriantationVec=Vector3(_x,_y,_z);
    
}


Vector3 ContactOrientationPlugin::getOriantationVector(const CellG * _cell){
    return contactOrientationDataAccessor.get(_cell->extraAttribPtr)->oriantationVec;
}

void ContactOrientationPlugin::setAlpha(CellG * _cell,double _alpha){
    contactOrientationDataAccessor.get(_cell->extraAttribPtr)->alpha=_alpha;
}
double ContactOrientationPlugin::getAlpha(const CellG * _cell){
    return contactOrientationDataAccessor.get(_cell->extraAttribPtr)->alpha;
}


std::string ContactOrientationPlugin::toString(){
    return "ContactOrientation";
}


std::string ContactOrientationPlugin::steerableName(){
    return toString();
}

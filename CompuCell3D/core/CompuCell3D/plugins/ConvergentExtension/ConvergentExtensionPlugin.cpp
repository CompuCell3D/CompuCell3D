/*************************************************************************
*    CompuCell - A software framework for multimodel simulations of     *
* biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
*                             Indiana                                   *
*                                                                       *
* This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
*  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
*  CompuCell GNU General Public License RIDER you can redistribute it   *
* and/or modify it under the terms of the GNU General Public License as *
*  published by the Free Software Foundation; either version 2 of the   *
*         License, or (at your option) any later version.               *
*                                                                       *
* This program is distributed in the hope that it will be useful, but   *
*      WITHOUT ANY WARRANTY; without even the implied warranty of       *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
*             General Public License for more details.                  *
*                                                                       *
*  You should have received a copy of the GNU General Public License    *
*     along with this program; if not, write to the Free Software       *
*      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
*************************************************************************/

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>


// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// // // #include <iostream>
// // // #include <algorithm>

using namespace std;

#include "ConvergentExtensionPlugin.h"
#include<core/CompuCell3D/CC3DLogger.h>

#define sign(x) (((x>0)-(x<0)))


ConvergentExtensionPlugin::ConvergentExtensionPlugin():xmlData(0)   {
}

ConvergentExtensionPlugin::~ConvergentExtensionPlugin() {

}

void ConvergentExtensionPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	potts=simulator->getPotts();
	xmlData=_xmlData;
	simulator->getPotts()->registerEnergyFunctionWithName(this,toString());
	simulator->registerSteerableObject(this);

	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("MomentOfInertia",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);
}




void ConvergentExtensionPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}

void ConvergentExtensionPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
		set<unsigned char> cellTypesSet;

	interactingTypes.clear();
	alphaConvExtVec.clear();
	typeNameAlphaConvExtMap.clear();
	CC3DXMLElementList alphaVec=_xmlData->getElements("Alpha");

	for (int i = 0 ; i<alphaVec.size(); ++i){
		typeNameAlphaConvExtMap.insert(make_pair(alphaVec[i]->getAttribute("Type"),alphaVec[i]->getDouble()));


		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(automaton->getTypeId(alphaVec[i]->getAttribute("Type")));


	}

	//Now that we know all the types used in the simulation we will find size of the contactEnergyArray
	vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector

	int size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
	maxTypeId=size;
	size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated

	int index ;
	alphaConvExtVec.assign(size,0.0);
	//inserting alpha values to alphaConvExtVec;
	for(map<std::string , double>::iterator mitr=typeNameAlphaConvExtMap.begin() ; mitr!=typeNameAlphaConvExtMap.end(); ++mitr){
		alphaConvExtVec[automaton->getTypeId(mitr->first)]=mitr->second;
	}
	Log(LOG_DEBUG) << "size="<<size;
	for(int i = 0 ; i < size ; ++i){
		Log(LOG_DEBUG) << "alphaConvExt["<<i<<"]="<<alphaConvExtVec[i];
			}

	//Here I initialize max neighbor index for direct acces to the list of neighbors 
	boundaryStrategy=BoundaryStrategy::getInstance();
	maxNeighborIndex=0;

	if(_xmlData->getFirstElement("Depth")){
		maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
	}else{
		if(_xmlData->getFirstElement("NeighborOrder")){

			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());	
		}else{
			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

		}

	}
	Log(LOG_DEBUG) << "ConvergentExtension maxNeighborIndex="<<maxNeighborIndex;

}

double ConvergentExtensionPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {

	// Assumption simulation is 2D in xy plane

	double energy = 0;
	unsigned int token = 0;
	double distance = 0;
	double nCellAlpha,cellAlpha;
	Point3D n;

	CellG *nCell=0;
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	Neighbor neighbor;
	Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);

	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
		if(!neighbor.distance){
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}

		nCell = fieldG->get(neighbor.pt);
		Coordinates3D<double> ptTransNeighbor=boundaryStrategy->calculatePointCoordinates(neighbor.pt);

		if(nCell!=oldCell){
			if( !nCell || !oldCell || nCell->type>maxTypeId || oldCell->type>maxTypeId || !alphaConvExtVec[oldCell->type] || !alphaConvExtVec[nCell->type]){
				; //convergent extention contribution from this pair of cells is zero
			}else{
				//nCell
				double deltaNCell=alphaConvExtVec[nCell->type]*nCell->ecc;

				Coordinates3D<double> nCellCM((nCell->xCM / (float) nCell->volume),(nCell->yCM / (float) nCell->volume),(nCell->zCM / (float) nCell->volume));
				Coordinates3D<double> nCellCMPtVec=ptTransNeighbor-nCellCM;

				Coordinates3D<double> orientationVecNCell;
				if(nCell->lX==0.0 && nCell->lY==0.0){
					if(nCell->iXX>nCell->iYY){
						orientationVecNCell=Coordinates3D<double>(0,sqrt(nCell->iXX),0);
					}else{
						orientationVecNCell=Coordinates3D<double>(sqrt(nCell->iYY),0,0);
					}
				}else{
					orientationVecNCell=Coordinates3D<double>(nCell->lX,nCell->lY,0);
				}



				double rSinThetaNCell=(orientationVecNCell.x*nCellCMPtVec.y-orientationVecNCell.y*nCellCMPtVec.x)/
					sqrt(orientationVecNCell.x*orientationVecNCell.x+orientationVecNCell.y*orientationVecNCell.y);

				//double NNCell=(orientationVecNCell.x*nCellCMPtVec.y-orientationVecNCell.y*nCellCMPtVec.x);

				//double nsintheta=(orientationVecNCell.x*nCellCMPtVec.y-orientationVecNCell.y*nCellCMPtVec.x)/(
				//	sqrt(orientationVecNCell.x*orientationVecNCell.x+orientationVecNCell.y*orientationVecNCell.y)*(nCellCMPtVec.x*nCellCMPtVec.x+nCellCMPtVec.y*nCellCMPtVec.y));
				
				//double r_eccN=sqrt(nCellCMPtVec.x*nCellCMPtVec.x+nCellCMPtVec.y*nCellCMPtVec.y)*nCell->ecc;
				

				deltaNCell*=rSinThetaNCell;
				//oldCell
				double deltaOldCell=alphaConvExtVec[oldCell->type]*oldCell->ecc;
				Coordinates3D<double> oldCellCM((oldCell->xCM / (float) oldCell->volume),(oldCell->yCM / (float) oldCell->volume),(oldCell->zCM / (float) oldCell->volume));
				Coordinates3D<double> oldCellCMPtVec=ptTrans-oldCellCM;

				Coordinates3D<double> orientationVecOldCell;
				if(oldCell->lX==0.0 && oldCell->lY==0.0){
					if(oldCell->iXX>oldCell->iYY){
						orientationVecOldCell=Coordinates3D<double>(0,sqrt(oldCell->iXX),0);
					}else{
						orientationVecOldCell=Coordinates3D<double>(sqrt(oldCell->iYY),0,0);
					}
				}else{
					orientationVecOldCell=Coordinates3D<double>(oldCell->lX,oldCell->lY,0);
				}


				double rSinThetaOldCell=(orientationVecOldCell.x*oldCellCMPtVec.y-orientationVecOldCell.y*oldCellCMPtVec.x)/
					sqrt(orientationVecOldCell.x*orientationVecOldCell.x+orientationVecOldCell.y*orientationVecOldCell.y);
				
				//double r_eccOld=sqrt(oldCellCMPtVec.x*oldCellCMPtVec.x+oldCellCMPtVec.y*oldCellCMPtVec.y)*oldCell->ecc;
				
				deltaOldCell*=rSinThetaOldCell;

				double energyBefore=energy;
				if (nCell->volume==1 || oldCell->volume==1){
					;//dont do anything if cell is only one pixel in size - the math makes no sense in such a case
				}else{
					energy -= -deltaOldCell*deltaNCell;
				}


				  if(energy!=energy){
					Log(LOG_DEBUG) << "energyBefore="<<energyBefore;
					Log(LOG_DEBUG) << "oldCellCMPtVec="<<oldCellCMPtVec<<" oldCell->lX="<<oldCell->lX<<" oldCell->lY="<<oldCell->lY;
					Log(LOG_DEBUG) << "oldCell->iXX="<<oldCell->iXX<<" oldCell->iYY="<<oldCell->iYY<<" oldCell->iXY="<<oldCell->iXY;
					Log(LOG_DEBUG) << "nCellCMPtVec="<<nCellCMPtVec<<" nCell->lX="<<nCell->lX<<" nCell->lY="<<nCell->lY;
					Log(LOG_DEBUG) << "deltaNCell="<<deltaNCell<<" rSinThetaNCell="<<rSinThetaNCell<<" nCell->ecc="<<nCell->ecc;
					Log(LOG_DEBUG) << "nCell->volume="<<nCell->volume;
					Log(LOG_DEBUG) << "deltaOldCell="<<deltaOldCell<<" rSinThetaOldCell="<<rSinThetaOldCell<<" oldCell->ecc="<<oldCell->ecc;
					Log(LOG_DEBUG) << "oldCell->volume="<<oldCell->volume;
					Log(LOG_DEBUG) << "deltaOldCell="<<deltaOldCell<<" deltaNCell="<<deltaNCell;
					Log(LOG_DEBUG) << "OLD N CELL CONTR="<<energy;
					exit(0);
					}

			}
		}
		if(nCell!=newCell){
			if( !nCell || !newCell || nCell->type>maxTypeId || newCell->type>maxTypeId || !alphaConvExtVec[newCell->type] || !alphaConvExtVec[nCell->type]){
				; //convergent extention contribution from this pair of cells is zero
			}else{
				//calculating quantities needed for delta computations for newCell after pixel copy

				double xcm;
				double ycm;
				if(newCell->volume<1){
					xcm = 0.0;
					ycm = 0.0;

				}else{
					xcm = (newCell->xCM / (float) newCell->volume);
					ycm = (newCell->yCM / (float) newCell->volume);

				}
				double newXCM = (newCell->xCM + ptTrans.x)/((float)newCell->volume + 1);
				double newYCM = (newCell->yCM + ptTrans.y)/((float)newCell->volume + 1);	 

				double newIxx=newCell->iXX+(newCell->volume )*ycm*ycm-(newCell->volume+1)*(newYCM*newYCM)+ptTrans.y*ptTrans.y;
				double newIyy=newCell->iYY+(newCell->volume )*xcm*xcm-(newCell->volume+1)*(newXCM*newXCM)+ptTrans.x*ptTrans.x;
				double newIxy=newCell->iXY-(newCell->volume )*xcm*ycm+(newCell->volume+1)*newXCM*newYCM-ptTrans.x*ptTrans.y;

				double radicalNew=0.5*sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy);			
				double lMinNew=0.5*(newIxx+newIyy)-radicalNew;
				double lMaxNew=0.5*(newIxx+newIyy)+radicalNew;
				double newEcc=sqrt(1.0-lMinNew/lMaxNew);

				Coordinates3D<double> orientationVecNew;
				if(newIxy!=0.0){
				
					orientationVecNew=Coordinates3D<double>(newIxy,lMaxNew-newIxx,0.0);
				}else{
					if(newIxx>newIyy)
						orientationVecNew=Coordinates3D<double>(0.0,sqrt(newIxx),0.0);
					else
						orientationVecNew=Coordinates3D<double>(sqrt(newIyy),0.0,0.0);
				}

				double deltaNewCell=alphaConvExtVec[newCell->type]*newEcc;

				Coordinates3D<double> newCellCM(newXCM,newYCM,0.0);
				Coordinates3D<double> newCellCMPtVec=ptTrans-newCellCM;
				
				double N=sqrt((orientationVecNew.x*newCellCMPtVec.y-orientationVecNew.y*newCellCMPtVec.x)*(orientationVecNew.x*newCellCMPtVec.y-orientationVecNew.y*newCellCMPtVec.x));
				double D=sqrt(orientationVecNew.x*orientationVecNew.x+orientationVecNew.y*orientationVecNew.y);
				double rSinThetaNewCell=(orientationVecNew.x*newCellCMPtVec.y-orientationVecNew.y*newCellCMPtVec.x)/
					sqrt(orientationVecNew.x*orientationVecNew.x+orientationVecNew.y*orientationVecNew.y);

				//double rSinThetaNewCell=sqrt((orientationVecNew.x*newCellCMPtVec.y-orientationVecNew.y*newCellCMPtVec.x)*(orientationVecNew.y*newCellCMPtVec.y-orientationVecNew.y*newCellCMPtVec.x))/
				//	sqrt(orientationVecNew.x*orientationVecNew.x+orientationVecNew.y*orientationVecNew.y);

				deltaNewCell*=rSinThetaNewCell;


				if(nCell==oldCell){
					//special case - need to calculate delta for oldCell after pixel copy
					//oldCell						
					double xcmOldCell = (oldCell->xCM / (float) oldCell->volume);
					double ycmOldCell = (oldCell->yCM / (float) oldCell->volume);
					double newXCMOldCell;
					double newYCMOldCell;

					if(oldCell->volume==1){
						newXCMOldCell = 0.0;
						newYCMOldCell = 0.0;	 
					}else{
						newXCMOldCell = (oldCell->xCM - ptTrans.x)/((float)oldCell->volume - 1);
						newYCMOldCell = (oldCell->yCM - ptTrans.y)/((float)oldCell->volume - 1);	 		
					}

					double newIxxOldCell =oldCell->iXX+(oldCell->volume )*(ycmOldCell*ycmOldCell)-(oldCell->volume-1)*(newYCMOldCell*newYCMOldCell)-ptTrans.y*ptTrans.y;
					double newIyyOldCell =oldCell->iYY+(oldCell->volume )*(xcmOldCell*xcmOldCell)-(oldCell->volume-1)*(newXCMOldCell*newXCMOldCell)-ptTrans.x*ptTrans.x;
					double newIxyOldCell =oldCell->iXY-(oldCell->volume )*(xcmOldCell*ycmOldCell)+(oldCell->volume-1)*newXCMOldCell*newYCMOldCell+ptTrans.x*ptTrans.y;

					double radicalNewOldCell=0.5*sqrt((newIxxOldCell-newIyyOldCell)*(newIxxOldCell-newIyyOldCell)+4.0*newIxyOldCell*newIxyOldCell);			
					double lMinNewOldCell=0.5*(newIxxOldCell+newIyyOldCell)-radicalNew;
					double lMaxNewOldCell=0.5*(newIxxOldCell+newIyyOldCell)+radicalNew;
					double newEccOldCell=sqrt(1.0-lMinNewOldCell/lMaxNewOldCell);
					Coordinates3D<double> orientationVecNewOldCell;
					if(newIxyOldCell!=0.0){
						orientationVecNewOldCell=Coordinates3D<double>(newIxyOldCell,lMaxNewOldCell-newIxxOldCell,0.0);
					}else{
						if(newIxxOldCell>newIyyOldCell)
							orientationVecNewOldCell=Coordinates3D<double>(0.0,sqrt(newIxxOldCell),0.0);
						else
							orientationVecNewOldCell=Coordinates3D<double>(sqrt(newIyyOldCell),0.0,0.0);

					}


					//oldCell
					double deltaOldCell=alphaConvExtVec[oldCell->type]*newEccOldCell;
					Coordinates3D<double> oldCellCM(newXCMOldCell,newXCMOldCell,0.0);
					Coordinates3D<double> oldCellCMPtVec=ptTransNeighbor-oldCellCM;


					double rSinThetaOldCell=(orientationVecNewOldCell.x*oldCellCMPtVec.y-orientationVecNewOldCell.y*oldCellCMPtVec.x)/
						sqrt(orientationVecNewOldCell.x*orientationVecNewOldCell.x+orientationVecNewOldCell.y*orientationVecNewOldCell.y);


					deltaOldCell*=rSinThetaOldCell;
					
					double energyBefore=energy;

					if (newCell->volume==1 || oldCell->volume<=2 ){
						;//dont do anything if cell is only one pixel in size - the math makes no sense in such a case
					}else{

						energy += -deltaOldCell*deltaNewCell;
				  }

				  if(energy!=energy){
					Log(LOG_DEBUG) << "energyBefore="<<energyBefore;
					Log(LOG_DEBUG) << "oldCell->volume="<<oldCell->volume;
					Log(LOG_DEBUG) << "oldCell->iXX="<<oldCell->iXX<<" oldCell->iYY="<<oldCell->iYY<<" oldCell->iXY="<<oldCell->iXY;
					Log(LOG_DEBUG) << "newIxxOldCell="<<newIxxOldCell<<" newIyyOldCell="<<newIyyOldCell<<" newIxyOldCell="<<newIxyOldCell;
					Log(LOG_DEBUG) << "orientationVecNewOldCell="<<orientationVecNewOldCell;
					Log(LOG_DEBUG) << "deltaOldCell="<<deltaOldCell<<" deltaNewCell="<<deltaNewCell;
					Log(LOG_DEBUG) << "NEW OLD CELL CONTR="<<energy;
					exit(0);
					}

				}else{

					//nCell
					double deltaNCell=alphaConvExtVec[nCell->type]*nCell->ecc;

					Coordinates3D<double> nCellCM((nCell->xCM / (float) nCell->volume),(nCell->yCM / (float) nCell->volume),(nCell->zCM / (float) nCell->volume));
					Coordinates3D<double> nCellCMPtVec=ptTransNeighbor-nCellCM;

					Coordinates3D<double> orientationVecNCell;
					if(nCell->lX==0.0 && nCell->lY==0.0){
						if(nCell->iXX>nCell->iYY){
							orientationVecNCell=Coordinates3D<double>(0.,sqrt(nCell->iXX),0);
						}else{
							orientationVecNCell=Coordinates3D<double>(sqrt(nCell->iYY),0,0);
						}
					}else{
						orientationVecNCell=Coordinates3D<double>(nCell->lX,nCell->lY,0);
					}



					double rSinThetaNCell=(orientationVecNCell.x*nCellCMPtVec.y-orientationVecNCell.y*nCellCMPtVec.x)/
						sqrt(orientationVecNCell.x*orientationVecNCell.x+orientationVecNCell.y*orientationVecNCell.y);



					//double rSinThetaNCell=sqrt((nCell->lX*nCellCMPtVec.y-nCell->lY*nCellCMPtVec.x)*(nCell->lX*nCellCMPtVec.y-nCell->lY*nCellCMPtVec.x))/
					//	sqrt(nCell->lX*nCell->lX+nCell->lY*nCell->lY);

					deltaNCell*=rSinThetaNCell;

					

				

					double energyBefore=energy;

					if (nCell->volume==1 || newCell->volume==1){
						;//dont do anything if  cell is only one pixel in size - the math makes no sense in such a case
					}else{

						energy += -deltaNCell*deltaNewCell;
					}

					if(energy!=energy){
						Log(LOG_DEBUG) << "deltaNCell="<<deltaNCell<<" rSinThetaNCell="<<rSinThetaNCell;
						Log(LOG_DEBUG) << "deltaNewCell="<<deltaNewCell<<" rSinThetaNewCell="<<rSinThetaNewCell<<" newCell->volume="<<newCell->volume;
						Log(LOG_DEBUG) << "N="<<N<<" D="<<D;
						Log(LOG_DEBUG) << "orientationVecNew="<<orientationVecNew;
						Log(LOG_DEBUG) << "newCellCMPtVec="<<newCellCMPtVec;
						Log(LOG_DEBUG) << "newIxx="<<newIxx;
						Log(LOG_DEBUG) << "newIyy="<<newIyy;
						Log(LOG_DEBUG) << "newIxy="<<newIxy;
						Log(LOG_DEBUG) << "xcm="<<xcm<<" ycm="<<ycm<<" newXCM="<<newXCM<<" newYCM="<<newYCM;
						Log(LOG_DEBUG) << "radicalNew="<<radicalNew;
						Log(LOG_DEBUG) << "lMinNew="<<lMinNew;
						Log(LOG_DEBUG) << "lMaxNew="<<lMaxNew;
						Log(LOG_DEBUG) << "newEcc="<<newEcc;						
						Log(LOG_DEBUG) << "energyBefore="<<energyBefore;
						Log(LOG_DEBUG) << "deltaNewCell="<<deltaNewCell<<" deltaNCell="<<deltaNCell;
						Log(LOG_DEBUG) << "NEW N CELL CONTR="<<energy;
						exit(0);
					}


	

				}
				//energy += contactEnergy(newCell, nCell);

			}

		}


	}
	if(energy!=energy){
		return 0.0;
	}
	else{
		return energy;
	}

}


std::string ConvergentExtensionPlugin::steerableName(){return "ConvergentExtansion";}
std::string ConvergentExtensionPlugin::toString(){return steerableName();}



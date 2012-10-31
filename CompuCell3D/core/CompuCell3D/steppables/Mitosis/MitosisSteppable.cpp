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


#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <PublicUtilities/NumericalUtils.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>

#include <time.h>
#include <limits>


using namespace CompuCell3D;

#include <iostream>
#include <complex>
using namespace std;

#undef max
#undef min

#include "MitosisSteppable.h"
#include "CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h"
#include "CompuCell3D/plugins/PixelTracker/PixelTracker.h"


MitosisSteppable::MitosisSteppable()
{
	parentChildPositionFlag=0;
	potts=0;
	doDirectionalMitosis2DPtr=0;
	// // // getOrientationVectorsMitosis2DPtr=0;

}

MitosisSteppable::~MitosisSteppable() {}

// // // void MitosisSteppable::setDivideAlongMinorAxis(){
// // // divideAlongMinorAxisFlag=true;
// // // divideAlongMajorAxisFlag=false;

// // // }

// // // void MitosisSteppable::setDivideAlongMajorAxis(){
// // // divideAlongMinorAxisFlag=false;
// // // divideAlongMajorAxisFlag=true;
// // // }

void MitosisSteppable::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

	potts = simulator->getPotts();
	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("VolumeTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	cerr<<"GOT HERE BEFORE CALLING INIT"<<endl;
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	Plugin *pluginCOM=Simulator::pluginManager.get("CenterOfMass",&pluginAlreadyRegisteredFlag); //this will load CenterOfMass plugin if it is not already loaded
	cerr<<"GOT HERE BEFORE CALLING INIT"<<endl;
	if(!pluginAlreadyRegisteredFlag)
		pluginCOM->init(simulator);


	pixelTrackerPlugin=(PixelTrackerPlugin*)Simulator::pluginManager.get("PixelTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!pluginAlreadyRegisteredFlag)
		pixelTrackerPlugin->init(simulator);

	pixelTrackerAccessorPtr=pixelTrackerPlugin->getPixelTrackerAccessorPtr();

	fieldDim=simulator->getPotts()->getCellFieldG()->getDim();

	// // // if( fieldDim.x !=1 && fieldDim.y != 1 && fieldDim.z != 1){
	// // // flag3D=true;
	// // // }else{
	// // // if(fieldDim.x ==1){
	// // // doDirectionalMitosis2DPtr=&MitosisSteppable::doDirectionalMitosis2D_yz;
	// // // getOrientationVectorsMitosis2DPtr=&MitosisSteppable::getOrientationVectorsMitosis2D_yz;
	// // // flag3D=false;
	// // // }
	// // // else if (fieldDim.y ==1){
	// // // doDirectionalMitosis2DPtr=&MitosisSteppable::doDirectionalMitosis2D_xz;
	// // // getOrientationVectorsMitosis2DPtr=&MitosisSteppable::getOrientationVectorsMitosis2D_xz;
	// // // flag3D=false;
	// // // }
	// // // else if (fieldDim.z ==1){
	// // // doDirectionalMitosis2DPtr=&MitosisSteppable::doDirectionalMitosis2D_xy;
	// // // getOrientationVectorsMitosis2DPtr=&MitosisSteppable::getOrientationVectorsMitosis2D_xy;
	// // // flag3D=false;
	// // // }
	// // // }

	// // // turnOn();
	boundaryStrategy=BoundaryStrategy::getInstance();
	maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(5);

	potts->getBoundaryXName()=="Periodic" ? boundaryConditionIndicator.x=1 : boundaryConditionIndicator.x=0 ;
	potts->getBoundaryYName()=="Periodic" ? boundaryConditionIndicator.y=1 : boundaryConditionIndicator.y=0;
	potts->getBoundaryZName()=="Periodic" ? boundaryConditionIndicator.z=1 : boundaryConditionIndicator.z=0;
	
	fieldDim=potts->getCellFieldG()->getDim();

	if( fieldDim.x !=1 && fieldDim.y != 1 && fieldDim.z != 1){
		flag3D=true;
	}else{
	    flag3D=false;
		if(fieldDim.x ==1){
			getOrientationVectorsMitosis2DPtr=&MitosisSteppable::getOrientationVectorsMitosis2D_yz;			
		}
		else if (fieldDim.y ==1){			
			getOrientationVectorsMitosis2DPtr=&MitosisSteppable::getOrientationVectorsMitosis2D_xz;			
		}
		else if (fieldDim.z ==1){			
			getOrientationVectorsMitosis2DPtr=&MitosisSteppable::getOrientationVectorsMitosis2D_xy;			
		}

	}

	LatticeType latticeType=boundaryStrategy->getLatticeType();
	//we use these factors to rescale pixels expressed in absolute coordinates to those of the underlying cartesian lattice - this is done for pixel lookup 
	//because pixels are stored in sets as integers - i.e. in a "cartesian form"

	xFactor=yFactor=zFactor=1.0;
	if (latticeType==HEXAGONAL_LATTICE){
		yFactor=2.0/sqrt(3.0);
		zFactor=3.0/sqrt(6.0);
	}
	

	//if (boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z){
	//	ASSERT_OR_THROW("CURRENT VERSION OF MITOSIS STEPPABLE CANNOT BE USED WITH PERIODIC BOUNDARY CONDITIONS.WE ARE WORKING ON SOLVING THIS LIMITATION",
	//	!boundaryConditionIndicator.x && !boundaryConditionIndicator.y && !boundaryConditionIndicator.z);
	//}	


		if(!simulator->ppdCC3DPtr->seed){
			srand(time(0));
			unsigned int randomSeed=(unsigned int)rand()*((std::numeric_limits<unsigned int>::max)()-1);                
			randGen.setSeed(randomSeed);
		}else{
			randGen.setSeed(simulator->ppdCC3DPtr->seed);
		}            

}

void MitosisSteppable::setParentChildPositionFlag(int _flag){
	parentChildPositionFlag=_flag;
}

int MitosisSteppable::getParentChildPositionFlag(){
	return parentChildPositionFlag;
}

Vector3 MitosisSteppable::getShiftVector(std::set<PixelTrackerData> & _sourcePixels){
	Point3D ptRef=_sourcePixels.begin()->pixel; //reference point
	//////cerr<<"ptRef="<<ptRef<<endl;
	//////cerr<<"boundaryConditionIndicator="<<boundaryConditionIndicator<<endl;
	//calculate shift vector as a difference between reference point and a center of the lattice
	Vector3 shiftVec;
	shiftVec.fX=(ptRef.x-((int)fieldDim.x/2))*boundaryConditionIndicator.x;
	shiftVec.fY=(ptRef.y-((int)fieldDim.y/2))*boundaryConditionIndicator.y;
	shiftVec.fZ=(ptRef.z-((int)fieldDim.z/2))*boundaryConditionIndicator.z;

	return shiftVec;
}

void MitosisSteppable::shiftCellPixels(std::set<PixelTrackerData> & _sourcePixels, std::set<PixelTrackerData> & _targetPixels,Vector3 _shiftVec){
	//this method shifts all the pixels from _sourcePixels set and puts them in the _targetPixels set and makes sure targetPixels are inthe lattice
	//for this function to work properly we have to make sure that shift vector coordinates are <= lattice dimenntions


	////_targetPixels.insert(_sourcePixels.begin(),_sourcePixels.end());
	////return ;


	Point3D pt;
	//Point3D ptRef=_sourcePixels.begin()->pixel; //reference point
	////calculate shift vector as a difference between reference point and a center of the lattice
	//Vector3 shiftVec;
	//shiftVec.x=(ptRef.x-((int)fieldDimTrans.x)/2))*boundaryConditionIndicator.x;
	//shiftVec.y=(ptRef.y-((int)fieldDimTrans.y)/2))*boundaryConditionIndicator.y;
	//shiftVec.z=(ptRef.z-((int)fieldDimTrans.z)/2))*boundaryConditionIndicator.z;

	for(set<PixelTrackerData>::iterator sitr=_sourcePixels.begin() ; sitr != _sourcePixels.end() ;++sitr){
		pt=sitr->pixel;
		
		pt.x-=_shiftVec.fX;
		pt.y-=_shiftVec.fY;
		pt.z-=_shiftVec.fZ;


		//making sure that shifted point is in the lattice
		if(pt.x < 0){
		  pt.x += fieldDim.x;
		}else if (pt.x >= fieldDim.x){
		  pt.x -= fieldDim.x;
		}  

		if(pt.y < 0){
		  pt.y += fieldDim.y;
		}else if (pt.y >= fieldDim.y){
		  pt.y -= fieldDim.y;
		}  

		if(pt.z < 0){
		  pt.z += fieldDim.z;
		}else if (pt.z >= fieldDim.z){
		  pt.z -= fieldDim.z;
		}  

		_targetPixels.insert(PixelTrackerData(pt));
	}

}


SteppableOrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis(CellG * _cell){
	//set<PixelTrackerData>  & cellPixels=pixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;
	//return getOrientationVectorsMitosis(cellPixels);

	set<PixelTrackerData>  & cellPixels=pixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;
	if (boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z){
		
		Vector3 shiftVec=getShiftVector(cellPixels);
		set<PixelTrackerData> targetPixels;
		shiftCellPixels(cellPixels,targetPixels,shiftVec);
		return getOrientationVectorsMitosis(targetPixels);

	}else{

		return getOrientationVectorsMitosis(cellPixels);

	}
}

SteppableOrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosisCompartments(long _clusterId){
	CellInventory & inventory=potts->getCellInventory();
	CC3DCellList compartmentsVec=inventory.getClusterCells(_clusterId);

	//construct a set containing all pixels of the cluster
	set<PixelTrackerData> clusterPixels;		
	for(int i = 0 ; i < compartmentsVec.size(); ++i){
		CellG *cell=compartmentsVec[i];
		set<PixelTrackerData> & cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
		clusterPixels.insert(cellPixels.begin(),cellPixels.end());
	}
	//now depending if b.c.'s are defined or not we will shift cluster pixeld to "center of the lattice"
	if (boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z){

		Vector3 shiftVec=getShiftVector(clusterPixels);
		set<PixelTrackerData> targetPixels;
		shiftCellPixels(clusterPixels,targetPixels,shiftVec);
		return getOrientationVectorsMitosis(targetPixels);
	}else{

		return getOrientationVectorsMitosis(clusterPixels);
	}

}



SteppableOrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis(std::set<PixelTrackerData> & clusterPixels){
	if(flag3D){

		return getOrientationVectorsMitosis3D(clusterPixels);
	}
	else{
		return (this->*getOrientationVectorsMitosis2DPtr)(clusterPixels);
	}
}

SteppableOrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis2D_xy(std::set<PixelTrackerData> & clusterPixels){
		Vector3 clusterCOM=calculateClusterPixelsCOM(clusterPixels);
		double xcm=clusterCOM.fX;
		double ycm=clusterCOM.fY;
		double zcm=clusterCOM.fZ;
		//cerr<<"get orientation vector clusterCOM="<<clusterCOM<<endl;

		//first calculate and diagonalize inertia tensor
		vector<vector<double> > inertiaTensor(2,vector<double>(2,0.0));
		

		
		for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr != clusterPixels.end() ;++sitr){
			//cerr<<"point cell="<<sitr->pixel<<endl;
			Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
			inertiaTensor[0][0]+=(pixelTrans.y-ycm)*(pixelTrans.y-ycm);
			inertiaTensor[0][1]+=-(pixelTrans.x-xcm)*(pixelTrans.y-ycm);
			inertiaTensor[1][1]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm);
		}
		inertiaTensor[1][0]=inertiaTensor[0][1];

		double radical=0.5*sqrt((inertiaTensor[0][0]-inertiaTensor[1][1])*(inertiaTensor[0][0]-inertiaTensor[1][1])+4.0*inertiaTensor[0][1]*inertiaTensor[0][1]);			
		double lMin=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])-radical;
		double lMax=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])+radical;

		//orientationVec points along semiminor axis (it corresponds to larger eigenvalue)		
		Vector3 orientationVec;
		if(inertiaTensor[0][1]!=0.0){

			orientationVec=Vector3(inertiaTensor[0][1],lMax-inertiaTensor[0][0],0.0);
			double length=orientationVec.Mag();
			orientationVec*=1.0/length;
		}else{
			if(inertiaTensor[0][0]>inertiaTensor[1][1])
				orientationVec=Vector3(0.0,1.0,0.0);
			else
				orientationVec=Vector3(1.0,0.0,0.0);
		}
		
		
		SteppableOrientationVectorsMitosis orientationVectorsMitosis;
		orientationVectorsMitosis.semiminorVec=orientationVec;
		orientationVectorsMitosis.semimajorVec.fX=-orientationVectorsMitosis.semiminorVec.fY;
		orientationVectorsMitosis.semimajorVec.fY=orientationVectorsMitosis.semiminorVec.fX;

		return orientationVectorsMitosis;
}

SteppableOrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis2D_xz(std::set<PixelTrackerData> & clusterPixels){
		Vector3 clusterCOM=calculateClusterPixelsCOM(clusterPixels);
		double xcm=clusterCOM.fX;
		double ycm=clusterCOM.fY;
		double zcm=clusterCOM.fZ;

		//first calculate and diagonalize inertia tensor
		vector<vector<double> > inertiaTensor(2,vector<double>(2,0.0));

		
		for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr != clusterPixels.end() ;++sitr){
			//cerr<<"point cell="<<sitr->pixel<<endl;
			Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
			inertiaTensor[0][0]+=(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
			inertiaTensor[0][1]+=-(pixelTrans.x-xcm)*(pixelTrans.z-zcm);
			inertiaTensor[1][1]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm);
		}
		inertiaTensor[1][0]=inertiaTensor[0][1];

		double radical=0.5*sqrt((inertiaTensor[0][0]-inertiaTensor[1][1])*(inertiaTensor[0][0]-inertiaTensor[1][1])+4.0*inertiaTensor[0][1]*inertiaTensor[0][1]);			
		double lMin=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])-radical;
		double lMax=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])+radical;

		//orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
		Vector3 orientationVec;
		if(inertiaTensor[0][1]!=0.0){

			orientationVec=Vector3(inertiaTensor[0][1],0.0,lMax-inertiaTensor[0][0]);
			double length=orientationVec.Mag();
			orientationVec*=1.0/length;
		}else{
			if(inertiaTensor[0][0]>inertiaTensor[1][1])
				orientationVec=Vector3(0.0,0.0,1.0);
			else
				orientationVec=Vector3(1.0,0.0,0.0);
		}

		SteppableOrientationVectorsMitosis orientationVectorsMitosis;
		orientationVectorsMitosis.semiminorVec=orientationVec;
		orientationVectorsMitosis.semimajorVec.fX=-orientationVectorsMitosis.semiminorVec.fZ;
		orientationVectorsMitosis.semimajorVec.fZ=orientationVectorsMitosis.semiminorVec.fX;

		return orientationVectorsMitosis;

}

SteppableOrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis2D_yz(std::set<PixelTrackerData> & clusterPixels){
		Vector3 clusterCOM=calculateClusterPixelsCOM(clusterPixels);
		double xcm=clusterCOM.fX;
		double ycm=clusterCOM.fY;
		double zcm=clusterCOM.fZ;

		//first calculate and diagonalize inertia tensor
		vector<vector<double> > inertiaTensor(2,vector<double>(2,0.0));

		
		for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr != clusterPixels.end() ;++sitr){
			Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
			//cerr<<"point cell="<<sitr->pixel<<endl;
			inertiaTensor[0][0]+=(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
			inertiaTensor[0][1]+=-(pixelTrans.y-ycm)*(pixelTrans.z-zcm);
			inertiaTensor[1][1]+=(pixelTrans.y-ycm)*(pixelTrans.y-ycm);
		}
		inertiaTensor[1][0]=inertiaTensor[0][1];

		double radical=0.5*sqrt((inertiaTensor[0][0]-inertiaTensor[1][1])*(inertiaTensor[0][0]-inertiaTensor[1][1])+4.0*inertiaTensor[0][1]*inertiaTensor[0][1]);			
		double lMin=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])-radical;
		double lMax=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])+radical;

		//orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
		Vector3 orientationVec;
		if(inertiaTensor[0][1]!=0.0){

			orientationVec=Vector3(0.0,inertiaTensor[0][1],lMax-inertiaTensor[0][0]);
			double length=orientationVec.Mag();
			orientationVec*=1.0/length;
		}else{
			if(inertiaTensor[0][0]>inertiaTensor[1][1])
				orientationVec=Vector3(0.0,0.0,1.0);
			else
				orientationVec=Vector3(0.0,1.0,0.0);
		}

		SteppableOrientationVectorsMitosis orientationVectorsMitosis;
		orientationVectorsMitosis.semiminorVec=orientationVec;
		orientationVectorsMitosis.semimajorVec.fY=-orientationVectorsMitosis.semiminorVec.fZ;
		orientationVectorsMitosis.semimajorVec.fZ=orientationVectorsMitosis.semiminorVec.fY;

		return orientationVectorsMitosis;

}

SteppableOrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis3D(std::set<PixelTrackerData> & clusterPixels){

		Vector3 clusterCOM=calculateClusterPixelsCOM(clusterPixels);
		double xcm=clusterCOM.fX;
		double ycm=clusterCOM.fY;
		double zcm=clusterCOM.fZ;

		//first calculate and diagonalize inertia tensor
		vector<vector<double> > inertiaTensor(3,vector<double>(3,0.0));

		for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr != clusterPixels.end() ;++sitr){
			Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
			//cerr<<"point cell="<<sitr->pixel<<endl;
			inertiaTensor[0][0]+=(pixelTrans.y-ycm)*(pixelTrans.y-ycm)+(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
			inertiaTensor[0][1]+=-(pixelTrans.x-xcm)*(pixelTrans.y-ycm);
			inertiaTensor[0][2]+=-(pixelTrans.x-xcm)*(pixelTrans.z-zcm);
			inertiaTensor[1][1]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm)+(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
			inertiaTensor[1][2]+=-(pixelTrans.y-ycm)*(pixelTrans.z-zcm);
			inertiaTensor[2][2]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm)+(pixelTrans.y-ycm)*(pixelTrans.y-ycm);
		}
		inertiaTensor[1][0]=inertiaTensor[0][1];
		inertiaTensor[2][0]=inertiaTensor[0][2];
		inertiaTensor[2][1]=inertiaTensor[1][2];
		
	 //Finding eigenvalues
	 vector<double> aCoeff(4,0.0);
	 vector<complex<double> > roots;
	 
	 //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
	 aCoeff[0]=-1.0;

	 aCoeff[1]=inertiaTensor[0][0] + inertiaTensor[1][1] + inertiaTensor[2][2];

	 aCoeff[2]=inertiaTensor[0][1]*inertiaTensor[0][1] + inertiaTensor[0][2]*inertiaTensor[0][2] + inertiaTensor[1][2]*inertiaTensor[1][2]
	 -inertiaTensor[0][0]*inertiaTensor[1][1] - inertiaTensor[0][0]*inertiaTensor[2][2] - inertiaTensor[1][1]*inertiaTensor[2][2];

	 aCoeff[3]=inertiaTensor[0][0]*inertiaTensor[1][1]*inertiaTensor[2][2] + 2*inertiaTensor[0][1]*inertiaTensor[0][2]*inertiaTensor[1][2]
	 -inertiaTensor[0][0]*inertiaTensor[1][2]*inertiaTensor[1][2]
	 -inertiaTensor[1][1]*inertiaTensor[0][2]*inertiaTensor[0][2]
	 -inertiaTensor[2][2]*inertiaTensor[0][1]*inertiaTensor[0][1];

	 roots=solveCubicEquationRealCoeeficients(aCoeff);
	 
	 

	 vector<Vector3> eigenvectors(3);

	 for (int i = 0 ; i < 3 ; ++i){
		 eigenvectors[i].fX=(inertiaTensor[0][2]*(inertiaTensor[1][1]-roots[i].real())-inertiaTensor[1][2]*inertiaTensor[0][1])/
		 (inertiaTensor[1][2]*(inertiaTensor[0][0]-roots[i].real())-inertiaTensor[0][1]*inertiaTensor[0][2]);
		 eigenvectors[i].fY=1.0;
		 eigenvectors[i].fZ = (inertiaTensor[0][2]*eigenvectors[i].fX+inertiaTensor[1][2]*eigenvectors[i].fY)/
		 (roots[i].real()-inertiaTensor[2][2]) ;
		 //cerr<<"eigenvectors["<<i<<"]="<<eigenvectors[i]<<endl;
		 if(eigenvectors[i].fX!=eigenvectors[i].fX || eigenvectors[i].fY!=eigenvectors[i].fY || eigenvectors[i].fZ!=eigenvectors[i].fZ){
			SteppableOrientationVectorsMitosis orientationVectorsMitosis;
			return orientationVectorsMitosis;//simply dont do mitosis if any of the eigenvector component is NaN
		 }
	 }



	 //finding semiaxes of the ellipsoid
	 //Ixx=m/5.0*(a_y^2+a_z^2) - andy cyclical permutations for other coordinate combinations
	 //a_x,a_y,a_z are lengths of semiaxes of the allipsoid
	 // We can invert above system of equations to get:
	vector<double> axes(3,0.0);
	int volume=clusterPixels.size();
	axes[0]=sqrt((2.5/volume)*(roots[1].real()+roots[2].real()-roots[0].real()));//corresponds to first eigenvalue
	axes[1]=sqrt((2.5/volume)*(roots[0].real()+roots[2].real()-roots[1].real()));//corresponds to second eigenvalue
	axes[2]=sqrt((2.5/volume)*(roots[0].real()+roots[1].real()-roots[2].real()));//corresponds to third eigenvalue

	vector<pair<double, int> > sortedAxes(3);
	sortedAxes[0]=make_pair(axes[0],0);
	sortedAxes[1]=make_pair(axes[1],1);
	sortedAxes[2]=make_pair(axes[2],2);

	//sorting semiaxes according the their lengths (shortest first)
	//sort(axes.begin(),axes.end());
	sort(sortedAxes.begin(),sortedAxes.end()); //by keeping track of original axes indices we also find which eigenvector corresponds to shortest/longest axis - that's why we use pair where first element is the length of the axis and the second one is index of the eigenvalue. 
															 //After sorting we can track back which eigenvector belongs to hosrtest/longest eigenvalue

	   SteppableOrientationVectorsMitosis orientationVectorsMitosis;
		orientationVectorsMitosis.semiminorVec=eigenvectors[sortedAxes[0].second];
		orientationVectorsMitosis.semimajorVec=eigenvectors[sortedAxes[2].second];
		

		return orientationVectorsMitosis;

}

bool MitosisSteppable::doDirectionalMitosisAlongMajorAxis(CellG* _cell){
	SteppableOrientationVectorsMitosis orientationVectors=getOrientationVectorsMitosis(_cell);
	return doDirectionalMitosisOrientationVectorBased(_cell,orientationVectors.semiminorVec.fX,orientationVectors.semiminorVec.fY,orientationVectors.semiminorVec.fZ);
}

bool MitosisSteppable::doDirectionalMitosisAlongMinorAxis(CellG* _cell){
	SteppableOrientationVectorsMitosis orientationVectors=getOrientationVectorsMitosis(_cell);
	return doDirectionalMitosisOrientationVectorBased(_cell,orientationVectors.semimajorVec.fX,orientationVectors.semimajorVec.fY,orientationVectors.semimajorVec.fZ);
}


bool MitosisSteppable::doDirectionalMitosisRandomOrientation(CellG* _cell){
	BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
	double cos_theta=-1.0+rand->getRatio()*2.0;
	double sin_theta=sqrt(1.0-cos_theta*cos_theta);
	double sin_phi=-1.0+rand->getRatio()*2.0;
	double cos_phi=sqrt(1.0-sin_phi*sin_phi);
	if (cos_theta==1.0 || cos_theta==0.0)
		return doDirectionalMitosisOrientationVectorBased(_cell,1,1,1);
	else
		return doDirectionalMitosisOrientationVectorBased(_cell,sin_theta*cos_phi,sin_theta*sin_phi,cos_theta);


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//bool MitosisSteppable::doDirectionalMitosisOrientationVectorBased(CellG* _cell,double _nx, double _ny, double _nz){
//	//will do directional mitosis using division axis/plane passing through center of mass and perpendicular to 
//	// vector (_nx,_ny,_nz)
//
//
//
//	if(!_nx&&!_ny&&!_nz){
//		return false; //orientation vector is 0
//	}
//	Coordinates3D<double> nVec(_nx,_ny,_nz);
//	double norm=sqrt(nVec*nVec);
//	nVec.x/=norm;
//	nVec.y/=norm;
//	nVec.z/=norm;
//
//
//	// // // if (split && on) {
//	//split = false;
//
//	WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
//	//reseting pointers to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
//	childCell=0;
//	parentCell=0;
//
//	// // // CellG *cell = cellField->get(splitPt);//cells that is being divided
//	CellG *cell = _cell;//cells that is being divided        
//
//	//have to make a copy of pixel set before iterating over it. Reason - pixel set will change when we assign pixels to another cell , hence iterators will be invalidated
//	set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
//
//	
//	parentCell=cell;
//	//cerr<<"dividing parent cell "<<cell->id<<endl;
//	double xcm=cell->xCM/(float)cell->volume;
//	double ycm=cell->yCM/(float)cell->volume;
//	double zcm=cell->zCM/(float)cell->volume;
//
//	//first calculate and diagonalize inertia tensor
//
//	//plane/line equation is of the form (r-p)*n=0 where p is vector pointing to point through which the plane will pass (COM)
//	// n is a normal vector to the plane/line
//	// r is (x,y,z) vector
//	//nx*x+ny*y+nz*z-p*n=0 
//	//or nx*x+ny*y+nz*z+d=0 where d is a scalar product -p*n
//	Coordinates3D<double> pVec(xcm,ycm,zcm);
//	double d=-(pVec*nVec);
//
//	//cerr<<"cell->volume="<<cell->volume<<endl;
//	int parentCellVolume=0;
//
//
//	for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//		Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
//		//cerr<<"pt="<<sitr->pixel<<endl;
//		if(nVec.x*pixelTrans.x+nVec.y*pixelTrans.y + nVec.z*pixelTrans.z+d<= 0.0){
//
//			if(!childCell){
//				//cerr<<" creating cell and adding pt="<<sitr->pixel<<endl;
//				childCell = potts->createCellG(sitr->pixel);
//				//cerr<<"childCell="<<childCell<<endl;
//			}else{
//				//cerr<<" adding pt="<<sitr->pixel<<endl;
//				cellField->set(sitr->pixel, childCell);				
//			}			
//		}else{
//
//			parentCellVolume++;
//		}
//
//	}
//	//cerr<<"parentCellVolume="<<parentCellVolume<<endl;
//
//	if(childCell)
//		return true;
//	else
//		return false;
//
//	// // // }
//}
//


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Vector3 MitosisSteppable::calculateCOMPixels(std::set<PixelTrackerData> & _pixels){
	Vector3 com(0.,0.,0.);
	for(set<PixelTrackerData>::iterator sitr=_pixels.begin() ; sitr != _pixels.end() ;++sitr){
		Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
		com.fX+=pixelTrans.x;
		com.fY+=pixelTrans.y;
		com.fZ+=pixelTrans.z;
	}	
	com*=1.0/_pixels.size();
	return com;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CellG * MitosisSteppable::createChildCell(std::set<PixelTrackerData> & _pixels){
	CellG * childCell=0;
	WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	for(set<PixelTrackerData>::iterator sitr=_pixels.begin() ; sitr != _pixels.end() ;++sitr){
		if(!childCell){
			//cerr<<" creating cell and adding pt="<<sitr->pixel<<endl;
			childCell = potts->createCellG(sitr->pixel);
			//cerr<<"childCell="<<childCell<<endl;
		}else{
			//cerr<<" adding pt="<<sitr->pixel<<endl;
			cellField->set(sitr->pixel, childCell);				
		}	
	}
	return childCell;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool MitosisSteppable::doDirectionalMitosisOrientationVectorBased(CellG* _cell,double _nx, double _ny, double _nz){
	//will do directional mitosis using division axis/plane passing through center of mass and perpendicular to 
	// vector (_nx,_ny,_nz)



	if(!_nx&&!_ny&&!_nz){
		return false; //orientation vector is 0
	}
	Coordinates3D<double> nVec(_nx,_ny,_nz);
	double norm=sqrt(nVec*nVec);
	nVec.x/=norm;
	nVec.y/=norm;
	nVec.z/=norm;


	// // // if (split && on) {
	//split = false;

	//WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	//reseting pointers to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
	childCell=0;
	parentCell=0;

	// // // CellG *cell = cellField->get(splitPt);//cells that is being divided
	CellG *cell = _cell;//cells that is being divided        
	//cerr<<"Original cell volume="<<cell->volume<<endl;
	//have to make a copy of pixel set before iterating over it. Reason - pixel set will change when we assign pixels to another cell , hence iterators will be invalidated
	set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
	Vector3 shiftVector(0.,0.,0.);
	set<PixelTrackerData> shiftedPixels;

	set<PixelTrackerData> * pixelsToDividePtr=&cellPixels;

	//cerr<<"dividing parent cell "<<cell->id<<endl;
	double xcm=cell->xCM/(float)cell->volume;
	double ycm=cell->yCM/(float)cell->volume;
	double zcm=cell->zCM/(float)cell->volume;

	//in the case of periodic b.c.'s we shift pixels to the middle of the lattice
	if (boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z){

		shiftVector = getShiftVector(cellPixels);
		shiftCellPixels(cellPixels,shiftedPixels,shiftVector);
		Vector3 com=calculateCOMPixels(shiftedPixels);
		xcm=com.fX;	
		ycm=com.fY;
		zcm=com.fZ;
		pixelsToDividePtr=&shiftedPixels;

		//////cerr<<"shiftVector="<<shiftVector <<" x="<<xcm<<" y="<<ycm<<" z="<<zcm<<endl;



	}

	parentCell=cell;

	//childCell=createChildCell(shiftedPixels);
	//return true;

	//first calculate and diagonalize inertia tensor

	//plane/line equation is of the form (r-p)*n=0 where p is vector pointing to point through which the plane will pass (COM)
	// n is a normal vector to the plane/line
	// r is (x,y,z) vector
	//nx*x+ny*y+nz*z-p*n=0 
	//or nx*x+ny*y+nz*z+d=0 where d is a scalar product -p*n
	Coordinates3D<double> pVec(xcm,ycm,zcm);
	double d=-(pVec*nVec);

	//cerr<<"cell->volume="<<cell->volume<<endl;
	int parentCellVolume=0;

	set<PixelTrackerData> parentPixels;
	set<PixelTrackerData> childPixels;


	bool lessThanFlag;
	if (parentChildPositionFlag<0){
		lessThanFlag=false;
	}else if (parentChildPositionFlag==0){
		lessThanFlag=(randGen.getRatio()<0.5);
	}else{
		lessThanFlag=true;
	}
	//bool lessThanFlag=(randGen.getRatio()<0.5);
	//cerr<<"lessThanFlag="<<lessThanFlag<<endl;

	for(set<PixelTrackerData>::iterator sitr=pixelsToDividePtr->begin() ; sitr != pixelsToDividePtr->end() ;++sitr){
		Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
		//cerr<<"pt="<<sitr->pixel<<endl;
		
		//Randomizing which position of the child/parent cells
		if (lessThanFlag){
			
			if(nVec.x*pixelTrans.x+nVec.y*pixelTrans.y + nVec.z*pixelTrans.z+d<= 0.0){

				childPixels.insert(PixelTrackerData(sitr->pixel));
				//if(!childCell){
				//	//cerr<<" creating cell and adding pt="<<sitr->pixel<<endl;
				//	childCell = potts->createCellG(sitr->pixel);
				//	//cerr<<"childCell="<<childCell<<endl;
				//}else{
				//	//cerr<<" adding pt="<<sitr->pixel<<endl;
				//	cellField->set(sitr->pixel, childCell);				
				//}			
			}else{
				parentPixels.insert(PixelTrackerData(sitr->pixel));
				parentCellVolume++;
				
			}
			
		}else{
			
			if(nVec.x*pixelTrans.x+nVec.y*pixelTrans.y + nVec.z*pixelTrans.z+d> 0.0){

				childPixels.insert(PixelTrackerData(sitr->pixel));
			}else{
				parentPixels.insert(PixelTrackerData(sitr->pixel));
				parentCellVolume++;
			}

		}

	}
	//cerr<<"parentCellVolume="<<parentCellVolume<<endl;
	//now we may have to shift back pixels before assigning them to parent and daughter cells again
	if (boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z){
		set<PixelTrackerData> parentPixelsShifted;
		set<PixelTrackerData> childPixelsShifted;

		shiftCellPixels(childPixels,childPixelsShifted,-1.0*shiftVector);
		shiftCellPixels(parentPixels,parentPixelsShifted,-1.0*shiftVector);

		childCell=createChildCell(childPixelsShifted);

	}else{

		childCell=createChildCell(childPixels);

	}
	//cerr<<"parentCellVolume="<<parentCellVolume<<endl;

	if(childCell)
		return true;
	else
		return false;

	// // // }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool MitosisSteppable::doDirectionalMitosisAlongMajorAxisCompartments(long _clusterId){
	SteppableOrientationVectorsMitosis orientationVectors=getOrientationVectorsMitosisCompartments(_clusterId);
	return doDirectionalMitosisOrientationVectorBasedCompartments(_clusterId,orientationVectors.semiminorVec.fX,orientationVectors.semiminorVec.fY,orientationVectors.semiminorVec.fZ);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool MitosisSteppable::doDirectionalMitosisAlongMinorAxisCompartments(long _clusterId){
	SteppableOrientationVectorsMitosis orientationVectors=getOrientationVectorsMitosisCompartments(_clusterId);
	return doDirectionalMitosisOrientationVectorBasedCompartments(_clusterId,orientationVectors.semimajorVec.fX,orientationVectors.semimajorVec.fY,orientationVectors.semimajorVec.fZ);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool MitosisSteppable::doDirectionalMitosisRandomOrientationCompartments(long _clusterId){

	BasicRandomNumberGenerator *rand = BasicRandomNumberGenerator::getInstance();
	double cos_theta=-1.0+rand->getRatio()*2.0;
	double sin_theta=sqrt(1.0-cos_theta*cos_theta);
	double sin_phi=-1.0+rand->getRatio()*2.0;
	double cos_phi=sqrt(1.0-sin_phi*sin_phi);
	//cerr<<"MITOSIS N VEC="<<sin_theta*cos_phi<<","<<sin_theta*sin_phi<<","<<cos_theta<<endl;
	return doDirectionalMitosisOrientationVectorBasedCompartments(_clusterId,sin_theta*cos_phi,sin_theta*sin_phi,cos_theta);
	
	if (cos_theta==1.0 || cos_theta==0.0)
		return doDirectionalMitosisOrientationVectorBasedCompartments(_clusterId,1,1,1);
	else
		return doDirectionalMitosisOrientationVectorBasedCompartments(_clusterId,sin_theta*cos_phi,sin_theta*sin_phi,cos_theta);

	
}

bool MitosisSteppable::tryAdjustingCompartmentCOM(Vector3 & _com, const set<PixelTrackerData> & _clusterPixels){
	//this function explores if truncation errors may lead to misplaced COM for compartments
	//it might adjust compartment COM for the purpose of conduction mitosis - only

		Point3D pt=Point3D((int)round(_com.fX*xFactor),(int)round(_com.fY*yFactor),(int)round(_com.fZ*zFactor));

		if(_clusterPixels.find(PixelTrackerData(pt))!=_clusterPixels.end()){			
			return true;			
		}
		else if (_clusterPixels.find(PixelTrackerData(Point3D(pt.x-1,pt.y,pt.z)))!=_clusterPixels.end()){
			//notice that to go from cartesian pixel coordinates to , here hex lattice coordinates we use inverse of xFactor , yFactor, or zFactor 
			// this is inverse operation to going from hex lattice to cartesian coordinates
			_com.fX-=1/xFactor;
			return true;

		}else if (_clusterPixels.find(PixelTrackerData(Point3D(pt.x+1,pt.y,pt.z)))!=_clusterPixels.end()){			
			_com.fX+=1/xFactor;
			return true;
		}else if (_clusterPixels.find(PixelTrackerData(Point3D(pt.x,pt.y-1,pt.z)))!=_clusterPixels.end()){			
			_com.fY-=1/yFactor;
			return true;
		}else if (_clusterPixels.find(PixelTrackerData(Point3D(pt.x,pt.y+1,pt.z)))!=_clusterPixels.end()){
			_com.fY+=1/yFactor;
			return true;
		}else if (_clusterPixels.find(PixelTrackerData(Point3D(pt.x,pt.y,pt.z-1)))!=_clusterPixels.end()){
			_com.fZ-=1/zFactor;
			return true;
		}else if (_clusterPixels.find(PixelTrackerData(Point3D(pt.x,pt.y,pt.z+1)))!=_clusterPixels.end()){
			_com.fZ+=1/zFactor;
			return true;
		}

		return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool MitosisSteppable::doDirectionalMitosisOrientationVectorBasedCompartments(long _clusterId,double _nx, double _ny, double _nz){
	//CHECK IF IT WILL WORK WITH PERIODIC BOUNDARY CONDITIONS ALL DISTANCES AND DISPLACEMENT VECTORS MAY NEED TO BE RECALCULATED THEN

	//REMARK: numberOfClusters shold be interpreted as numberOfCompartments 
	CellInventory & inventory=potts->getCellInventory();
	CC3DCellList compartmentsVec=inventory.getClusterCells(_clusterId);
	//////cerr<<"Got the following compartments for cluster id="<<_clusterId<<endl;

	//we use these factors to rescale pixels expressed in absolute coordinates to those of the underlying cartesian lattice - this is done for pixel lookup 
	//because pixels are stored in sets as integers - i.e. in a "cartesian form"

	Vector3 cleavegeVector(_nx,_ny,_nz);

	//have to massage the vector in case of 2D simulation - the vector coordinate corresponding to 'flat' direction is set to zero
	cleavegeVector.fX*=fabs((double)sgn(fieldDim.x-1));
	cleavegeVector.fY*=fabs((double)sgn(fieldDim.y-1));
	cleavegeVector.fZ*=fabs((double)sgn(fieldDim.z-1));
	//cerr<<"cleavegeVector="<<cleavegeVector<<endl;

	int numberOfClusters=compartmentsVec.size();

	comOffsetsMitosis.assign(numberOfClusters,CompartmentMitosisData());
	parentBeforeMitosis.assign(numberOfClusters,CompartmentMitosisData());	
	parentAfterMitosis.assign(numberOfClusters,CompartmentMitosisData());
	childAfterMitosis.assign(numberOfClusters,CompartmentMitosisData());
	vector<int> originalCompartmentVolumeVec(numberOfClusters,0);


	for(int i = 0 ; i < numberOfClusters; ++i){
		CellG *cell=compartmentsVec[i];
		parentBeforeMitosis[i].com=Vector3(cell->xCM/(double)cell->volume,cell->yCM/(double)cell->volume,cell->zCM/(double)cell->volume);
		parentBeforeMitosis[i].cell=cell;
		originalCompartmentVolumeVec[i]=cell->volume;
	}
	


	//////cerr<<"INSIDE DIRECTIONAL MITOSIS FOR CLUSTERS"<<endl;
	//construct a set containing all pixels of the cluster
	set<PixelTrackerData> clusterPixels;		
	for(int i = 0 ; i < numberOfClusters; ++i){
		CellG *cell=compartmentsVec[i];
		set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
		clusterPixels.insert(cellPixels.begin(),cellPixels.end());
	}

	////have to make a copy of pixel set before iterating over it. Reason - pixel set will change when we assign pixels to another cell , hence iterators will be invalidated
	//set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
	Vector3 shiftVector(0.,0.,0.);
	set<PixelTrackerData> shiftedPixels;

	set<PixelTrackerData> * pixelsToDividePtr=&clusterPixels;

	//in the case of periodic b.c.'s we shift pixels to the middle of the lattice
	if (boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z){

		shiftVector = getShiftVector(clusterPixels);
		//shiftVector = Vector3(0.,0.,0.);

		shiftCellPixels(clusterPixels,shiftedPixels,shiftVector);
		Vector3 com=calculateCOMPixels(shiftedPixels);
		//xcm=com.fX;	
		//ycm=com.fY;
		//zcm=com.fZ;
		pixelsToDividePtr=&shiftedPixels;

		//have to update cluster information after the shift takes place 
		for(int i = 0 ; i < numberOfClusters; ++i){
			CellG *cell=compartmentsVec[i];
			set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
			set<PixelTrackerData> cellPixelsShifted;
			shiftCellPixels(cellPixels,cellPixelsShifted,shiftVector);
			parentBeforeMitosis[i].com=calculateCOMPixels(cellPixelsShifted);
		}

		//////cerr<<"shiftVector="<<shiftVector <<endl;

	}

	//divide cluster pixels using mitosis orientation based algorithm
	set<PixelTrackerData> clusterParent;
	set<PixelTrackerData> clusterChild;
	//bool clusterDivideFlag=divideClusterPixelsOrientationVectorBased(clusterPixels,clusterParent,clusterChild,_nx,_ny,_nz);

	//bool clusterDivideFlag=divideClusterPixelsOrientationVectorBased(*pixelsToDividePtr,clusterParent,clusterChild,_nx,_ny,_nz);

	bool clusterDivideFlag=divideClusterPixelsOrientationVectorBased(*pixelsToDividePtr,clusterParent,clusterChild,cleavegeVector.fX,cleavegeVector.fY,cleavegeVector.fZ);
	//for (set<PixelTrackerData>::iterator sitr=clusterParent.begin() ; sitr!=clusterParent.end() ; ++sitr){
	//	cerr<<"parent pixel ="<<sitr->pixel<<endl;
	//}

	//for (set<PixelTrackerData>::iterator sitr=clusterChild.begin() ; sitr!=clusterChild.end() ; ++sitr){
	//	cerr<<"Child pixel ="<<sitr->pixel<<endl;
	//}

	//cerr<<"clusterParent.size()="<<clusterParent.size()<<endl;
	//cerr<<"clusterChild.size()="<<clusterChild.size()<<endl;


	//////cerr<<"clusterChild.size()="<<clusterChild.size()<<" clusterParent.size()="<<clusterParent.size()<<endl;
	//Vector3 clusterCOMBeforeMitosis=calculateClusterPixelsCOM(clusterPixels);

	Vector3 clusterCOMBeforeMitosis=calculateClusterPixelsCOM(*pixelsToDividePtr);

	Vector3 clusterParentCOM=calculateClusterPixelsCOM(clusterParent);
	Vector3 clusterChildCOM=calculateClusterPixelsCOM(clusterChild);
	//////cerr<<"clusterParentCOM="<<clusterParentCOM<<endl;
	//////cerr<<"clusterChildCOM="<<clusterChildCOM<<endl;

	//cerr<<"clusterCOMBeforeMitosis="<<clusterCOMBeforeMitosis<<endl;
	//cerr<<"clusterParentCOM="<<clusterParentCOM<<endl;
	//cerr<<"clusterChildCOM="<<clusterChildCOM<<endl;

	//now will calculate offsets of displacement vectors after mitosis w.r.t COM of the cluster. We shrink original displacements by appropriate factor (2 should be sufficient) 
	//we will use parallel/perpendicular vector decomposition w.r.t nx, ny, nz and scale only parallel component of the offset vector
	//offset_original=parentBeforeMitosis[i].com-referenceVector
	//offset_scaled=offset_original+(offset_original*n)n[scalingFactor-1]
	if(!cleavegeVector.fX&&!cleavegeVector.fY&&!cleavegeVector.fZ){
		return false; //orientation vector is 0
	}
	//Vector3 nVec(_nx,_ny,_nz);
	Vector3 nVec(cleavegeVector);
	
	double norm=nVec.Mag();
	nVec*=1.0/norm;
	//////cerr<<"nVec="<<nVec<<endl;
	double scalingFactor=0.50;
	Vector3 referenceVector=clusterCOMBeforeMitosis;
	for(int i = 0 ; i < numberOfClusters; ++i){

		//comOffsetsMitosis[i].com=(parentBeforeMitosis[i].com-referenceVector)*scalingFactor;			
		Vector3 offsetOriginal=parentBeforeMitosis[i].com-referenceVector;
		comOffsetsMitosis[i].com=offsetOriginal+(offsetOriginal*nVec)*nVec*(scalingFactor-1);
		//////cerr<<"i="<<i<<" offsetOriginal="<<offsetOriginal<<endl;
		//////cerr<<"comOffsetsMitosis[i].com="<<comOffsetsMitosis[i].com<<endl;
	}	


	//now will calculate radii of attractions for compartments of child clusters
	//have to do it before modifying parent cluster
	vector<double> attractionRadiusParent(numberOfClusters,0.0);
	vector<double> attractionRadiusChild(numberOfClusters,0.0);
	double pi=acos(-1.0);
	if(flag3D){
		for(int i = 0 ; i < numberOfClusters; ++i){
			attractionRadiusParent[i]=0.5*pow(3*parentBeforeMitosis[i].cell->volume/pi,1/3.0); //note that we are calculating the radius assuming half volume		
		}
	}else{
		for(int i = 0 ; i < numberOfClusters; ++i){
			attractionRadiusParent[i]=sqrt(parentBeforeMitosis[i].cell->volume/(2*pi)); //note that we are calculating the radius assuming half volume		
		}
	}


	//calculate positions of the com's of cluster compartments after mitosis for parent and child clusters 
	for(int i = 0 ; i < numberOfClusters; ++i){
		parentAfterMitosis[i].com=clusterParentCOM+comOffsetsMitosis[i].com;
		childAfterMitosis[i].com=clusterChildCOM+comOffsetsMitosis[i].com;
		//cerr<<"parentAfterMitosis[i].com="<<parentAfterMitosis[i].com<<endl;
		//cerr<<"childAfterMitosis[i].com="<<childAfterMitosis[i].com<<endl;
	}
	//initialize coms for parent and child cells
	set<PixelTrackerData> parentCellKernelsSet;
	set<PixelTrackerData> childCellKernelsSet;

	//initialize coms for parent and child cells
	vector<CompartmentMitosisData> parentCellKernels;
	vector<CompartmentMitosisData> childCellKernels;

	WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();

	//for (set<PixelTrackerData>::iterator sitr = clusterParent.begin() ; sitr!= clusterParent.end() ; ++sitr){
	//	cerr<<"parentPixel="<<sitr->pixel<<endl;
	//}

	//first check if coms belong to set of cluster pixels
	//cerr<<"xFactor="<<xFactor<<" yFactor="<<yFactor<<" zFactor="<<zFactor<<endl;
	for(int i = 0 ; i < numberOfClusters; ++i){
		//Point3D pt=Point3D(parentAfterMitosis[i].com.fX,parentAfterMitosis[i].com.fY,parentAfterMitosis[i].com.fZ);
		
		//Point3D pt=Point3D(parentAfterMitosis[i].com.fX,parentAfterMitosis[i].com.fY*yFactor,parentAfterMitosis[i].com.fZ*zFactor);

		Point3D pt=Point3D((int)round(parentAfterMitosis[i].com.fX*xFactor),(int)round(parentAfterMitosis[i].com.fY*yFactor),(int)round(parentAfterMitosis[i].com.fZ*zFactor));


		//if(clusterParent.find(PixelTrackerData(pt))==clusterParent.end()){
		if(!tryAdjustingCompartmentCOM(parentAfterMitosis[i].com,clusterParent)){
			//cerr<<"parentAfterMitosis[i].com="<<parentAfterMitosis[i].com<<endl;
			//cerr<< "pt="<<pt<<endl;
	
			//cerr<<"com for compartment "<<i<<" of parent cell does not belong to pixels of the parent cell after division"<<endl;
			return false;

		}

		//pt=Point3D(childAfterMitosis[i].com.fX,childAfterMitosis[i].com.fY,childAfterMitosis[i].com.fZ);


		//pt=Point3D(childAfterMitosis[i].com.fX,childAfterMitosis[i].com.fY*yFactor,childAfterMitosis[i].com.fZ*zFactor);
		pt=Point3D((int)round(childAfterMitosis[i].com.fX*xFactor),(int)round(childAfterMitosis[i].com.fY*yFactor),(int)round(childAfterMitosis[i].com.fZ*zFactor));
		if(!tryAdjustingCompartmentCOM(childAfterMitosis[i].com,clusterChild)){
		//if(clusterChild.find(PixelTrackerData(pt))==clusterChild.end()){

			//cerr<<"childAfterMitosis[i].com="<<childAfterMitosis[i].com<<endl;
			//cerr<< "pt="<<pt<<endl;

			//cerr<<"com for compartment "<<i<<" of child cell does not belong to pixels of the child cell after division"<<endl;
			return false;

		}

	}

	long childClusterId=0;
	for(int i = 0 ; i < numberOfClusters; ++i){
		Point3D pt;	

		CompartmentMitosisData parentCMD;

		//pt=Point3D(parentAfterMitosis[i].com.fX,parentAfterMitosis[i].com.fY,parentAfterMitosis[i].com.fZ);
		pt=Point3D(parentAfterMitosis[i].com.fX*xFactor,parentAfterMitosis[i].com.fY*yFactor,parentAfterMitosis[i].com.fZ*zFactor);
		//CellG * newCell1 = potts->createCellG(pt);
		//newCell1->type=6; 
		
		//cellField->set(pt,parentBeforeMitosis[i].cell);
		//parentCellKernels.insert(PixelTrackerData(pt));
		
		parentCMD.cell=parentBeforeMitosis[i].cell;		
		//////cerr<<"i="<<i<<" parentBeforeMitosis[i].cell.type="<<(int)parentBeforeMitosis[i].cell->type<<endl;
		parentCMD.pt=pt;
		parentCellKernels.push_back(parentCMD);
		parentCellKernelsSet.insert(PixelTrackerData(pt));
		//parentBeforeMitosis[i].cell->type=6; 

		CompartmentMitosisData childCMD;

		//pt=Point3D(childAfterMitosis[i].com.fX,childAfterMitosis[i].com.fY,childAfterMitosis[i].com.fZ);
		pt=Point3D(childAfterMitosis[i].com.fX*xFactor,childAfterMitosis[i].com.fY*yFactor,childAfterMitosis[i].com.fZ*zFactor);
		

		childCMD.type=parentBeforeMitosis[i].cell->type;
		
		//////cerr<<"i="<<i<<" childCMD.type="<<(int)childCMD.type<<endl;
		childCMD.pt=pt;

		childCellKernels.push_back(childCMD);

		//CellG * childCompartmentCell = potts->createCellG(pt,potts->getRecentlyCreatedClusterId()+1);//have to initialize cluster id to be greater than any pther cluster id in the inventory
		//childCompartmentCell->type=parentBeforeMitosis[i].cell->type;
		////childCompartmentCell->type=6; 
		//if (i==0){
		//	childClusterId=childCompartmentCell->clusterId;
		//}else{//have to reassign cluster id for compartments of child cluster
		//	potts->getCellInventory().reassignClusterId(childCompartmentCell,childClusterId);			
		//}

		////cellField->set(pt,parentBeforeMitosis[i].cell);
		//childCellKernels.insert(PixelTrackerData(pt));

		////parentBeforeMitosis[i].cell->type=6; 

	}


	//now will reset all the pixels except of kernel pixels to be medium
	//for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr!=clusterPixels.end() ; ++sitr){
	//	if(parentCellKernelsSet.find(*sitr)==parentCellKernelsSet.end()){
	//		cellField->set(sitr->pixel,0);
	//	}
	//}
	//now we fill child and parent cells with pixels following the algorothm based on pixel distances and attraction radii
	//parent hasto be initialized first otherwise we may loose some information about original cells if any of them will be overwritted during child assignment
	
	initializeClusters(originalCompartmentVolumeVec , clusterParent ,parentCellKernels,attractionRadiusParent,parentBeforeMitosis,shiftVector);
	//getting parent cell is different in the case of periodic and non periodic b.c.'s
	if (boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z){
		set<PixelTrackerData> originalSinglePixelSet;
		set<PixelTrackerData> shiftedSinglePixelSet;
		originalSinglePixelSet.insert(PixelTrackerData(parentCellKernels[0].pt));
		shiftCellPixels(originalSinglePixelSet,shiftedSinglePixelSet,-1.0*shiftVector) ; //we are shifting back so shift Vec is multiplied by -1
		parentCell=cellField->get(shiftedSinglePixelSet.begin()->pixel); //we assign parent cell to be first cell of the cluster 
		//////cerr<<"\n\n PBC parentCell->type="<<(int)parentCell->type<<endl;
	}else{
		parentCell=cellField->get(parentCellKernels[0].pt);//we assign parent cell to be first cell of the cluster 
		//////cerr<<"\n\n NON PBC parentCell->type="<<(int)parentCell->type<<endl;
	}
	
	//cerr<<"parentClusterId="<<parentCell->clusterId<<endl;
	


	//childCell=parentCell;
	//child
	initializeClusters(originalCompartmentVolumeVec, clusterChild ,childCellKernels,attractionRadiusChild,parentBeforeMitosis,shiftVector);  
	if (boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z){
		set<PixelTrackerData> originalSinglePixelSet;
		set<PixelTrackerData> shiftedSinglePixelSet;
		originalSinglePixelSet.insert(PixelTrackerData(childCellKernels[0].pt));
		shiftCellPixels(originalSinglePixelSet,shiftedSinglePixelSet,-1.0*shiftVector) ; //we are shifting back so shift Vec is multiplied by -1
		childCell=cellField->get(shiftedSinglePixelSet.begin()->pixel); //we assign parent cell to be first cell of the cluster 
		//////cerr<<"\n\n PBC childCell->type="<<(int)childCell->type<<endl;
	}else{
		childCell=cellField->get(childCellKernels[0].pt);//we assign parent cell to be first cell of the cluster 
		//////cerr<<"\n\n NON PBC childCell->type="<<(int)childCell->type<<endl;
	}

	//////cerr<<"childCell="<<childCell<<" parentCell="<<parentCell<<endl;

	//cerr<<"childClusterId="<<childCell->clusterId<<endl;

	//for (int i = 0 ; i < compartmentsVec.size() ; ++i){
	//	cerr<<"Compartment is="<<compartmentsVec[i]->id<<" type="<<(int)compartmentsVec[i]->type<<" volume="<<compartmentsVec[i]->volume<<endl;
	//}

	return true;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MitosisSteppable::initializeClusters(std::vector<int> & originalCompartmentVolumeVec , std::set<PixelTrackerData> & clusterPixels,std::vector<CompartmentMitosisData> &clusterKernels,std::vector<double> & attractionRadiusVec,std::vector<CompartmentMitosisData> & parentBeforeMitosisCMDVec,Vector3 shiftVec){

	//vector<CellG*> kernelCellVec(clusterKernels.size(),0);
	//////cerr<<"clusterPixels.size()="<<clusterPixels.size()<<endl;
	WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
	////getting pointers to cells at kernel points
	//int counter=0;
	//for(set<PixelTrackerData>::iterator sitr=clusterKernels.begin() ; sitr!=clusterKernels.end() ; ++sitr){
	//	kernelCellVec[counter]=cellField->get(sitr->pixel);
	//	cerr<<"kernelCellVec["<<counter<<"]="<<kernelCellVec[counter]<<endl;
	//	if(kernelCellVec[counter]){
	//		cerr<<"cell type="<<(int)kernelCellVec[counter]->type<<endl;
	//	}
	//	++counter;
	//}

	//set<CellG*> kernelCellSet(kernelCellVec.begin(),kernelCellVec.end()); 

	vector<set<PixelTrackerData> > compartmentPixelsVector(clusterKernels.size()); //those sets will hold pixels belonging to a given compartment

	vector<CompartmentMitosisData> clusterKernelsVec(clusterKernels.begin(),clusterKernels.end());

	map<double,int> pixel2KernelDistMap;	
	map<double,int> pixelWithinAttractionRadiusMap;	
	double dist;
	Point3D pixel;
	bool pixelAssignedFlag;
	for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr!=clusterPixels.end() ; ++sitr){
		pixel2KernelDistMap.clear();
		pixelWithinAttractionRadiusMap.clear();
		pixelAssignedFlag=false;
		pixel=sitr->pixel;

		for (int i = 0 ; i < clusterKernelsVec.size() ; ++i){			
			Point3D kernel=clusterKernelsVec[i].pt;
			dist=distInvariantCM(pixel.x,pixel.y,pixel.z,kernel.x,kernel.y,kernel.z,fieldDim,boundaryStrategy);
			pixel2KernelDistMap.insert(make_pair(dist,i));
			if(dist<attractionRadiusVec[i]){
				pixelWithinAttractionRadiusMap.insert(make_pair(dist,i));
			}
		}
		//we know the kernels for which pixel is within the attraction radius and we know which kernel is closest to the pixel. 
		//We first try to assign pixel to the cell whose kernel is within attraction radius
		for(map<double,int>::iterator mitr=pixelWithinAttractionRadiusMap.begin() ; mitr!=pixelWithinAttractionRadiusMap.end() ; ++mitr){
			if (compartmentPixelsVector[mitr->second].size() < originalCompartmentVolumeVec[mitr->second]/2){
				compartmentPixelsVector[mitr->second].insert(pixel);
				//cellField->set(pixel,kernelCellVec[mitr->second]);				
				pixelAssignedFlag=true;
				break;
			}
		}

		if(pixelAssignedFlag)
			continue;

		//next if pixel is not within attraction radius of any kernel or if it is but such cells cannot accept more pixels we assign the pixel to the cell 
		//whose kernel is the closest 		
		for(map<double,int>::iterator mitr=pixel2KernelDistMap.begin() ; mitr!=pixel2KernelDistMap.end() ; ++mitr){
			//if (kernelCellVec[mitr->second]->volume < originalCompartmentVolumeVec[mitr->second]/2){
				compartmentPixelsVector[mitr->second].insert(pixel);
				//cellField->set(pixel,kernelCellVec[mitr->second]);
				pixelAssignedFlag=true;
				break;
			//}

		}

		//if(pixelAssignedFlag)
		//	continue;

		////finally we assign reminder pixels (if any) the the closest kernel
		//cerr<<"assign pixel="<<pixel<<" to cluster="<<pixel2KernelDistMap.begin()->second<<endl;
		//cellField->set(pixel,kernelCellVec[pixel2KernelDistMap.begin()->second]);
	}



	//Now we will create cells based on compartmentPixelsVector
	// first we will deal with cluster kernels
	long childClusterId=0; //holder for cluster id of child cell 
	vector<CellG *> kernelCellVec(clusterKernels.size(),0); //holds pointers to cells at kernel pixels
	for (int i = 0 ; i < clusterKernels.size() ; ++i){
		//set<PixelTrackerData> shiftedCompartmentPixels;
		//shiftCellPixels(compartmentPixelsVector[i],shiftedCompartmentPixels,-1.0*shiftVec) ; //we are shifting back so shift Vec is multiplied by -1

		if(clusterKernels[i].cell){ //this is parent cell because cell ptr  in CMD is non zero
			set<PixelTrackerData> originalSinglePixelSet;
			set<PixelTrackerData> shiftedSinglePixelSet;

			originalSinglePixelSet.insert(PixelTrackerData(clusterKernels[i].pt));
			shiftCellPixels(originalSinglePixelSet,shiftedSinglePixelSet,-1.0*shiftVec) ; //we are shifting back so shift Vec is multiplied by -1
			cellField->set(shiftedSinglePixelSet.begin()->pixel,parentBeforeMitosisCMDVec[i].cell);
			kernelCellVec[i] = parentBeforeMitosisCMDVec[i].cell;
		}else{ //dealing with child cell
			set<PixelTrackerData> originalSinglePixelSet;
			set<PixelTrackerData> shiftedSinglePixelSet;	
			originalSinglePixelSet.insert(PixelTrackerData(clusterKernels[i].pt));
			shiftCellPixels(originalSinglePixelSet,shiftedSinglePixelSet,-1.0*shiftVec) ; //we are shifting back so shift Vec is multiplied by -1
            // cerr<<"potts->getRecentlyCreatedClusterId()="<<potts->getRecentlyCreatedClusterId()<<endl;
            //have to initialize cluster id to be greater than any pther cluster id in the inventory            
            if (!childClusterId){
                childClusterId=potts->getRecentlyCreatedClusterId()+1;
            }
            
			// CellG * childCompartmentCell = potts->createCellG(shiftedSinglePixelSet.begin()->pixel,potts->getRecentlyCreatedClusterId()+1);//have to initialize cluster id to be greater than any pther cluster id in the inventory
			CellG * childCompartmentCell = potts->createCellG(shiftedSinglePixelSet.begin()->pixel,childClusterId);
            //////cerr<<"childCompartmentCell->id="<<childCompartmentCell->id<<endl;
			//////cerr<<" child lernel pixel "<<shiftedSinglePixelSet.begin()->pixel<<" childCell="<<childCompartmentCell<<endl;
			//////cerr<<"i="<<i<<" kernelCellVec.size()="<<kernelCellVec.size()<<endl;
			kernelCellVec[i]=childCompartmentCell;

			//////cerr<<"i="<<i<<"kernelCellVec[i]="<<kernelCellVec[i]<<endl;
			//////cerr<<"parentBeforeMitosisCMDVec[i].cell->type="<<(int)parentBeforeMitosisCMDVec[i].cell->type<<endl;
			childCompartmentCell->type=parentBeforeMitosisCMDVec[i].cell->type;
			//////cerr<<"type="<<childCompartmentCell->type<<endl;
			
            //have to use this trick to circumvent default cluster id assignment algorithm in Potts3D - > createCellG function
            // potts->getCellInventory().reassignClusterId(childCompartmentCell,childClusterId+1);			
            
			// if (i==0){
				// childClusterId=childCompartmentCell->clusterId;
			// }else{//have to reassign cluster id for compartments of child cluster
				// potts->getCellInventory().reassignClusterId(childCompartmentCell,childClusterId);			
			// }

		}
	}
	//at this point kernels have been initialized now we have to deal with the rest of pixels
	for (int i = 0 ; i < compartmentPixelsVector.size() ; ++i){
		set<PixelTrackerData> &comparmentPixels=compartmentPixelsVector[i];
		set<PixelTrackerData> comparmentPixelsShifted;
		shiftCellPixels(comparmentPixels,comparmentPixelsShifted,-1.0*shiftVec) ; //we are shifting back so shift Vec is multiplied by -1
		for (set<PixelTrackerData>::iterator sitr=comparmentPixelsShifted.begin() ; sitr != comparmentPixelsShifted.end() ; ++sitr){
			cellField->set(sitr->pixel,kernelCellVec[i]);
		}
	}
	
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool  MitosisSteppable::doDirectionalMitosisOrientationVectorBasedCompartments(CellG* _cell,double _nx, double _ny, double _nz){
	return doDirectionalMitosisOrientationVectorBasedCompartments(_cell->clusterId, _nx,  _ny,  _nz);

	//CellInventory & inventory=potts->getCellInventory();
	//long clusterId=_cell->clusterId;
	//CC3DCellList compartmentsVec=inventory.getClusterCells(clusterId);
	//cerr<<"Got the following compartments for cluster id="<<clusterId<<endl;
	//for (int i = 0 ; i < compartmentsVec.size() ; ++i){
	//	cerr<<"Compartment is="<<compartmentsVec[i]->id<<" type="<<compartmentsVec[i]->type<<" volume="<<compartmentsVec[i]->volume<<endl;
	//	
	//}
	//return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Vector3 MitosisSteppable::calculateClusterPixelsCOM(set<PixelTrackerData> & clusterPixels){

	Vector3 clusterCOM;

	if ( !boundaryConditionIndicator.x && !boundaryConditionIndicator.y && !boundaryConditionIndicator.z ){
		for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr!=clusterPixels.end() ; ++sitr){
			Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(sitr->pixel); 
			clusterCOM.fX+=ptTrans.x;			
			clusterCOM.fY+=ptTrans.y;			
			clusterCOM.fZ+=ptTrans.z;			
		}
		return clusterCOM*(1.0/(float)clusterPixels.size());
	}

	//calculating COM for periodic boundary conditions
	Coordinates3D<double> shiftVec;
	Coordinates3D<double> shiftedPt;
	Coordinates3D<double> distanceVecMin;
	//determines minimum coordinates for the perpendicular lines paccinig through pt
	Coordinates3D<double> distanceVecMax;
	Coordinates3D<double> distanceVecMax_1;
	//determines minimum coordinates for the perpendicular lines paccinig through pt
	Coordinates3D<double> distanceVec; //measures lattice diatances along x,y,z - they can be different for different lattices. The lines have to pass through pt

	//    cerr<<"distanceVec="<<distanceVec<<" distanceVecMin="<<distanceVecMin<<" distanceVecMax="<<distanceVecMax<<endl;


	Coordinates3D<double> fieldDimTrans= boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,fieldDim.y-1,fieldDim.z-1));
	//cerr<<"fieldDimTrans="<<fieldDimTrans<<endl;



	double xCM,yCM,zCM; //temporary centroids

	//double x,y,z;
	//double xo,yo,zo;



	int numberOfVisitedPixels=0;
	for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr!=clusterPixels.end() ; ++sitr){
		++numberOfVisitedPixels; //equivalent of updating a volume of a growing cell - as in the COM plugin the volume is updated before COM calculations

		Point3D pt=sitr->pixel;

		Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt); 
		//cerr<<"ptTrans="<<ptTrans<<endl;

		//if there are boundary conditions defined that we have to do some shifts to correctly calculate center of mass
		//This approach will work only for cells whose span is much smaller that lattice dimension in the "periodic "direction
		//e.g. cell that is very long and "wraps lattice" will have miscalculated CM using this algorithm. On the other hand, you do not really expect
		//cells to have dimensions comparable to lattice...

		distanceVecMin.x=boundaryStrategy->calculatePointCoordinates(Point3D(0,pt.y,pt.z)).x;
		distanceVecMin.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,0,pt.z)).y;
		distanceVecMin.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,0)).z;

		distanceVecMax.x=boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x,pt.y,pt.z)).x;
		distanceVecMax.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,fieldDim.y,pt.z)).y;
		distanceVecMax.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,fieldDim.z)).z;

		distanceVecMax_1.x=boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,pt.y,pt.z)).x;
		distanceVecMax_1.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,fieldDim.y-1,pt.z)).y;
		distanceVecMax_1.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,fieldDim.z-1)).z;

		distanceVec=distanceVecMax-distanceVecMin;

		//xo=newCell->xCM;
		//yo=newCell->yCM;
		//zo=newCell->zCM;


		//    x=newCell->xCM+pt.x;
		//    y=newCell->yCM+pt.y;
		//    z=newCell->zCM+pt.z;


		if(numberOfVisitedPixels==1){
			shiftVec.x=0;
			shiftVec.y=0;
			shiftVec.z=0;      
		}else{
			shiftVec.x= (clusterCOM.fX/(numberOfVisitedPixels-1)-((int)fieldDimTrans.x)/2)*boundaryConditionIndicator.x;
			shiftVec.y= (clusterCOM.fY/(numberOfVisitedPixels-1)-((int)fieldDimTrans.y)/2)*boundaryConditionIndicator.y;
			shiftVec.z= (clusterCOM.fZ/(numberOfVisitedPixels-1)-((int)fieldDimTrans.z)/2)*boundaryConditionIndicator.z;      
		}


		//shift CM to approximately center of lattice , new centroids are:
		xCM = clusterCOM.fX - shiftVec.x*(numberOfVisitedPixels-1);
		yCM = clusterCOM.fY - shiftVec.y*(numberOfVisitedPixels-1);
		zCM = clusterCOM.fZ - shiftVec.z*(numberOfVisitedPixels-1);
		//cerr<<"clusterCOM="<<clusterCOM<<endl;
		
		//Now shift pt
		shiftedPt=ptTrans;
		shiftedPt-=shiftVec;


		//making sure that shifted point is in the lattice
		if(shiftedPt.x < distanceVecMin.x){
			shiftedPt.x += distanceVec.x;
		}else if (shiftedPt.x > distanceVecMax_1.x){
			//       cerr<<"shifted pt="<<shiftedPt<<endl;
			shiftedPt.x -= distanceVec.x;
		}  

		if(shiftedPt.y < distanceVecMin.y){
			shiftedPt.y += distanceVec.y;
		}else if (shiftedPt.y > distanceVecMax_1.y){
			shiftedPt.y -= distanceVec.y;
		}  

		if(shiftedPt.z < distanceVecMin.z){
			shiftedPt.z += distanceVec.z;
		}else if (shiftedPt.z > distanceVecMax_1.z){
			shiftedPt.z -= distanceVec.z;
		}    

		//update shifted centroids
		xCM += shiftedPt.x;
		yCM += shiftedPt.y;
		zCM += shiftedPt.z;

		//shift back centroids
		xCM += shiftVec.x * numberOfVisitedPixels;
		yCM += shiftVec.y * numberOfVisitedPixels;
		zCM += shiftVec.z * numberOfVisitedPixels;

		//Check if CM is in the lattice
		if( xCM/(float)numberOfVisitedPixels < distanceVecMin.x){
			xCM += distanceVec.x*numberOfVisitedPixels;
		}else if ( xCM/(float)numberOfVisitedPixels > distanceVecMax.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
			//to avoid rollovers for unsigned int from oldCell->xCM
			xCM -= distanceVec.x*numberOfVisitedPixels;
		}

		if( yCM/(float)numberOfVisitedPixels < distanceVecMin.y){
			yCM += distanceVec.y*numberOfVisitedPixels;
		}else if ( yCM/(float)numberOfVisitedPixels > distanceVecMax.y){
			yCM -= distanceVec.y*numberOfVisitedPixels;
		}

		if( zCM/(float)numberOfVisitedPixels < distanceVecMin.z){
			zCM += distanceVec.z*numberOfVisitedPixels;
		}else if ( zCM/(float)numberOfVisitedPixels > distanceVecMax.z){
			zCM -= distanceVec.z*numberOfVisitedPixels;
		}

		clusterCOM.fX=xCM;			
		clusterCOM.fY=yCM;			
		clusterCOM.fZ=zCM;			
		//cerr<<"xCM="<<xCM<<" yCM="<<yCM<<" zCM="<<zCM<<endl;
	}

	return clusterCOM*(1.0/(float)numberOfVisitedPixels);


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool MitosisSteppable::divideClusterPixelsOrientationVectorBased(set<PixelTrackerData> & clusterPixels , set<PixelTrackerData> & clusterParent , set<PixelTrackerData> & clusterChild, double _nx, double _ny, double _nz){
	if(!_nx&&!_ny&&!_nz){
		return false; //orientation vector is 0
	}
	Vector3 nVec(_nx,_ny,_nz);
	double norm=nVec.Mag();
	nVec*=1.0/norm;

	Vector3 clusterCOM=calculateClusterPixelsCOM(clusterPixels);
	//////cerr<<"INSIDE DIVIDE CLUSTER COMPARTMENTS"<<endl;
	//////cerr<<"clusterCOM="<<clusterCOM<<endl;

	//plane/line equation is of the form (r-p)*n=0 where p is vector pointing to point through which the plane will pass (COM)
	// n is a normal vector to the plane/line
	// r is (x,y,z) vector
	//nx*x+ny*y+nz*z-p*n=0 
	//or nx*x+ny*y+nz*z+d=0 where d is a scalar product -p*n

	Vector3 pVec(clusterCOM);
	double d=-(pVec*nVec);


	int parentCellVolume=0;
	bool lessThanFlag;
	if (parentChildPositionFlag<0){
		lessThanFlag=false;
	}else if (parentChildPositionFlag==0){
		lessThanFlag=(randGen.getRatio()<0.5);
	}else{
		lessThanFlag=true;
	}
	//bool lessThanFlag=(randGen.getRatio()<0.5);

	for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr != clusterPixels.end() ;++sitr){
		Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);

		//randomizing position of the parent/child cluster
		if(lessThanFlag){ 
			if(nVec.fX*pixelTrans.x+nVec.fY*pixelTrans.y + nVec.fZ*pixelTrans.z+d<= 0.0){
				clusterChild.insert(*sitr);
			}else{
				clusterParent.insert(*sitr);

			}
		}else{
			if(nVec.fX*pixelTrans.x+nVec.fY*pixelTrans.y + nVec.fZ*pixelTrans.z+d>0.0){
				clusterChild.insert(*sitr);
			}else{
				clusterParent.insert(*sitr);

			}

		}

	}


	if(clusterChild.size())
		return true;
	else
		return false;


}

// // // bool MitosisSteppable::doDirectionalMitosis(){


// // // if(flag3D){

// // // return doDirectionalMitosis3D();
// // // }
// // // else{

// // // return (this->*doDirectionalMitosis2DPtr)();
// // // }
// // // }

// // // OrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis(CellG * _cell){
// // // if(flag3D){

// // // return getOrientationVectorsMitosis3D(_cell);
// // // }
// // // else{
// // // return (this->*getOrientationVectorsMitosis2DPtr)(_cell);
// // // }
// // // }

// // // OrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis2D_xy(CellG * cell){
// // // double xcm=cell->xCM/(float)cell->volume;
// // // double ycm=cell->yCM/(float)cell->volume;
// // // double zcm=cell->zCM/(float)cell->volume;

// // // //first calculate and diagonalize inertia tensor
// // // vector<vector<double> > inertiaTensor(2,vector<double>(2,0.0));


// // // set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // //cerr<<"point cell="<<sitr->pixel<<endl;
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // inertiaTensor[0][0]+=(pixelTrans.y-ycm)*(pixelTrans.y-ycm);
// // // inertiaTensor[0][1]+=-(pixelTrans.x-xcm)*(pixelTrans.y-ycm);
// // // inertiaTensor[1][1]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm);
// // // }
// // // inertiaTensor[1][0]=inertiaTensor[0][1];

// // // double radical=0.5*sqrt((inertiaTensor[0][0]-inertiaTensor[1][1])*(inertiaTensor[0][0]-inertiaTensor[1][1])+4.0*inertiaTensor[0][1]*inertiaTensor[0][1]);			
// // // double lMin=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])-radical;
// // // double lMax=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])+radical;

// // // //orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
// // // Coordinates3D<double> orientationVec;
// // // if(inertiaTensor[0][1]!=0.0){

// // // orientationVec=Coordinates3D<double>(inertiaTensor[0][1],lMax-inertiaTensor[0][0],0.0);
// // // double length=sqrt(orientationVec.x*orientationVec.x+orientationVec.y*orientationVec.y+orientationVec.z*orientationVec.z);
// // // orientationVec.x/=length;
// // // orientationVec.y/=length;
// // // orientationVec.z/=length;
// // // }else{
// // // if(inertiaTensor[0][0]>inertiaTensor[1][1])
// // // orientationVec=Coordinates3D<double>(0.0,1.0,0.0);
// // // else
// // // orientationVec=Coordinates3D<double>(1.0,0.0,0.0);
// // // }


// // // OrientationVectorsMitosis orientationVectorsMitosis;
// // // orientationVectorsMitosis.semiminorVec=orientationVec;
// // // orientationVectorsMitosis.semimajorVec.x=-orientationVectorsMitosis.semiminorVec.y;
// // // orientationVectorsMitosis.semimajorVec.y=orientationVectorsMitosis.semiminorVec.x;

// // // return orientationVectorsMitosis;
// // // }

// // // bool MitosisSteppable::doDirectionalMitosis2D_xy(){

// // // //this implementation is valid in 2D only
// // // if (split && on) {
// // // split = false;

// // // WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
// // // //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
// // // childCell=0;
// // // parentCell=0;

// // // CellG *cell = cellField->get(splitPt);//cells that is being divided
// // // parentCell=cell;

// // // double xcm=cell->xCM/(float)cell->volume;
// // // double ycm=cell->yCM/(float)cell->volume;
// // // double zcm=cell->zCM/(float)cell->volume;

// // // set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;

// // // OrientationVectorsMitosis orientationVectorsMitosis=getOrientationVectorsMitosis2D_xy(cell);
// // // Coordinates3D<double> orientationVec=orientationVectorsMitosis.semiminorVec;

// // // // assume the following form of the equation of the straight line passing through COM (xcm,ycm,zcm) of cell being divided
// // // //y=a*x+b;
// // // double a;
// // // double b;
// // // //determining coefficients of the straight line passing through
// // // if(divideAlongMinorAxisFlag){
// // // //cerr<<"doing minor axis division"<<endl;


// // // a=orientationVec.y/orientationVec.x;
// // // b=ycm-xcm*a;

// // // if(a!=a){//a is Nan - will happen when orientationVec.x is 0.0 thus minor axis is along y axis
// // // a=0.0;
// // // b=0.0;
// // // }
// // // }

// // // if(divideAlongMajorAxisFlag){
// // // //cerr<<"doing major axis division"<<endl;
// // // if(orientationVec.y==0.0){//then perpendicular vector (major axis) is along y axis meaning:
// // // a=0.0;
// // // b=0.0;

// // // } else{
// // // a=-orientationVec.x/orientationVec.y;
// // // b=ycm-xcm*a;
// // // }
// // // }

// // // //cerr<<"a="<<a<<" b="<<b<<endl;
// // // //now do the division

// // // if(a==0.0 && b==0.0){//division along y axis

// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // if(pixelTrans.x<xcm){
// // // if(!childCell){
// // // childCell = potts->createCellG(sitr->pixel);
// // // }else{
// // // cellField->set(sitr->pixel, childCell);
// // // }
// // // }
// // // }

// // // }else{//division will be done along axis different than y axis
// // // //cerr<<"before iteration"<<endl;
// // // //cerr<<"cell->volume="<<cell->volume<<endl;
// // // int parentCellVolume=0;
// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // //cerr<<"pt="<<sitr->pixel<<endl;
// // // if(pixelTrans.y <= a*pixelTrans.x+b){

// // // if(!childCell){
// // // //cerr<<" creating cell and adding pt="<<sitr->pixel<<endl;
// // // childCell = potts->createCellG(sitr->pixel);
// // // //cerr<<"childCell="<<childCell<<endl;
// // // }else{
// // // //cerr<<" adding pt="<<sitr->pixel<<endl;
// // // cellField->set(sitr->pixel, childCell);
// // // }
// // // }else{

// // // parentCellVolume++;
// // // }

// // // }
// // // //cerr<<"parentCellVolume="<<parentCellVolume<<endl;
// // // }

// // // //if childCell was created this means mitosis was sucessful. If child cell was not created there was no mitosis
// // // if(childCell)
// // // return true;
// // // else
// // // return false;
// // // }
// // // }






// // // OrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis2D_xz(CellG * cell){
// // // double xcm=cell->xCM/(float)cell->volume;
// // // double ycm=cell->yCM/(float)cell->volume;
// // // double zcm=cell->zCM/(float)cell->volume;

// // // //first calculate and diagonalize inertia tensor
// // // vector<vector<double> > inertiaTensor(2,vector<double>(2,0.0));

// // // set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // //cerr<<"point cell="<<sitr->pixel<<endl;
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // inertiaTensor[0][0]+=(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
// // // inertiaTensor[0][1]+=-(pixelTrans.x-xcm)*(pixelTrans.z-zcm);
// // // inertiaTensor[1][1]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm);
// // // }
// // // inertiaTensor[1][0]=inertiaTensor[0][1];

// // // double radical=0.5*sqrt((inertiaTensor[0][0]-inertiaTensor[1][1])*(inertiaTensor[0][0]-inertiaTensor[1][1])+4.0*inertiaTensor[0][1]*inertiaTensor[0][1]);			
// // // double lMin=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])-radical;
// // // double lMax=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])+radical;

// // // //orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
// // // Coordinates3D<double> orientationVec;
// // // if(inertiaTensor[0][1]!=0.0){

// // // orientationVec=Coordinates3D<double>(inertiaTensor[0][1],0.0,lMax-inertiaTensor[0][0]);
// // // double length=sqrt(orientationVec.x*orientationVec.x+orientationVec.y*orientationVec.y+orientationVec.z*orientationVec.z);
// // // orientationVec.x/=length;
// // // orientationVec.y/=length;
// // // orientationVec.z/=length;
// // // }else{
// // // if(inertiaTensor[0][0]>inertiaTensor[1][1])
// // // orientationVec=Coordinates3D<double>(0.0,0.0,1.0);
// // // else
// // // orientationVec=Coordinates3D<double>(1.0,0.0,0.0);
// // // }

// // // OrientationVectorsMitosis orientationVectorsMitosis;
// // // orientationVectorsMitosis.semiminorVec=orientationVec;
// // // orientationVectorsMitosis.semimajorVec.x=-orientationVectorsMitosis.semiminorVec.z;
// // // orientationVectorsMitosis.semimajorVec.z=orientationVectorsMitosis.semiminorVec.x;

// // // return orientationVectorsMitosis;


// // // }

// // // bool MitosisSteppable::doDirectionalMitosis2D_xz(){
// // // //cerr<<"DOING XZ mitosis"<<endl;
// // // //this implementation is valid in 2D only
// // // if (split && on) {
// // // split = false;

// // // WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
// // // //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
// // // childCell=0;
// // // parentCell=0;

// // // CellG *cell = cellField->get(splitPt);//cells that is being divided
// // // parentCell=cell;
// // // //cerr<<"dividing parent cell "<<cell->id<<endl;
// // // double xcm=cell->xCM/(float)cell->volume;
// // // double ycm=cell->yCM/(float)cell->volume;
// // // double zcm=cell->zCM/(float)cell->volume;

// // // set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;

// // // OrientationVectorsMitosis orientationVectorsMitosis=getOrientationVectorsMitosis2D_xz(cell);
// // // Coordinates3D<double> orientationVec=orientationVectorsMitosis.semiminorVec;


// // // //  cerr<<"orientationVec="<<orientationVec<<endl;
// // // //cerr<<"inertiaTensor[0][0]="<<inertiaTensor[0][0]<<endl;
// // // //cerr<<"inertiaTensor[0][1]="<<inertiaTensor[0][1]<<endl;
// // // //cerr<<"inertiaTensor[1][1]="<<inertiaTensor[1][1]<<endl;

// // // //cerr<<"xcm="<<xcm<<" zcm="<<zcm<<endl;

// // // //once we know orientation vector corresponding to bigger eigenvalue (pointing along semiminor axis) we may divide cell 
// // // //along major or minor axis

// // // // assume the following form of the equation of the straight line passing through COM (xcm,ycm,zcm) of cell being divided
// // // //z=a*x+b;
// // // double a;
// // // double b;
// // // //determining coefficients of the straight line passing through
// // // if(divideAlongMinorAxisFlag){
// // // //cerr<<"doing minor axis division"<<endl;


// // // a=orientationVec.z/orientationVec.x;
// // // b=zcm-xcm*a;

// // // if(a!=a){//a is Nan - will happen when orientationVec.x is 0.0 thus minor axis is along y axis
// // // a=0.0;
// // // b=0.0;
// // // }
// // // }

// // // if(divideAlongMajorAxisFlag){
// // // //cerr<<"doing major axis division"<<endl;
// // // if(orientationVec.z==0.0){//then perpendicular vector (major axis) is along z axis meaning:
// // // a=0.0;
// // // b=0.0;

// // // } else{
// // // a=-orientationVec.x/orientationVec.z;
// // // b=zcm-xcm*a;
// // // }
// // // }

// // // //cerr<<"a="<<a<<" b="<<b<<endl;
// // // //now do the division

// // // if(a==0.0 && b==0.0){//division along y axis

// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // if(pixelTrans.x<xcm){
// // // if(!childCell){
// // // childCell = potts->createCellG(sitr->pixel);
// // // }else{
// // // cellField->set(sitr->pixel, childCell);
// // // }
// // // }
// // // }

// // // }else{//division will be done along axis different than y axis
// // // //cerr<<"before iteration"<<endl;
// // // //cerr<<"cell->volume="<<cell->volume<<endl;
// // // int parentCellVolume=0;
// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // //cerr<<"pt="<<sitr->pixel<<endl;
// // // if(pixelTrans.z <= a*pixelTrans.x+b){

// // // if(!childCell){
// // // //cerr<<" creating cell and adding pt="<<sitr->pixel<<endl;
// // // childCell = potts->createCellG(sitr->pixel);
// // // //cerr<<"childCell="<<childCell<<endl;
// // // }else{
// // // //cerr<<" adding pt="<<sitr->pixel<<endl;
// // // cellField->set(sitr->pixel, childCell);
// // // }
// // // }else{

// // // parentCellVolume++;
// // // }

// // // }
// // // //cerr<<"parentCellVolume="<<parentCellVolume<<endl;
// // // }

// // // //if childCell was created this means mitosis was sucessful. If child cell was not created there was no mitosis
// // // if(childCell)
// // // return true;
// // // else
// // // return false;
// // // }
// // // }

// // // OrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis2D_yz(CellG * cell){
// // // double xcm=cell->xCM/(float)cell->volume;
// // // double ycm=cell->yCM/(float)cell->volume;
// // // double zcm=cell->zCM/(float)cell->volume;

// // // //first calculate and diagonalize inertia tensor
// // // vector<vector<double> > inertiaTensor(2,vector<double>(2,0.0));

// // // set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // //cerr<<"point cell="<<sitr->pixel<<endl;
// // // inertiaTensor[0][0]+=(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
// // // inertiaTensor[0][1]+=-(pixelTrans.y-ycm)*(pixelTrans.z-zcm);
// // // inertiaTensor[1][1]+=(pixelTrans.y-ycm)*(pixelTrans.y-ycm);
// // // }
// // // inertiaTensor[1][0]=inertiaTensor[0][1];

// // // double radical=0.5*sqrt((inertiaTensor[0][0]-inertiaTensor[1][1])*(inertiaTensor[0][0]-inertiaTensor[1][1])+4.0*inertiaTensor[0][1]*inertiaTensor[0][1]);			
// // // double lMin=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])-radical;
// // // double lMax=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])+radical;

// // // //orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
// // // Coordinates3D<double> orientationVec;
// // // if(inertiaTensor[0][1]!=0.0){

// // // orientationVec=Coordinates3D<double>(0.0,inertiaTensor[0][1],lMax-inertiaTensor[0][0]);
// // // double length=sqrt(orientationVec.x*orientationVec.x+orientationVec.y*orientationVec.y+orientationVec.z*orientationVec.z);
// // // orientationVec.x/=length;
// // // orientationVec.y/=length;
// // // orientationVec.z/=length;
// // // }else{
// // // if(inertiaTensor[0][0]>inertiaTensor[1][1])
// // // orientationVec=Coordinates3D<double>(0.0,0.0,1.0);
// // // else
// // // orientationVec=Coordinates3D<double>(0.0,1.0,0.0);
// // // }

// // // OrientationVectorsMitosis orientationVectorsMitosis;
// // // orientationVectorsMitosis.semiminorVec=orientationVec;
// // // orientationVectorsMitosis.semimajorVec.y=-orientationVectorsMitosis.semiminorVec.z;
// // // orientationVectorsMitosis.semimajorVec.z=orientationVectorsMitosis.semiminorVec.y;

// // // return orientationVectorsMitosis;


// // // }

// // // bool MitosisSteppable::doDirectionalMitosis2D_yz(){

// // // //this implementation is valid in 2D only
// // // if (split && on) {
// // // split = false;

// // // WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
// // // //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
// // // childCell=0;
// // // parentCell=0;

// // // CellG *cell = cellField->get(splitPt);//cells that is being divided
// // // parentCell=cell;

// // // double xcm=cell->xCM/(float)cell->volume;
// // // double ycm=cell->yCM/(float)cell->volume;
// // // double zcm=cell->zCM/(float)cell->volume;

// // // set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;

// // // OrientationVectorsMitosis orientationVectorsMitosis=getOrientationVectorsMitosis2D_xz(cell);
// // // Coordinates3D<double> orientationVec=orientationVectorsMitosis.semiminorVec;


// // // //  cerr<<"orientationVec="<<orientationVec<<endl;
// // // //cerr<<"inertiaTensor[0][0]="<<inertiaTensor[0][0]<<endl;
// // // //cerr<<"inertiaTensor[0][1]="<<inertiaTensor[0][1]<<endl;
// // // //cerr<<"inertiaTensor[1][1]="<<inertiaTensor[1][1]<<endl;

// // // //cerr<<"xcm="<<xcm<<" zcm="<<zcm<<endl;

// // // //once we know orientation vector corresponding to bigger eigenvalue (pointing along semiminor axis) we may divide cell 
// // // //along major or minor axis

// // // // assume the following form of the equation of the straight line passing through COM (xcm,ycm,zcm) of cell being divided
// // // //z=a*y+b;
// // // double a;
// // // double b;
// // // //determining coefficients of the straight line passing through
// // // if(divideAlongMinorAxisFlag){
// // // //cerr<<"doing minor axis division"<<endl;


// // // a=orientationVec.z/orientationVec.y;
// // // b=zcm-ycm*a;

// // // if(a!=a){//a is Nan - will happen when orientationVec.x is 0.0 thus minor axis is along y axis
// // // a=0.0;
// // // b=0.0;
// // // }
// // // }

// // // if(divideAlongMajorAxisFlag){
// // // //cerr<<"doing major axis division"<<endl;
// // // if(orientationVec.z==0.0){//then perpendicular vector (major axis) is along z axis meaning:
// // // a=0.0;
// // // b=0.0;

// // // } else{
// // // a=-orientationVec.y/orientationVec.z;
// // // b=zcm-ycm*a;
// // // }
// // // }

// // // //cerr<<"a="<<a<<" b="<<b<<endl;
// // // //now do the division

// // // if(a==0.0 && b==0.0){//division along y axis

// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // if(pixelTrans.y<ycm){
// // // if(!childCell){
// // // childCell = potts->createCellG(sitr->pixel);
// // // }else{
// // // cellField->set(sitr->pixel, childCell);
// // // }
// // // }
// // // }

// // // }else{//division will be done along axis different than y axis
// // // //cerr<<"before iteration"<<endl;
// // // //cerr<<"cell->volume="<<cell->volume<<endl;
// // // int parentCellVolume=0;
// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // //cerr<<"pt="<<sitr->pixel<<endl;
// // // if(pixelTrans.z <= a*pixelTrans.y+b){

// // // if(!childCell){
// // // //cerr<<" creating cell and adding pt="<<sitr->pixel<<endl;
// // // childCell = potts->createCellG(sitr->pixel);
// // // //cerr<<"childCell="<<childCell<<endl;
// // // }else{
// // // //cerr<<" adding pt="<<sitr->pixel<<endl;
// // // cellField->set(sitr->pixel, childCell);
// // // }
// // // }else{

// // // parentCellVolume++;
// // // }

// // // }
// // // //cerr<<"parentCellVolume="<<parentCellVolume<<endl;
// // // }

// // // //if childCell was created this means mitosis was sucessful. If child cell was not created there was no mitosis
// // // if(childCell)
// // // return true;
// // // else
// // // return false;
// // // }
// // // }

// // // OrientationVectorsMitosis MitosisSteppable::getOrientationVectorsMitosis3D(CellG * cell){

// // // double xcm=cell->xCM/(float)cell->volume;
// // // double ycm=cell->yCM/(float)cell->volume;
// // // double zcm=cell->zCM/(float)cell->volume;

// // // //first calculate and diagonalize inertia tensor
// // // vector<vector<double> > inertiaTensor(3,vector<double>(3,0.0));

// // // set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // //cerr<<"point cell="<<sitr->pixel<<endl;
// // // inertiaTensor[0][0]+=(pixelTrans.y-ycm)*(pixelTrans.y-ycm)+(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
// // // inertiaTensor[0][1]+=-(pixelTrans.x-xcm)*(pixelTrans.y-ycm);
// // // inertiaTensor[0][2]+=-(pixelTrans.x-xcm)*(pixelTrans.z-zcm);
// // // inertiaTensor[1][1]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm)+(pixelTrans.z-zcm)*(pixelTrans.z-zcm);
// // // inertiaTensor[1][2]+=-(pixelTrans.y-ycm)*(pixelTrans.z-zcm);
// // // inertiaTensor[2][2]+=(pixelTrans.x-xcm)*(pixelTrans.x-xcm)+(pixelTrans.y-ycm)*(pixelTrans.y-ycm);
// // // }
// // // inertiaTensor[1][0]=inertiaTensor[0][1];
// // // inertiaTensor[2][0]=inertiaTensor[0][2];
// // // inertiaTensor[2][1]=inertiaTensor[1][2];

// // // //Finding eigenvalues
// // // vector<double> aCoeff(4,0.0);
// // // vector<complex<double> > roots;

// // // //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
// // // aCoeff[0]=-1.0;

// // // aCoeff[1]=inertiaTensor[0][0] + inertiaTensor[1][1] + inertiaTensor[2][2];

// // // aCoeff[2]=inertiaTensor[0][1]*inertiaTensor[0][1] + inertiaTensor[0][2]*inertiaTensor[0][2] + inertiaTensor[1][2]*inertiaTensor[1][2]
// // // -inertiaTensor[0][0]*inertiaTensor[1][1] - inertiaTensor[0][0]*inertiaTensor[2][2] - inertiaTensor[1][1]*inertiaTensor[2][2];

// // // aCoeff[3]=inertiaTensor[0][0]*inertiaTensor[1][1]*inertiaTensor[2][2] + 2*inertiaTensor[0][1]*inertiaTensor[0][2]*inertiaTensor[1][2]
// // // -inertiaTensor[0][0]*inertiaTensor[1][2]*inertiaTensor[1][2]
// // // -inertiaTensor[1][1]*inertiaTensor[0][2]*inertiaTensor[0][2]
// // // -inertiaTensor[2][2]*inertiaTensor[0][1]*inertiaTensor[0][1];

// // // roots=solveCubicEquationRealCoeeficients(aCoeff);



// // // vector<Coordinates3D<double> > eigenvectors(3);

// // // for (int i = 0 ; i < 3 ; ++i){
// // // eigenvectors[i].x=(inertiaTensor[0][2]*(inertiaTensor[1][1]-roots[i].real())-inertiaTensor[1][2]*inertiaTensor[0][1])/
// // // (inertiaTensor[1][2]*(inertiaTensor[0][0]-roots[i].real())-inertiaTensor[0][1]*inertiaTensor[0][2]);
// // // eigenvectors[i].y=1.0;
// // // eigenvectors[i].z = (inertiaTensor[0][2]*eigenvectors[i].x+inertiaTensor[1][2]*eigenvectors[i].y)/
// // // (roots[i].real()-inertiaTensor[2][2]) ;
// // // //cerr<<"eigenvectors["<<i<<"]="<<eigenvectors[i]<<endl;
// // // if(eigenvectors[i].x!=eigenvectors[i].x || eigenvectors[i].y!=eigenvectors[i].y || eigenvectors[i].z!=eigenvectors[i].z){
// // // OrientationVectorsMitosis orientationVectorsMitosis;
// // // return orientationVectorsMitosis;//simply dont do mitosis if any of the eigenvector component is NaN
// // // }
// // // }



// // // //finding semiaxes of the ellipsoid
// // // //Ixx=m/5.0*(a_y^2+a_z^2) - andy cyclical permutations for other coordinate combinations
// // // //a_x,a_y,a_z are lengths of semiaxes of the allipsoid
// // // // We can invert above system of equations to get:
// // // vector<double> axes(3,0.0);

// // // axes[0]=sqrt((2.5/cell->volume)*(roots[1].real()+roots[2].real()-roots[0].real()));//corresponds to first eigenvalue
// // // axes[1]=sqrt((2.5/cell->volume)*(roots[0].real()+roots[2].real()-roots[1].real()));//corresponds to second eigenvalue
// // // axes[2]=sqrt((2.5/cell->volume)*(roots[0].real()+roots[1].real()-roots[2].real()));//corresponds to third eigenvalue

// // // vector<pair<double, int> > sortedAxes(3);
// // // sortedAxes[0]=make_pair(axes[0],0);
// // // sortedAxes[1]=make_pair(axes[1],1);
// // // sortedAxes[2]=make_pair(axes[2],2);

// // // //sorting semiaxes according the their lengths (shortest first)
// // // //sort(axes.begin(),axes.end());
// // // sort(sortedAxes.begin(),sortedAxes.end()); //by keeping track of original axes indices we also find which eigenvector corresponds to shortest/longest axis - that's why we use pair where first element is the length of the axis and the second one is index of the eigenvalue. 
// // // //After sorting we can track back which eigenvector belongs to hosrtest/longest eigenvalue

// // // OrientationVectorsMitosis orientationVectorsMitosis;
// // // orientationVectorsMitosis.semiminorVec=eigenvectors[sortedAxes[0].second];
// // // orientationVectorsMitosis.semimajorVec=eigenvectors[sortedAxes[2].second];


// // // return orientationVectorsMitosis;

// // // }


// // // bool MitosisSteppable::doDirectionalMitosis3D(){

// // // //this implementation is valid in 3D only
// // // if (split && on) {
// // // split = false;

// // // WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
// // // //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
// // // childCell=0;
// // // parentCell=0;

// // // CellG *cell = cellField->get(splitPt);//cells that is being divided
// // // parentCell=cell;

// // // double xcm=cell->xCM/(float)cell->volume;
// // // double ycm=cell->yCM/(float)cell->volume;
// // // double zcm=cell->zCM/(float)cell->volume;

// // // set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;

// // // OrientationVectorsMitosis orientationVectorsMitosis=getOrientationVectorsMitosis3D(cell);



// // // //cerr<<"CHECKING eigenvectors"<<endl;
// // // //cerr<<"0*1="<<eigenvectors[0]*eigenvectors[1]<<endl;
// // // //cerr<<"0*2="<<eigenvectors[0]*eigenvectors[2]<<endl;
// // // //  cerr<<"1*2="<<eigenvectors[1]*eigenvectors[2]<<endl;
// // // // to divide cell along semiminor axis we take eigenvector corresponging to semimajor axis and this vector is 
// // // // perpendicular to the division plane
// // // //cerr<<"semimajor axis eigenvector is "<<eigenvectors[sortedAxes[2].second]<<endl;
// // // // to divide cell along semimajor axis we take eigenvector corresponging to semiminor axis and this vector is 
// // // // perpendicular to the division plane

// // // //cerr<<"semiminor axis eigenvector is "<<eigenvectors[sortedAxes[0].second]<<endl;


// // // //plane equation is of the form (r-p)*n=0 where p is vector pointing to point through which the plane will pass (COM)
// // // // n is a normal vector to the plane
// // // // r is (x,y,z) vector
// // // //nx*x+ny*y+nz*z-p*n=0 
// // // //or nx*x+ny*y+nz*z+d=0 where d is a scalar product -p*n
// // // Coordinates3D<double> pVec(xcm,ycm,zcm);
// // // Coordinates3D<double> nVec;
// // // double d;

// // // if(divideAlongMajorAxisFlag){
// // // //cerr<<"doing major axis division"<<endl;

// // // nVec=orientationVectorsMitosis.semiminorVec;;
// // // d=-(pVec*nVec);
// // // //cerr<<"before iteration"<<endl;

// // // } else{
// // // //cerr<<"doing minor axis division"<<endl;
// // // nVec=orientationVectorsMitosis.semimajorVec;;
// // // d=-(pVec*nVec);
// // // }


// // // //cerr<<"cell->volume="<<cell->volume<<endl;
// // // int parentCellVolume=0;
// // // for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
// // // Coordinates3D<double> pixelTrans= boundaryStrategy->calculatePointCoordinates(sitr->pixel);
// // // //cerr<<"pt="<<sitr->pixel<<endl;
// // // if(nVec.x*pixelTrans.x+nVec.y*pixelTrans.y + nVec.z*pixelTrans.z+d<= 0.0){

// // // if(!childCell){
// // // //cerr<<" creating cell and adding pt="<<sitr->pixel<<endl;
// // // childCell = potts->createCellG(sitr->pixel);
// // // //cerr<<"childCell="<<childCell<<endl;
// // // }else{
// // // //cerr<<" adding pt="<<sitr->pixel<<endl;
// // // cellField->set(sitr->pixel, childCell);
// // // }
// // // }else{

// // // parentCellVolume++;
// // // }

// // // }
// // // //cerr<<"parentCellVolume="<<parentCellVolume<<endl;

// // // if(childCell)
// // // return true;
// // // else
// // // return false;

// // // }
// // // }



//bool MitosisSteppable::doDirectionalMitosis3D(){
//
//	//this implementation is valid in 3D only
//	if (split && on) {
//		split = false;
//
//		WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
//		//reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
//		childCell=0;
//		parentCell=0;
//
//		CellG *cell = cellField->get(splitPt);//cells that is being divided
//		parentCell=cell;
//		//cerr<<"dividing parent cell "<<cell->id<<endl;
//		double xcm=cell->xCM/(float)cell->volume;
//		double ycm=cell->yCM/(float)cell->volume;
//		double zcm=cell->zCM/(float)cell->volume;
//
//		//first calculate and diagonalize inertia tensor
//		vector<vector<double> > inertiaTensor(3,vector<double>(3,0.0));
//
//		set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
//		for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//			//cerr<<"point cell="<<sitr->pixel<<endl;
//			inertiaTensor[0][0]+=(sitr->pixel.y-ycm)*(sitr->pixel.y-ycm)+(sitr->pixel.z-zcm)*(sitr->pixel.z-zcm);
//			inertiaTensor[0][1]+=-(sitr->pixel.x-xcm)*(sitr->pixel.y-ycm);
//			inertiaTensor[0][2]+=-(sitr->pixel.x-xcm)*(sitr->pixel.z-zcm);
//			inertiaTensor[1][1]+=(sitr->pixel.x-xcm)*(sitr->pixel.x-xcm)+(sitr->pixel.z-zcm)*(sitr->pixel.z-zcm);
//			inertiaTensor[1][2]+=-(sitr->pixel.y-ycm)*(sitr->pixel.z-zcm);
//			inertiaTensor[2][2]+=(sitr->pixel.x-xcm)*(sitr->pixel.x-xcm)+(sitr->pixel.y-ycm)*(sitr->pixel.y-ycm);
//		}
//		inertiaTensor[1][0]=inertiaTensor[0][1];
//		inertiaTensor[2][0]=inertiaTensor[0][2];
//		inertiaTensor[2][1]=inertiaTensor[1][2];
//		
//	 //Finding eigenvalues
//	 vector<double> aCoeff(4,0.0);
//	 vector<complex<double> > roots;
//	 
//	 //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
//	 aCoeff[0]=-1.0;
//
//	 aCoeff[1]=inertiaTensor[0][0] + inertiaTensor[1][1] + inertiaTensor[2][2];
//
//	 aCoeff[2]=inertiaTensor[0][1]*inertiaTensor[0][1] + inertiaTensor[0][2]*inertiaTensor[0][2] + inertiaTensor[1][2]*inertiaTensor[1][2]
//	 -inertiaTensor[0][0]*inertiaTensor[1][1] - inertiaTensor[0][0]*inertiaTensor[2][2] - inertiaTensor[1][1]*inertiaTensor[2][2];
//
//	 aCoeff[3]=inertiaTensor[0][0]*inertiaTensor[1][1]*inertiaTensor[2][2] + 2*inertiaTensor[0][1]*inertiaTensor[0][2]*inertiaTensor[1][2]
//	 -inertiaTensor[0][0]*inertiaTensor[1][2]*inertiaTensor[1][2]
//	 -inertiaTensor[1][1]*inertiaTensor[0][2]*inertiaTensor[0][2]
//	 -inertiaTensor[2][2]*inertiaTensor[0][1]*inertiaTensor[0][1];
//
//	 roots=solveCubicEquationRealCoeeficients(aCoeff);
//	 
//	 
//
//	 vector<Coordinates3D<double> > eigenvectors(3);
//
//	 for (int i = 0 ; i < 3 ; ++i){
//		 eigenvectors[i].x=(inertiaTensor[0][2]*(inertiaTensor[1][1]-roots[i].real())-inertiaTensor[1][2]*inertiaTensor[0][1])/
//		 (inertiaTensor[1][2]*(inertiaTensor[0][0]-roots[i].real())-inertiaTensor[0][1]*inertiaTensor[0][2]);
//		 eigenvectors[i].y=1.0;
//		 eigenvectors[i].z = (inertiaTensor[0][2]*eigenvectors[i].x+inertiaTensor[1][2]*eigenvectors[i].y)/
//		 (roots[i].real()-inertiaTensor[2][2]) ;
//		 //cerr<<"eigenvectors["<<i<<"]="<<eigenvectors[i]<<endl;
//		 if(eigenvectors[i].x!=eigenvectors[i].x || eigenvectors[i].y!=eigenvectors[i].y || eigenvectors[i].z!=eigenvectors[i].z){
//			return false;//simply dont do mitosis if any of the eigenvector component is NaN
//		 }
//	 }
//
//
//
//	 //finding semiaxes of the ellipsoid
//	 //Ixx=m/5.0*(a_y^2+a_z^2) - andy cyclical permutations for other coordinate combinations
//	 //a_x,a_y,a_z are lengths of semiaxes of the allipsoid
//	 // We can invert above system of equations to get:
//	vector<double> axes(3,0.0);
//
//	axes[0]=sqrt((2.5/cell->volume)*(roots[1].real()+roots[2].real()-roots[0].real()));//corresponds to first eigenvalue
//	axes[1]=sqrt((2.5/cell->volume)*(roots[0].real()+roots[2].real()-roots[1].real()));//corresponds to second eigenvalue
//	axes[2]=sqrt((2.5/cell->volume)*(roots[0].real()+roots[1].real()-roots[2].real()));//corresponds to third eigenvalue
//
//	vector<pair<double, int> > sortedAxes(3);
//	sortedAxes[0]=make_pair(axes[0],0);
//	sortedAxes[1]=make_pair(axes[1],1);
//	sortedAxes[2]=make_pair(axes[2],2);
//
//	//sorting semiaxes according the their lengths (shortest first)
//	//sort(axes.begin(),axes.end());
//	sort(sortedAxes.begin(),sortedAxes.end()); //by keeping track of original axes indices we also find which eigenvector corresponds to shortest/longest axis - that's why we use pair where first element is the length of the axis and the second one is index of the eigenvalue. 
//															 //After sorting we can track back which eigenvector belongs to hosrtest/longest eigenvalue
//
//	//for(int i =0 ; i< 3 ;++i){
//	//	cerr<<"roots["<<i<<"]="<<roots[i]<<endl;
//	//	cerr<<"sortedAxes["<<i<<"].first="<<sortedAxes[i].first<<" original index="<<sortedAxes[i].second<<endl;
//	//}
//
//		
//		//cerr<<"CHECKING eigenvectors"<<endl;
//		//cerr<<"0*1="<<eigenvectors[0]*eigenvectors[1]<<endl;
//		//cerr<<"0*2="<<eigenvectors[0]*eigenvectors[2]<<endl;
//	 //  cerr<<"1*2="<<eigenvectors[1]*eigenvectors[2]<<endl;
//		// to divide cell along semiminor axis we take eigenvector corresponging to semimajor axis and this vector is 
//		// perpendicular to the division plane
//		//cerr<<"semimajor axis eigenvector is "<<eigenvectors[sortedAxes[2].second]<<endl;
//		// to divide cell along semimajor axis we take eigenvector corresponging to semiminor axis and this vector is 
//		// perpendicular to the division plane
//		
//		//cerr<<"semiminor axis eigenvector is "<<eigenvectors[sortedAxes[0].second]<<endl;
//
//
//	//plane equation is of the form (r-p)*n=0 where p is vector pointing to point through which the plane will pass (COM)
//	// n is a normal vector to the plane
//   // r is (x,y,z) vector
//	//nx*x+ny*y+nz*z-p*n=0 
//	//or nx*x+ny*y+nz*z+d=0 where d is a scalar product -p*n
//	Coordinates3D<double> pVec(xcm,ycm,zcm);
//	Coordinates3D<double> nVec;
//	double d;
//
//	if(divideAlongMajorAxisFlag){
//		//cerr<<"doing major axis division"<<endl;
//	
//		nVec=eigenvectors[sortedAxes[0].second];
//		d=-(pVec*nVec);
//			//cerr<<"before iteration"<<endl;
//		
//	} else{
//		//cerr<<"doing minor axis division"<<endl;
//		nVec=eigenvectors[sortedAxes[2].second];
//		d=-(pVec*nVec);
//	}
//	
//
//			//cerr<<"cell->volume="<<cell->volume<<endl;
//			int parentCellVolume=0;
//			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//				//cerr<<"pt="<<sitr->pixel<<endl;
//				if(nVec.x*sitr->pixel.x+nVec.y*sitr->pixel.y + nVec.z*sitr->pixel.z+d<= 0.0){
//					
//					if(!childCell){
//						//cerr<<" creating cell and adding pt="<<sitr->pixel<<endl;
//						childCell = potts->createCellG(sitr->pixel);
//						//cerr<<"childCell="<<childCell<<endl;
//					}else{
//						//cerr<<" adding pt="<<sitr->pixel<<endl;
//						cellField->set(sitr->pixel, childCell);
//					}
//				}else{
//					
//					parentCellVolume++;
//				}
//
//			}
//			//cerr<<"parentCellVolume="<<parentCellVolume<<endl;
//
//		if(childCell)
//			return true;
//		else
//			return false;
//
//	}
//}


//bool MitosisSteppable::doDirectionalMitosis2D_xy(){
//	cerr<<"doDirectionalMitosis2D_xy"<<endl;
//	//this implementation is valid in 2D only
//	if (split && on) {
//		split = false;
//
//		WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
//		//reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitisis is aborted
//		childCell=0;
//		parentCell=0;
//
//		CellG *cell = cellField->get(splitPt);//cells that is being divided
//		parentCell=cell;
//		//cerr<<"dividing parent cell "<<cell->id<<endl;
//		double xcm=cell->xCM/(float)cell->volume;
//		double ycm=cell->yCM/(float)cell->volume;
//		double zcm=cell->zCM/(float)cell->volume;
//
//		//first calculate and diagonalize inertia tensor
//		vector<vector<double> > inertiaTensor(2,vector<double>(2,0.0));
//
//		set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
//		for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//			//cerr<<"point cell="<<sitr->pixel<<endl;
//			inertiaTensor[0][0]+=(sitr->pixel.y-ycm)*(sitr->pixel.y-ycm);
//			inertiaTensor[0][1]+=-(sitr->pixel.x-xcm)*(sitr->pixel.y-ycm);
//			inertiaTensor[1][1]+=(sitr->pixel.x-xcm)*(sitr->pixel.x-xcm);
//		}
//		inertiaTensor[1][0]=inertiaTensor[0][1];
//
//		double radical=0.5*sqrt((inertiaTensor[0][0]-inertiaTensor[1][1])*(inertiaTensor[0][0]-inertiaTensor[1][1])+4.0*inertiaTensor[0][1]*inertiaTensor[0][1]);			
//		double lMin=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])-radical;
//		double lMax=0.5*(inertiaTensor[0][0]+inertiaTensor[1][1])+radical;
//
//		//orientationVec points along semiminor axis (it corresponds to larger eigenvalue)
//		Coordinates3D<double> orientationVec;
//		if(inertiaTensor[0][1]!=0.0){
//
//			orientationVec=Coordinates3D<double>(inertiaTensor[0][1],lMax-inertiaTensor[0][0],0.0);
//			double length=sqrt(orientationVec.x*orientationVec.x+orientationVec.y*orientationVec.y+orientationVec.z*orientationVec.z);
//			orientationVec.x/=length;
//			orientationVec.y/=length;
//			orientationVec.z/=length;
//		}else{
//			if(inertiaTensor[0][0]>inertiaTensor[1][1])
//				orientationVec=Coordinates3D<double>(0.0,1.0,0.0);
//			else
//				orientationVec=Coordinates3D<double>(1.0,0.0,0.0);
//		}
//
//	 //  cerr<<"orientationVec="<<orientationVec<<endl;
//		//cerr<<"inertiaTensor[0][0]="<<inertiaTensor[0][0]<<endl;
//		//cerr<<"inertiaTensor[0][1]="<<inertiaTensor[0][1]<<endl;
//		//cerr<<"inertiaTensor[1][1]="<<inertiaTensor[1][1]<<endl;
//
//		//cerr<<"xcm="<<xcm<<" ycm="<<ycm<<endl;
//
//		//once we know orientation vector corresponding to bigger eigenvalue (pointing along semiminor axis) we may divide cell 
//		//along major or minor axis
//
//		// assume the following form of the equation of the straight line passing through COM (xcm,ycm,zcm) of cell being divided
//		//y=a*x+b;
//		double a;
//		double b;
//		//determining coefficients of the straight line passing through
//		if(divideAlongMinorAxisFlag){
//			//cerr<<"doing minor axis division"<<endl;
//
//
//			a=orientationVec.y/orientationVec.x;
//			b=ycm-xcm*a;
//
//			if(a!=a){//a is Nan - will happen when orientationVec.x is 0.0 thus minor axis is along y axis
//				a=0.0;
//				b=0.0;
//			}
//		}
//
//		if(divideAlongMajorAxisFlag){
//			//cerr<<"doing major axis division"<<endl;
//			if(orientationVec.y==0.0){//then perpendicular vector (major axis) is along y axis meaning:
//				a=0.0;
//				b=0.0;
//
//			} else{
//				a=-orientationVec.x/orientationVec.y;
//				b=ycm-xcm*a;
//			}
//		}
//
//		//cerr<<"a="<<a<<" b="<<b<<endl;
//		//now do the division
//
//		if(a==0.0 && b==0.0){//division along y axis
//
//			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//				if(sitr->pixel.x<xcm){
//					if(!childCell){
//						childCell = potts->createCellG(sitr->pixel);
//					}else{
//						cellField->set(sitr->pixel, childCell);
//					}
//				}
//			}
//
//		}else{//division will be done along axis different than y axis
//			//cerr<<"before iteration"<<endl;
//			//cerr<<"cell->volume="<<cell->volume<<endl;
//			int parentCellVolume=0;
//			for(set<PixelTrackerData>::iterator sitr=cellPixels.begin() ; sitr != cellPixels.end() ;++sitr){
//				//cerr<<"pt="<<sitr->pixel<<endl;
//				if(sitr->pixel.y <= a*sitr->pixel.x+b){
//					
//					if(!childCell){
//						//cerr<<" creating cell and adding pt="<<sitr->pixel<<endl;
//						childCell = potts->createCellG(sitr->pixel);
//						//cerr<<"childCell="<<childCell<<endl;
//					}else{
//						//cerr<<" adding pt="<<sitr->pixel<<endl;
//						cellField->set(sitr->pixel, childCell);
//					}
//				}else{
//					
//					parentCellVolume++;
//				}
//
//			}
//			//cerr<<"parentCellVolume="<<parentCellVolume<<endl;
//		}
//
//		//if childCell was created this means mitosis was sucessful. If child cell was not created there was no mitosis
//		if(childCell)
//			return true;
//		else
//			return false;
//	}
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//bool MitosisSteppable::doDirectionalMitosisOrientationVectorBasedCompartments(long _clusterId,double _nx, double _ny, double _nz){
//	//CHECK IF IT WILL WORK WITH PERIODIC BOUNDARY CONDITIONS ALL DISTANCES AND DISPLACEMENT VECTORS MAY NEED TO BE RECALCULATED THEN
//	CellInventory & inventory=potts->getCellInventory();
//	CC3DCellList compartmentsVec=inventory.getClusterCells(_clusterId);
//	//cerr<<"Got the following compartments for cluster id="<<_clusterId<<endl;
//	int numberOfClusters=compartmentsVec.size();
//
//	comOffsetsMitosis.assign(numberOfClusters,CompartmentMitosisData());
//	parentBeforeMitosis.assign(numberOfClusters,CompartmentMitosisData());	
//	parentAfterMitosis.assign(numberOfClusters,CompartmentMitosisData());
//	childAfterMitosis.assign(numberOfClusters,CompartmentMitosisData());
//	vector<int> originalCompartmentVolumeVec(numberOfClusters,0);
//
//	for(int i = 0 ; i < numberOfClusters; ++i){
//		CellG *cell=compartmentsVec[i];
//		parentBeforeMitosis[i].com=Vector3(cell->xCM/(double)cell->volume,cell->yCM/(double)cell->volume,cell->zCM/(double)cell->volume);
//		parentBeforeMitosis[i].cell=cell;
//		originalCompartmentVolumeVec[i]=cell->volume;
//	}
//
//
//
//	//construct a set containing all pixels of the cluster
//	set<PixelTrackerData> clusterPixels;		
//	for(int i = 0 ; i < numberOfClusters; ++i){
//		CellG *cell=compartmentsVec[i];
//		set<PixelTrackerData> cellPixels=pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
//		clusterPixels.insert(cellPixels.begin(),cellPixels.end());
//	}
//	//divide cluster pixels using mitosis orientation based algorithm
//	set<PixelTrackerData> clusterParent;
//	set<PixelTrackerData> clusterChild;
//	bool clusterDivideFlag=divideClusterPixelsOrientationVectorBased(clusterPixels,clusterParent,clusterChild,_nx,_ny,_nz);
//
//	Vector3 clusterCOMBeforeMitosis=calculateClusterPixelsCOM(clusterPixels);
//
//	Vector3 clusterParentCOM=calculateClusterPixelsCOM(clusterParent);
//	Vector3 clusterChildCOM=calculateClusterPixelsCOM(clusterChild);
//
//	//cerr<<"clusterCOMBeforeMitosis="<<clusterCOMBeforeMitosis<<endl;
//	//cerr<<"clusterParentCOM="<<clusterParentCOM<<endl;
//	//cerr<<"clusterChildCOM="<<clusterChildCOM<<endl;
//
//	//now will calculate offsets of displacement vectors after mitosis w.r.t COM of the cluster. We shrink original displacements by appropriate factor (2 should be sufficient) 
//	//we will use parallel/perpendicular vector decomposition w.r.t nx, ny, nz and scale only parallel component of the offset vector
//	//offset_original=parentBeforeMitosis[i].com-referenceVector
//	//offset_scaled=offset_original+(offset_original*n)n[scalingFactor-1]
//	if(!_nx&&!_ny&&!_nz){
//		return false; //orientation vector is 0
//	}
//	Vector3 nVec(_nx,_ny,_nz);
//	double norm=nVec.Mag();
//	nVec*=1.0/norm;
//
//	double scalingFactor=0.5;
//	Vector3 referenceVector=clusterCOMBeforeMitosis;
//	for(int i = 0 ; i < numberOfClusters; ++i){
//
//		//comOffsetsMitosis[i].com=(parentBeforeMitosis[i].com-referenceVector)*scalingFactor;			
//		Vector3 offsetOriginal=parentBeforeMitosis[i].com-referenceVector;
//		comOffsetsMitosis[i].com=offsetOriginal+(offsetOriginal*nVec)*nVec*(scalingFactor-1);
//	}
//
//
//	//now will calculate radii of attractions for compartments of child clusters
//	//have to do it before modifying parent cluster
//	vector<double> attractionRadiusParent(numberOfClusters,0.0);
//	vector<double> attractionRadiusChild(numberOfClusters,0.0);
//	double pi=acos(-1.0);
//	if(flag3D){
//		for(int i = 0 ; i < numberOfClusters; ++i){
//			attractionRadiusParent[i]=0.5*pow(3*parentBeforeMitosis[i].cell->volume/pi,1/3.0); //note that we are calculating the radius assuming half volume		
//		}
//	}else{
//		for(int i = 0 ; i < numberOfClusters; ++i){
//			attractionRadiusParent[i]=sqrt(parentBeforeMitosis[i].cell->volume/(2*pi)); //note that we are calculating the radius assuming half volume		
//		}
//	}
//
//
//	//calculate positions of the com's of cluster compartments after mitosis for parent and child clusters 
//	for(int i = 0 ; i < numberOfClusters; ++i){
//		parentAfterMitosis[i].com=clusterParentCOM+comOffsetsMitosis[i].com;
//		childAfterMitosis[i].com=clusterChildCOM+comOffsetsMitosis[i].com;
//	}
//	//initialize coms for parent and child cells
//	set<PixelTrackerData> parentCellKernels;
//	set<PixelTrackerData> childCellKernels;
//
//	WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
//
//	//first check if coms belong to set of cluster pixels
//	for(int i = 0 ; i < numberOfClusters; ++i){
//		Point3D pt=Point3D(parentAfterMitosis[i].com.fX,parentAfterMitosis[i].com.fY,parentAfterMitosis[i].com.fZ);
//		if(clusterParent.find(PixelTrackerData(pt))==clusterParent.end()){
//			//cerr<<"com for compartment "<<i<<" of parent cell does not belong to pixels of the parent cell after division"<<endl;
//			return false;
//			ASSERT_OR_THROW("Mitosis cannot be done for this parent cell",false);
//		}
//
//		pt=Point3D(childAfterMitosis[i].com.fX,childAfterMitosis[i].com.fY,childAfterMitosis[i].com.fZ);
//		if(clusterChild.find(PixelTrackerData(pt))==clusterChild.end()){
//			//cerr<<"com for compartment "<<i<<" of child cell does not belong to pixels of the child cell after division"<<endl;
//			return false;
//			ASSERT_OR_THROW("Mitosis cannot be done for this child cell",false);
//		}
//
//	}
//
//	long childClusterId=0;
//	for(int i = 0 ; i < numberOfClusters; ++i){
//		Point3D pt;	
//
//
//		pt=Point3D(parentAfterMitosis[i].com.fX,parentAfterMitosis[i].com.fY,parentAfterMitosis[i].com.fZ);
//		//CellG * newCell1 = potts->createCellG(pt);
//		//newCell1->type=6; 
//
//		cellField->set(pt,parentBeforeMitosis[i].cell);
//		parentCellKernels.insert(PixelTrackerData(pt));
//		//parentBeforeMitosis[i].cell->type=6; 
//
//		pt=Point3D(childAfterMitosis[i].com.fX,childAfterMitosis[i].com.fY,childAfterMitosis[i].com.fZ);
//
//		CellG * childCompartmentCell = potts->createCellG(pt,potts->getRecentlyCreatedClusterId()+1);//have to initialize cluster id to be greater than any pther cluster id in the inventory
//		childCompartmentCell->type=parentBeforeMitosis[i].cell->type;
//		//childCompartmentCell->type=6; 
//		if (i==0){
//			childClusterId=childCompartmentCell->clusterId;
//		}else{//have to reassign cluster id for compartments of child cluster
//			potts->getCellInventory().reassignClusterId(childCompartmentCell,childClusterId);			
//		}
//
//		//cellField->set(pt,parentBeforeMitosis[i].cell);
//		childCellKernels.insert(PixelTrackerData(pt));
//
//		//parentBeforeMitosis[i].cell->type=6; 
//
//	}
//
//
//	//now will reset all the pixels except of kernel pixels to be medium
//	for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr!=clusterPixels.end() ; ++sitr){
//		if(parentCellKernels.find(*sitr)==parentCellKernels.end() && childCellKernels.find(*sitr)==childCellKernels.end()){
//			cellField->set(sitr->pixel,0);
//		}
//	}
//	//now we fill child and parent cells with pixels following the algorothm based on pixel distances and attraction radii
//	//child
//	childCell=cellField->get(childCellKernels.begin()->pixel); //we assign child cell to be first cell of the cluster 
//	initializeClusters(originalCompartmentVolumeVec, clusterChild ,childCellKernels,attractionRadiusChild);  
//	//parent
//	parentCell=cellField->get(parentCellKernels.begin()->pixel); //we assign parent cell to be first cell of the cluster 
//	initializeClusters(originalCompartmentVolumeVec , clusterParent ,parentCellKernels,attractionRadiusParent);
//	//cerr<<"parentClusterId="<<parentCell->clusterId<<endl;
//	
//	cerr<<"childClusterId="<<childCell->clusterId<<endl;
//
//	//for (int i = 0 ; i < compartmentsVec.size() ; ++i){
//	//	cerr<<"Compartment is="<<compartmentsVec[i]->id<<" type="<<(int)compartmentsVec[i]->type<<" volume="<<compartmentsVec[i]->volume<<endl;
//	//}
//
//	return true;
//
//}
//void MitosisSteppable::initializeClusters(std::vector<int> & originalCompartmentVolumeVec , std::set<PixelTrackerData> & clusterPixels,std::set<PixelTrackerData> &clusterKernels,std::vector<double> & attractionRadiusVec){
//
//	vector<CellG*> kernelCellVec(clusterKernels.size(),0);
//	WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
//	//getting pointers to cells at kernel points
//	int counter=0;
//	for(set<PixelTrackerData>::iterator sitr=clusterKernels.begin() ; sitr!=clusterKernels.end() ; ++sitr){
//		kernelCellVec[counter]=cellField->get(sitr->pixel);
//		cerr<<"kernelCellVec["<<counter<<"]="<<kernelCellVec[counter]<<endl;
//		if(kernelCellVec[counter]){
//			cerr<<"cell type="<<(int)kernelCellVec[counter]->type<<endl;
//		}
//		++counter;
//	}
//
//	set<CellG*> kernelCellSet(kernelCellVec.begin(),kernelCellVec.end()); 
//
//	vector<PixelTrackerData> clusterKernelsVec(clusterKernels.begin(),clusterKernels.end());
//
//	map<double,int> pixel2KernelDistMap;	
//	map<double,int> pixelWithinAttractionRadiusMap;	
//	double dist;
//	Point3D pixel;
//	bool pixelAssignedFlag;
//	for(set<PixelTrackerData>::iterator sitr=clusterPixels.begin() ; sitr!=clusterPixels.end() ; ++sitr){
//		pixel2KernelDistMap.clear();
//		pixelWithinAttractionRadiusMap.clear();
//		pixelAssignedFlag=false;
//		pixel=sitr->pixel;
//
//		for (int i = 0 ; i < clusterKernelsVec.size() ; ++i){			
//			Point3D kernel=clusterKernelsVec[i].pixel;
//			dist=distInvariantCM(pixel.x,pixel.y,pixel.z,kernel.x,kernel.y,kernel.z,fieldDim,boundaryStrategy);
//			pixel2KernelDistMap.insert(make_pair(dist,i));
//			if(dist<attractionRadiusVec[i]){
//				pixelWithinAttractionRadiusMap.insert(make_pair(dist,i));
//			}
//		}
//		//we know the kernels for which pixel is within the attraction radius and we know which kernel is closest to the pixel. 
//		//We first try to assign pixel to the cell whose kernel is within attraction radius
//		for(map<double,int>::iterator mitr=pixelWithinAttractionRadiusMap.begin() ; mitr!=pixelWithinAttractionRadiusMap.end() ; ++mitr){
//			if (kernelCellVec[mitr->second]->volume < originalCompartmentVolumeVec[mitr->second]/2){
//				cellField->set(pixel,kernelCellVec[mitr->second]);				
//				pixelAssignedFlag=true;
//				break;
//			}
//		}
//
//		if(pixelAssignedFlag)
//			continue;
//
//		//next if pixel is not within attraction radius of any kernel or if it is but such cells cannot accept more pixels we assign the pixel to the cell 
//		//whose kernel is the closest 		
//		for(map<double,int>::iterator mitr=pixel2KernelDistMap.begin() ; mitr!=pixel2KernelDistMap.end() ; ++mitr){
//			//if (kernelCellVec[mitr->second]->volume < originalCompartmentVolumeVec[mitr->second]/2){
//				cellField->set(pixel,kernelCellVec[mitr->second]);
//				pixelAssignedFlag=true;
//				break;
//			//}
//
//		}
//
//		//if(pixelAssignedFlag)
//		//	continue;
//
//		////finally we assign reminder pixels (if any) the the closest kernel
//		//cerr<<"assign pixel="<<pixel<<" to cluster="<<pixel2KernelDistMap.begin()->second<<endl;
//		//cellField->set(pixel,kernelCellVec[pixel2KernelDistMap.begin()->second]);
//	}
//}

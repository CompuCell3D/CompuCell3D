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



#include "CellMomentOfInertia.h"

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/ClassRegistry.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>
// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// // // #include <PublicUtilities/NumericalUtils.h>
// // // #include <complex>
// // // #include <algorithm>

// // // #include <CompuCell3D/Potts3D/CellInventory.h>


using namespace CompuCell3D;



// // // #include <cmath>

// // // #include <iostream>


#include "MomentOfInertiaPlugin.h"

using namespace std;




MomentOfInertiaPlugin::MomentOfInertiaPlugin():potts(0),simulator(0),boundaryStrategy(0),lastMCSPrintedWarning(-1),cellOrientationFcnPtr(0) {}

MomentOfInertiaPlugin::~MomentOfInertiaPlugin() {}

void MomentOfInertiaPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	cerr<<"\n\n\n  \t\t\t CALLING INIT OF MOMENT OF INERTIA PLUGIN\n\n\n"<<endl;
	this->simulator=simulator;
	potts = simulator->getPotts();
	bool pluginAlreadyRegisteredFlag;
	Plugin *plugin=Simulator::pluginManager.get("CenterOfMass",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	//   Simulator::pluginManager.get("CenterOfMass"); //this will load CenterOfMass plugin if it is not already loaded

	potts->registerCellGChangeWatcher(this);

	//   simulator->getClassRegistry()
	//     ->registerRenderer("MomentOfInertia", new BasicClassFactory<FieldRenderer,
	// 		       MomentOfInertiaRenderer>);
	// 
	//   simulator->getClassRegistry()
	//     ->registerRenderer("MomentOfInertiaType", new BasicClassFactory<FieldRenderer,
	//                        MomentOfInertiaTypeRenderer>);

	potts->getBoundaryXName()=="Periodic" ? boundaryConditionIndicator.x=1 : boundaryConditionIndicator.x=0 ;
	potts->getBoundaryYName()=="Periodic" ? boundaryConditionIndicator.y=1 : boundaryConditionIndicator.y=0;
	potts->getBoundaryZName()=="Periodic" ? boundaryConditionIndicator.z=1 : boundaryConditionIndicator.z=0;





	fieldDim=potts->getCellFieldG()->getDim();

	if(fieldDim.x==1){
		cellOrientationFcnPtr=&MomentOfInertiaPlugin::cellOrientation_yz;
		getSemiaxesFctPtr=&MomentOfInertiaPlugin::getSemiaxesYZ;

	}else if(fieldDim.y==1){
		cellOrientationFcnPtr=&MomentOfInertiaPlugin::cellOrientation_xz;
		getSemiaxesFctPtr=&MomentOfInertiaPlugin::getSemiaxesXZ;

	}else if (fieldDim.z==1){
		cellOrientationFcnPtr=&MomentOfInertiaPlugin::cellOrientation_xy;
		getSemiaxesFctPtr=&MomentOfInertiaPlugin::getSemiaxesXY;

	}else{
		getSemiaxesFctPtr=&MomentOfInertiaPlugin::getSemiaxes3D;
	}


	boundaryStrategy=BoundaryStrategy::getInstance();
}




void CompuCell3D::MomentOfInertiaPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG * oldCell) {

	//to calculate CM for the case with periodic boundary conditions we need to do some translations rather than naively calculate centroids
	//naive calculations work in the case of no flux boundary conditions but periodic b.c. may cause troubles

	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;

	Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);

	//if no boundary conditions are present
	int currentStep=simulator->getStep();
	if( !(currentStep %100)	&& lastMCSPrintedWarning < currentStep){
		lastMCSPrintedWarning = currentStep;
		if ( boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z ){
			cerr<<"MomentOfInertia plugin may not work properly with periodic boundary conditions.Pleas see manual when it is OK to use this plugin with periodic boundary conditions"<<endl;
		}
	}
	double xcm, ycm, zcm;
	if (newCell != 0)
	{
		// Assumption: COM and Volume has been updated.
		double xcmOld, ycmOld, zcmOld, xcm, ycm, zcm;
		if (newCell->volume > 1) {
			xcmOld = (newCell->xCM - ptTrans.x)/ ((double)newCell->volume - 1);
			ycmOld = (newCell->yCM - ptTrans.y)/ ((double)newCell->volume - 1);
			zcmOld = (newCell->zCM - ptTrans.z)/ ((double)newCell->volume - 1);
		}
		else
		{
			xcmOld = 0.0;
			ycmOld = 0.0;
			zcmOld = 0.0;
		}
		xcm = (double) newCell->xCM / (double) newCell->volume;
		ycm = (double) newCell->yCM / (double) newCell->volume;
		zcm = (double) newCell->zCM / (double) newCell->volume;

		newCell->iXX=newCell->iXX+(newCell->volume - 1)*(ycmOld*ycmOld+zcmOld*zcmOld)-(newCell->volume)*(ycm*ycm+zcm*zcm)+ptTrans.y*ptTrans.y+ptTrans.z*ptTrans.z;
		newCell->iYY=newCell->iYY+(newCell->volume - 1)*(xcmOld*xcmOld+zcmOld*zcmOld)-(newCell->volume)*(xcm*xcm+zcm*zcm)+ptTrans.x*ptTrans.x+ptTrans.z*ptTrans.z;
		newCell->iZZ=newCell->iZZ+(newCell->volume - 1)*(xcmOld*xcmOld+ycmOld*ycmOld)-(newCell->volume)*(xcm*xcm+ycm*ycm)+ptTrans.x*ptTrans.x+ptTrans.y*ptTrans.y;

		newCell->iXY=newCell->iXY-(newCell->volume - 1)*xcmOld*ycmOld+(newCell->volume)*xcm*ycm-ptTrans.x*ptTrans.y;
		newCell->iXZ=newCell->iXZ-(newCell->volume - 1)*xcmOld*zcmOld+(newCell->volume)*xcm*zcm-ptTrans.x*ptTrans.z;	
		newCell->iYZ=newCell->iYZ-(newCell->volume - 1)*ycmOld*zcmOld+(newCell->volume)*ycm*zcm-ptTrans.y*ptTrans.z;	

	}  
	if (oldCell != 0)
	{
		// Assumption: COM and Volume has been updated.
		double xcmOld = (oldCell->xCM + ptTrans.x) / ((double)oldCell->volume + 1);
		double ycmOld = (oldCell->yCM + ptTrans.y) / ((double)oldCell->volume + 1);
		double zcmOld = (oldCell->zCM + ptTrans.z) / ((double)oldCell->volume + 1);

		if (oldCell->volume >= 1){
			xcm = (double) oldCell->xCM / (double) oldCell->volume;
			ycm = (double) oldCell->yCM / (double) oldCell->volume;
			zcm = (double) oldCell->zCM / (double) oldCell->volume;
		}else{
			xcm = 0.0;
			ycm = 0.0;
			zcm = 0.0;
		}

		oldCell->iXX =oldCell->iXX+(oldCell->volume + 1)*(ycmOld*ycmOld+zcmOld*zcmOld)-(oldCell->volume)*(ycm*ycm+zcm*zcm)-ptTrans.y*ptTrans.y-ptTrans.z*ptTrans.z;
		oldCell->iYY =oldCell->iYY+(oldCell->volume + 1)*(xcmOld*xcmOld+zcmOld*zcmOld)-(oldCell->volume)*(xcm*xcm+zcm*zcm)-ptTrans.x*ptTrans.x-ptTrans.z*ptTrans.z;
		oldCell->iZZ =oldCell->iZZ+(oldCell->volume + 1)*(xcmOld*xcmOld+ycmOld*ycmOld)-(oldCell->volume)*(xcm*xcm+ycm*ycm)-ptTrans.x*ptTrans.x-ptTrans.y*ptTrans.y;

		oldCell->iXY =oldCell->iXY-(oldCell->volume + 1)*xcmOld*ycmOld+(oldCell->volume)*xcm*ycm+ptTrans.x*ptTrans.y;
		oldCell->iXZ =oldCell->iXZ-(oldCell->volume + 1)*xcmOld*zcmOld+(oldCell->volume)*xcm*zcm+ptTrans.x*ptTrans.z;	
		oldCell->iYZ =oldCell->iYZ-(oldCell->volume + 1)*ycmOld*zcmOld+(oldCell->volume)*ycm*zcm+ptTrans.y*ptTrans.z;	

	}
	//calculating cell orientation parameters eccentricity,orienation vector
	if(cellOrientationFcnPtr){ //this will call cell orientation calculations only when simulation is 2D
		(this->*cellOrientationFcnPtr)(pt,newCell,oldCell);
	}

	return;

	//if there are boundary conditions defined that we have to do some shifts to correctly calculate center of mass
	//This approach will work only for cells whose span is much smaller that lattice dimension in the "periodic "direction
	//e.g. cell that is very long and "wraps lattice" will have miscalculated CM using this algorithm. On the other hand, you do not real expect
	//cells to have dimensions comparable to lattice...

	//THIS IS NOT IMPLEMENTED YET. WE WILL DO IT IN THE ONE OF THE COMING RELEASES
}

string MomentOfInertiaPlugin::toString(){return "MomentOfInertia";}

void MomentOfInertiaPlugin::cellOrientation_xy(const Point3D &pt, CellG *newCell,CellG *oldCell){

	double lMinNew;
	double lMaxNew;
	double lMinOld;
	double lMaxOld;

	if(newCell){
		double radicalNew=0.5*sqrt((newCell->iXX-newCell->iYY)*(newCell->iXX-newCell->iYY)+4.0*newCell->iXY*newCell->iXY);	
		lMinNew=0.5*(newCell->iXX+newCell->iYY)-radicalNew;
		lMaxNew=0.5*(newCell->iXX+newCell->iYY)+radicalNew;
		//cerr<<"MI lMinOld="<<lMinOld<<" lMaxOld="<<lMaxOld<<endl;
		double ratio=lMinNew/lMaxNew;
		if(ratio!=ratio || fabs(ratio)>1.0){
			newCell->ecc=sqrt(1.0);
		}
		else{
			newCell->ecc=sqrt(1.0-ratio);
		}


		//cerr<<"MI newCell->ecc="<<newCell->ecc<<endl;

		//components of eigenvector corersponding to larger eigenvalue - it points along semiminor axis
		//in the case of an ellipse larger moment of inertia is w.r.t. semiminor axis and is equal 1/4*a**2*M where a is semimajor axis
		//double xHat=newCell->iXY; 
		//double yHat=lMaxNew-newCell->iXX;
		//double cosAlpha=xHat*xHat/sqrt(xHat*xHat+yHat*yHat);
		//double sinAlpha=sqrt(1-cosAlpha);

		newCell->lX=newCell->iXY;
		newCell->lY=lMaxNew-newCell->iXX;

		//if(newCell->iXY!=0.0){
		//	newCell->lX=newCell->iXY;
		//	newCell->lY=lMaxOld-newCell->iXX;
		//}else{
		//	if(newCell->iXX>newCell->iYY){
		//		newCell->lX=newCell->iXX;
		//		newCell->lY=0.0;
		//	}else{

		//		newCell->lX=0.0;
		//		newCell->lY=newCell->iYY;
		//	}

		//}

	}
	if(oldCell){	
		double radicalOld=0.5*sqrt((oldCell->iXX-oldCell->iYY)*(oldCell->iXX-oldCell->iYY)+4.0*oldCell->iXY*oldCell->iXY);
		//cerr<<"MI oldCell->volume="<<oldCell->volume<<endl;
		//cerr<<" MI radicalOld="<<radicalOld<<endl;	
		lMinOld=0.5*(oldCell->iXX+oldCell->iYY)-radicalOld;
		lMaxOld=0.5*(oldCell->iXX+oldCell->iYY)+radicalOld;
		//cerr<<"MI lMinOld="<<lMinOld<<" lMaxOld="<<lMaxOld<<endl;
		double ratio=lMinNew/lMaxNew;
		if(ratio!=ratio || fabs(ratio)>1.0){
			oldCell->ecc=1.0;
		}
		else{
			oldCell->ecc=sqrt(1.0-ratio);
		}


		oldCell->lX=oldCell->iXY;
		oldCell->lY=lMaxOld-oldCell->iXX;

		
		//cerr<<"MI oldCell->ecc="<<oldCell->ecc<<endl;
		//if(oldCell->iXY!=0.0){
		//	oldCell->lX=oldCell->iXY;
		//	oldCell->lY=lMaxOld-oldCell->iXX;
		//}else{
		//	if(oldCell->iXX>oldCell->iYY){
		//		oldCell->lX=oldCell->iXX;
		//		oldCell->lY=0.0;
		//	}else{

		//		oldCell->lX=0.0;
		//		oldCell->lY=oldCell->iYY;
		//	}

		//}
	}

}


void MomentOfInertiaPlugin::cellOrientation_xz(const Point3D &pt, CellG *newCell,CellG *oldCell){

	double lMinNew;
	double lMaxNew;
	double lMinOld;
	double lMaxOld;
	if(newCell){
		//newCell
		double radicalNew=0.5*sqrt((newCell->iXX-newCell->iZZ)*(newCell->iXX-newCell->iZZ)+4.0*newCell->iXZ*newCell->iXZ);
		

		lMinNew=0.5*(newCell->iXX+newCell->iZZ)-radicalNew;
		lMaxNew=0.5*(newCell->iXX+newCell->iZZ)+radicalNew;


		
		double ratio=lMinNew/lMaxNew;
		if(ratio!=ratio || fabs(ratio)>1.0){
			newCell->ecc=sqrt(1.0);
		}
		else{
			newCell->ecc=sqrt(1.0-ratio);
		}


		newCell->lX=newCell->iXZ;
		newCell->lZ=lMaxNew-newCell->iXX;
	}
	if(oldCell){
		//oldCell
		double radicalOld=0.5*sqrt((oldCell->iXX-oldCell->iZZ)*(oldCell->iXX-oldCell->iZZ)+4.0*oldCell->iXZ*oldCell->iXZ);
		lMinOld=0.5*(oldCell->iXX+oldCell->iZZ)-radicalOld;
		lMaxOld=0.5*(oldCell->iXX+oldCell->iZZ)+radicalOld;

		double ratio=lMinNew/lMaxNew;
		if(ratio!=ratio || fabs(ratio)>1.0){
			oldCell->ecc=1.0;
		}
		else{
			oldCell->ecc=sqrt(1.0-ratio);
		}
		oldCell->lX=oldCell->iXZ;
		oldCell->lZ=lMaxOld-oldCell->iXX;
	}
}

void MomentOfInertiaPlugin::cellOrientation_yz(const Point3D &pt, CellG *newCell,CellG *oldCell){

	double lMinNew;
	double lMaxNew;
	double lMinOld;
	double lMaxOld;
	if(newCell){
		//newCell
		double radicalNew=0.5*sqrt((newCell->iYY-newCell->iZZ)*(newCell->iYY-newCell->iZZ)+4.0*newCell->iYZ*newCell->iYZ);
		lMinNew=0.5*(newCell->iYY+newCell->iZZ)-radicalNew;
		lMaxNew=0.5*(newCell->iYY+newCell->iZZ)+radicalNew;
		double ratio=lMinNew/lMaxNew;
		if(ratio!=ratio || fabs(ratio)>1.0){
			newCell->ecc=sqrt(1.0);
		}
		else{
			newCell->ecc=sqrt(1.0-ratio);
		}
		
		newCell->lY=newCell->iYZ;
		newCell->lZ=lMaxNew-newCell->iYY;
	}
	if(oldCell){
		//oldCell
		double radicalOld=0.5*sqrt((oldCell->iYY-oldCell->iZZ)*(oldCell->iYY-oldCell->iZZ)+4.0*oldCell->iYZ*oldCell->iYZ);
		lMinOld=0.5*(oldCell->iYY+oldCell->iZZ)-radicalOld;
		lMaxOld=0.5*(oldCell->iYY+oldCell->iZZ)+radicalOld;

		double ratio=lMinNew/lMaxNew;
		if(ratio!=ratio || fabs(ratio)>1.0){
			oldCell->ecc=1.0;
		}
		else{
			oldCell->ecc=sqrt(1.0-ratio);
		}
		oldCell->lY=oldCell->iYZ;
		oldCell->lZ=lMaxOld-oldCell->iYY;
	}
}


//void MomentOfInertiaPlugin::getSemiaxes(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis)
//{
//
//	(this->*getSemiaxesFctPtr)(_cell,_majorAxis,_medianAxis,_minorAxis);
//}
//
//void MomentOfInertiaPlugin::getSemiaxesXY(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis){
//	    //in the case of an ellipse larger moment of inertia is w.r.t. semiminor axis and is equal 1/4*a**2*M where a is semimajor axis
//		double radical=0.5*sqrt((_cell->iXX-_cell->iYY)*(_cell->iXX-_cell->iYY)+4.0*_cell->iXY*_cell->iXY);	
//		double lMin=0.5*(_cell->iXX+_cell->iYY)-radical;
//		double lMax=0.5*(_cell->iXX+_cell->iYY)+radical;
//		_majorAxis=2*sqrt(lMax/_cell->volume);
//		_medianAxis=0.0;
//	    _minorAxis=2*sqrt(lMin/_cell->volume);
//}
//
//void MomentOfInertiaPlugin::getSemiaxesXZ(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis){
//	    //in the case of an ellipse larger moment of inertia is w.r.t. semiminor axis and is equal 1/4*a**2*M where a is semimajor axis
//		double radical=0.5*sqrt((_cell->iXX-_cell->iZZ)*(_cell->iXX-_cell->iZZ)+4.0*_cell->iXZ*_cell->iXZ);	
//		double lMin=0.5*(_cell->iXX+_cell->iZZ)-radical;
//		double lMax=0.5*(_cell->iXX+_cell->iZZ)+radical;
//		_majorAxis=2*sqrt(lMax/_cell->volume);
//		_medianAxis=0.0;
//	    _minorAxis=2*sqrt(lMin/_cell->volume);
//}
//
//void MomentOfInertiaPlugin::getSemiaxesYZ(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis){
//	    //in the case of an ellipse larger moment of inertia is w.r.t. semiminor axis and is equal 1/4*a**2*M where a is semimajor axis
//		double radical=0.5*sqrt((_cell->iYY-_cell->iZZ)*(_cell->iYY-_cell->iZZ)+4.0*_cell->iYZ*_cell->iYZ);	
//		double lMin=0.5*(_cell->iYY+_cell->iZZ)-radical;
//		double lMax=0.5*(_cell->iYY+_cell->iZZ)+radical;
//		_majorAxis=2*sqrt(lMax/_cell->volume);
//		_medianAxis=0.0;
//	    _minorAxis=2*sqrt(lMin/_cell->volume);
//}
//
//
//void MomentOfInertiaPlugin::getSemiaxes3D(CellG *_cell,double & _majorAxis , double & _medianAxis, double &_minorAxis){
//
//	 vector<double> aCoeff(4,0.0);	 
//	 vector<complex<double> > roots;
//	 //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
//	 aCoeff[0]=-1.0;
//
//	 aCoeff[1]=_cell->iXX + _cell->iYY + _cell->iZZ;
//
//	 aCoeff[2]=_cell->iXY*_cell->iXY + _cell->iXZ*_cell->iXZ + _cell->iYZ*_cell->iYZ
//	 -_cell->iXX*_cell->iYY - _cell->iXX*_cell->iZZ - _cell->iYY*_cell->iZZ;
//
//	 aCoeff[3]=_cell->iXX*_cell->iYY*_cell->iZZ + 2*_cell->iXY*_cell->iXZ*_cell->iYZ
//	 -_cell->iXX*_cell->iYZ*_cell->iYZ
//	 -_cell->iYY*_cell->iXZ*_cell->iXZ
//	 -_cell->iZZ*_cell->iXY*_cell->iXY;
//
//	 roots=solveCubicEquationRealCoeeficients(aCoeff);
//
//	 //finding semiaxes of the ellipsoid
//	 //Ixx=m/5.0*(a_y^2+a_z^2) - andy cyclical permutations for other coordinate combinations
//	 //a_x,a_y,a_z are lengths of semiaxes of the allipsoid
//	 // We can invert above system of equations to get:
//	vector<double> axes(3,0.0);
//
//	axes[0]=sqrt((2.5/_cell->volume)*(roots[1].real()+roots[2].real()-roots[0].real()));
//	axes[1]=sqrt((2.5/_cell->volume)*(roots[0].real()+roots[2].real()-roots[1].real()));
//	axes[2]=sqrt((2.5/_cell->volume)*(roots[0].real()+roots[1].real()-roots[2].real()));
//
//	//sorting semiaxes according the their lengths (shortest first)
//	sort(axes.begin(),axes.end());
//	_majorAxis=axes[2];
//	_medianAxis=axes[1];
//	_minorAxis=axes[0];
//
//}

vector<double> MomentOfInertiaPlugin::getSemiaxes(CellG *_cell)
{

	return (this->*getSemiaxesFctPtr)(_cell);
}

vector<double> MomentOfInertiaPlugin::getSemiaxesXY(CellG *_cell){
	    //in the case of an ellipse larger moment of inertia is w.r.t. semiminor axis and is equal 1/4*a**2*M where a is semimajor axis
		double radical=0.5*sqrt((_cell->iXX-_cell->iYY)*(_cell->iXX-_cell->iYY)+4.0*_cell->iXY*_cell->iXY);	
		double lMin=0.5*(_cell->iXX+_cell->iYY)-radical;
		double lMax=0.5*(_cell->iXX+_cell->iYY)+radical;
		vector<double> axes(3,0);
		if (fabs(lMin)<0.000001){ //to deal with round off errors
			lMin=0.0;
		}
		axes[0]=2*sqrt(lMin/_cell->volume); //semiminor axis
		axes[1]=0.0; //semimedian axis
		axes[2]=2*sqrt(lMax/_cell->volume); //semiminor axis
		//if (lMin<0){
		//	cerr<<"\t\t\t lMin="<<lMin<<" axes[0]="<<axes[0]<<endl;
		//}
		return axes;

}

vector<double> MomentOfInertiaPlugin::getSemiaxesXZ(CellG *_cell){
	    //in the case of an ellipse larger moment of inertia is w.r.t. semiminor axis and is equal 1/4*a**2*M where a is semimajor axis
		double radical=0.5*sqrt((_cell->iXX-_cell->iZZ)*(_cell->iXX-_cell->iZZ)+4.0*_cell->iXZ*_cell->iXZ);	
		double lMin=0.5*(_cell->iXX+_cell->iZZ)-radical;
		double lMax=0.5*(_cell->iXX+_cell->iZZ)+radical;
		vector<double> axes(3,0);
		if (fabs(lMin)<0.000001){ //to deal with round off errors
			lMin=0.0;
		}
		axes[0]=2*sqrt(lMin/_cell->volume); //semiminor axis
		axes[1]=0.0; //semimedian axis
		axes[2]=2*sqrt(lMax/_cell->volume); //semiminor axis

		//if (lMin<0){
		//	cerr<<"\t\t\t lMin="<<lMin<<" axes[0]="<<axes[0]<<endl;
		//}
		return axes;
}

vector<double> MomentOfInertiaPlugin::getSemiaxesYZ(CellG *_cell){
	    //in the case of an ellipse larger moment of inertia is w.r.t. semiminor axis and is equal 1/4*a**2*M where a is semimajor axis
		double radical=0.5*sqrt((_cell->iYY-_cell->iZZ)*(_cell->iYY-_cell->iZZ)+4.0*_cell->iYZ*_cell->iYZ);	
		double lMin=0.5*(_cell->iYY+_cell->iZZ)-radical;
		double lMax=0.5*(_cell->iYY+_cell->iZZ)+radical;
		vector<double> axes(3,0);
		if (fabs(lMin)<0.000001){ //to deal with round off errors
			lMin=0.0;
		}

		axes[0]=2*sqrt(lMin/_cell->volume); //semiminor axis
		axes[1]=0.0; //semimedian axis
		axes[2]=2*sqrt(lMax/_cell->volume); //semiminor axis
		//if (lMin<0){
		//	cerr<<"\t\t\t lMin="<<lMin<<" axes[0]="<<axes[0]<<endl;
		//}
		return axes;
}


vector<double> MomentOfInertiaPlugin::getSemiaxes3D(CellG *_cell){

	 vector<double> aCoeff(4,0.0);	 
	 vector<complex<double> > roots;
	 //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
	 aCoeff[0]=-1.0;

	 aCoeff[1]=_cell->iXX + _cell->iYY + _cell->iZZ;

	 aCoeff[2]=_cell->iXY*_cell->iXY + _cell->iXZ*_cell->iXZ + _cell->iYZ*_cell->iYZ
	 -_cell->iXX*_cell->iYY - _cell->iXX*_cell->iZZ - _cell->iYY*_cell->iZZ;

	 aCoeff[3]=_cell->iXX*_cell->iYY*_cell->iZZ + 2*_cell->iXY*_cell->iXZ*_cell->iYZ
	 -_cell->iXX*_cell->iYZ*_cell->iYZ
	 -_cell->iYY*_cell->iXZ*_cell->iXZ
	 -_cell->iZZ*_cell->iXY*_cell->iXY;

	 roots=solveCubicEquationRealCoeeficients(aCoeff);

	 //finding semiaxes of the ellipsoid
	 //Ixx=m/5.0*(a_y^2+a_z^2) - andy cyclical permutations for other coordinate combinations
	 //a_x,a_y,a_z are lengths of semiaxes of the allipsoid
	 // We can invert above system of equations to get:
	vector<double> axes(3,0.0);

	axes[0]=sqrt((2.5/_cell->volume)*(roots[1].real()+roots[2].real()-roots[0].real()));
	axes[1]=sqrt((2.5/_cell->volume)*(roots[0].real()+roots[2].real()-roots[1].real()));
	axes[2]=sqrt((2.5/_cell->volume)*(roots[0].real()+roots[1].real()-roots[2].real()));

	//sorting semiaxes according the their lengths (shortest first)
	sort(axes.begin(),axes.end());

     return axes;

	// _majorAxis=axes[2];
	//_medianAxis=axes[1];
	//_minorAxis=axes[0];

}
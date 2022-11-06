#include "CellMomentOfInertia.h"

#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include "MomentOfInertiaPlugin.h"
#include <Logger/CC3DLogger.h>

using namespace std;

std::vector<double> CompuCell3D::minMaxComps(const double& i11, const double& i22, const double& i12) {
	double radical = 0.5*sqrt((i11-i22)*(i11-i22)+4.0*i12*i12);
	return std::vector<double>{0.5*(i11+i22)-radical, 0.5*(i11+i22)+radical};  // lMin, lMax
}

double CompuCell3D::eccFromComps(const double& lMin, const double& lMax) {
	double ratio=lMin/lMax;
	return fabs(ratio)>1.0 ? 1.0 : sqrt(1.0-ratio);
}

std::vector<double> CompuCell3D::cellOrientation_12(const double& i11, const double& i22, const double& i12) {
	auto lComps = CompuCell3D::minMaxComps(i11, i22, i12);
	return std::vector<double>{CompuCell3D::eccFromComps(lComps[0], lComps[1]), i12, lComps[1]-i11}; // ecc, l1, l2
}

vector<double> CompuCell3D::getSemiaxes12(const double& i11, const double& i22, const double& i12, const double& volume) {
	//in the case of an ellipse larger moment of inertia is w.r.t. semiminor axis and is equal 1/4*a**2*M
    // where a is semimajor axis
	auto lComps = CompuCell3D::minMaxComps(i11, i22, i12);
	auto lMin = fabs(lComps[0])<0.000001 ? 0.0 : lComps[0]; //to deal with round off errors
	return vector<double>{2*sqrt(lMin/volume), 2*sqrt(lComps[1]/volume)}; // semimajor axis, semimajor axis
}

MomentOfInertiaPlugin::MomentOfInertiaPlugin():potts(0),simulator(0),boundaryStrategy(0),lastMCSPrintedWarning(-1),cellOrientationFcnPtr(0) {}

MomentOfInertiaPlugin::~MomentOfInertiaPlugin() {}

void MomentOfInertiaPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	CC3D_Log(LOG_DEBUG) << std::endl << std::endl << std::endl << "  \t\t\t CALLING INIT OF MOMENT OF INERTIA PLUGIN" << std::endl << std::endl << std::endl;
	this->simulator=simulator;
	potts = simulator->getPotts();
	bool pluginAlreadyRegisteredFlag;
    //this will load VolumeTracker plugin if it is not already loaded
	Plugin *plugin=Simulator::pluginManager.get("CenterOfMass",&pluginAlreadyRegisteredFlag);
	if(!pluginAlreadyRegisteredFlag)
		plugin->init(simulator);

	potts->registerCellGChangeWatcher(this);

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

	//to calculate CM for the case with periodic boundary conditions we need
    // to do some translations rather than naively calculate centroids
	//naive calculations work in the case of no flux boundary conditions but periodic b.c. may cause troubles

	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;

	Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);

	//if no boundary conditions are present
	int currentStep=simulator->getStep();
	if( !(currentStep %100)	&& lastMCSPrintedWarning < currentStep){
		lastMCSPrintedWarning = currentStep;
		if ( boundaryConditionIndicator.x || boundaryConditionIndicator.y || boundaryConditionIndicator.z ){
			CC3D_Log(LOG_DEBUG) << "MomentOfInertia plugin may not work properly with periodic boundary conditions.Pleas see manual when it is OK to use this plugin with periodic boundary conditions";
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
	//calculating cell orientation parameters eccentricity,orientation vector
	if(cellOrientationFcnPtr){ //this will call cell orientation calculations only when simulation is 2D
		(this->*cellOrientationFcnPtr)(pt,newCell,oldCell);
	}

	return;

	//if there are boundary conditions defined that we have to do some shifts to correctly calculate center of mass
	//This approach will work only for cells whose span is much smaller that lattice dimension in the "periodic "direction
	//e.g. cell that is very long and "wraps lattice" will have miscalculated CM using this algorithm.
    // On the other hand, you do not real expect
	//cells to have dimensions comparable to lattice...

	//THIS IS NOT IMPLEMENTED YET. WE WILL DO IT IN THE ONE OF THE COMING RELEASES
}

string MomentOfInertiaPlugin::toString(){return "MomentOfInertia";}

void MomentOfInertiaPlugin::cellOrientation_xy(const Point3D &pt, CellG *newCell,CellG *oldCell){
	if(newCell){
		auto c = cellOrientation_12(newCell->iXX, newCell->iYY, newCell->iXY);
		newCell->ecc = (float)c[0];
		newCell->lX = (float)c[1];
		newCell->lY = (float)c[2];
	}
	if(oldCell){
		auto c = cellOrientation_12(oldCell->iXX, oldCell->iYY, oldCell->iXY);
		oldCell->ecc = (float)c[0];
		oldCell->lX = (float)c[1];
		oldCell->lY = (float)c[2];
	}

}


void MomentOfInertiaPlugin::cellOrientation_xz(const Point3D &pt, CellG *newCell,CellG *oldCell){
	if(newCell){
		auto c = cellOrientation_12(newCell->iXX, newCell->iZZ, newCell->iXZ);
		newCell->ecc = (float)c[0];
		newCell->lX = (float)c[1];
		newCell->lZ = (float)c[2];
	}
	if(oldCell){
		auto c = cellOrientation_12(oldCell->iXX, oldCell->iZZ, oldCell->iXZ);
		oldCell->ecc = (float)c[0];
		oldCell->lX = (float)c[1];
		oldCell->lZ = (float)c[2];
	}
}

void MomentOfInertiaPlugin::cellOrientation_yz(const Point3D &pt, CellG *newCell,CellG *oldCell){
	if(newCell){
		auto c = cellOrientation_12(newCell->iYY, newCell->iZZ, newCell->iYZ);
		newCell->ecc = (float)c[0];
		newCell->lY = (float)c[1];
		newCell->lZ = (float)c[2];
	}
	if(oldCell){
		auto c = cellOrientation_12(oldCell->iYY, oldCell->iZZ, oldCell->iYZ);
		oldCell->ecc = (float)c[0];
		oldCell->lY = (float)c[1];
		oldCell->lZ = (float)c[2];
	}
}

vector<double> MomentOfInertiaPlugin::getSemiaxes(CellG *_cell)
{

	return (this->*getSemiaxesFctPtr)(_cell);
}

vector<double> MomentOfInertiaPlugin::getSemiaxesXY(CellG *_cell){
	auto sa = getSemiaxes12(_cell->iXX, _cell->iYY, _cell->iXY, _cell->volume);
	return vector<double>{sa[0], 0.0, sa[1]};
}

vector<double> MomentOfInertiaPlugin::getSemiaxesXZ(CellG *_cell){
	auto sa = getSemiaxes12(_cell->iXX, _cell->iZZ, _cell->iXZ, _cell->volume);
	return vector<double>{sa[0], 0.0, sa[1]};
}

vector<double> MomentOfInertiaPlugin::getSemiaxesYZ(CellG *_cell){
	auto sa = getSemiaxes12(_cell->iYY, _cell->iZZ, _cell->iYZ, _cell->volume);
	return vector<double>{sa[0], 0.0, sa[1]};
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
}
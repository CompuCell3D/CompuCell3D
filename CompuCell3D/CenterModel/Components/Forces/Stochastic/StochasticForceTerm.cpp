#include "StochasticForceTerm.h"
#include <PublicUtilities/NumericalUtils.h>
#include <Components/SimulatorCM.h>
#include <XMLUtils/CC3DXMLElement.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <limits>
#include <cmath>


using namespace CenterModel;
using namespace std;





StochasticForceTerm::StochasticForceTerm():mag_min(0.0),mag_max(1.0),PI(acos(-1.0)){
    PI_HALF=PI/2.0;

}




StochasticForceTerm::~StochasticForceTerm(){}



void StochasticForceTerm::init(SimulatorCM *_simulator, CC3DXMLElement * _xmlData){

    if (!_simulator)
        return;

	srand((unsigned)time(0));
    //srand(time( NULL ));
	unsigned int randomSeed=(unsigned int)rand()*((std::numeric_limits<unsigned int>::max)()-1);                
	rGen.setSeed(randomSeed);
    
    xmlData=_xmlData;
    if (xmlData)
        update(xmlData);
        
    simulator=_simulator;
    simulator->registerForce(this);

    
}

Vector3 StochasticForceTerm::forceTerm(const CellCM * _cell1, const CellCM * _cell2, double _distance, const Vector3 & _unitDistVec){
    
    double mag=mag_min+rGen.getRatio()*(mag_max-mag_min);
    double theta=-PI_HALF+rGen.getRatio()*PI;
    double phi=2.0*PI*rGen.getRatio();
    
    
    Vector3 stochForce;
    stochForce.SetMagThetaPhi(mag,  theta,  phi);
    //cerr<<"stochForce= "<<stochForce<<endl;
    return stochForce;
    
    
}

void StochasticForceTerm::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

    CC3DXMLElement * magminElem=_xmlData->getFirstElement("MagnitudeMin");
    if (magminElem){
        mag_min=magminElem->getDouble();
    }

    CC3DXMLElement * magmaxElem=_xmlData->getFirstElement("MagnitudeMax");
    if (magmaxElem){
        mag_max=magmaxElem->getDouble();
    }


}
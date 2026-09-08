
#include <CompuCell3D/CC3D.h>


using namespace CompuCell3D;


using namespace std;


#include "PolarizationVectorPlugin.h"

extern "C" void cc3d_register_polarization_vector_plugin(void *, void *);
extern "C" void cc3d_unregister_polarization_vector_callbacks(void *);

PolarizationVectorPlugin::PolarizationVectorPlugin() {}

PolarizationVectorPlugin::~PolarizationVectorPlugin() { cc3d_unregister_polarization_vector_callbacks(this); }

void PolarizationVectorPlugin::setPolarizationVector(CellG *_cell, float _x, float _y, float _z) {
    polarizationVectorAccessor.get(_cell->extraAttribPtr)->x = _x;
    polarizationVectorAccessor.get(_cell->extraAttribPtr)->y = _y;
    polarizationVectorAccessor.get(_cell->extraAttribPtr)->z = _z;
}

vector<float> PolarizationVectorPlugin::getPolarizationVector(CellG *_cell) {
    vector<float> polarizationVec(3, 0.0);
    polarizationVec[0] = polarizationVectorAccessor.get(_cell->extraAttribPtr)->x;
    polarizationVec[1] = polarizationVectorAccessor.get(_cell->extraAttribPtr)->y;
    polarizationVec[2] = polarizationVectorAccessor.get(_cell->extraAttribPtr)->z;
    return polarizationVec;
}

void PolarizationVectorPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    cc3d_register_polarization_vector_plugin(simulator, this);
    Potts3D *potts = simulator->getPotts();
    potts->getCellFactoryGroupPtr()->registerClass(&polarizationVectorAccessor); //register new class with the factory
}

void PolarizationVectorPlugin::extraInit(Simulator *simulator) {
}




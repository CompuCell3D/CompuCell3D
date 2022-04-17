

#ifndef CELLDIFFUSIONDATAOUTPUT_H
#define CELLDIFFUSIONDATAOUTPUT_H

#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <plugins/NeighborTracker/NeighborTracker.h>
#include <string>
#include <vector>
#include <fstream>

#include <Utils/Coordinates3D.h>

template <typename Y> class BasicClassAccessor;

namespace CompuCell3D {
  class Potts3D;
  class CellInventory;

   
  class CellDiffusionDataOutput : public Steppable {
    Potts3D *potts;
    CellInventory * cellInventoryPtr;
    Dim3D dim;

    std::string fileName;
    bool cellIDFlag;
    bool deltaPositionFlag;
    std::vector<Coordinates3D<float> > cellPositions;
    std::vector<long int> cellIds;
    std::vector<CellG*> cellIdsPtrs;
    std::vector<std::ofstream *> filePtrVec;

  public:
    CellDiffusionDataOutput();
    
    virtual ~CellDiffusionDataOutput();
    void setPotts(Potts3D *potts) {this->potts = potts;}


    // SimObject interface
    virtual void init(Simulator *simulator);
    virtual void extraInit(Simulator *simulator);
    // Begin Steppable interface
    virtual void start();
    virtual void step(const unsigned int currentStep);
    virtual void finish() {}
    // End Steppable interface

    // Begin XMLSerializable interface
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    // End XMLSerializable interface
  };
};
#endif

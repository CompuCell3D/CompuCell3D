

#ifndef CHEMOTAXISSIMPLEENERGY_H
#define CHEMOTAXISSIMPLEENERGY_H

 #include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>


// // // #include <CompuCell3D/Potts3D/Cell.h>

// // // #include <string>
// // // #include <vector>

#include "ChemotaxisSimpleDLLSpecifier.h"


namespace CompuCell3D {

  class Potts3D;
  class Simulator; 



  class CHEMOTAXISSIMPLE_EXPORT ChemotaxisSimpleEnergy   {

    Potts3D *potts;
    Simulator *sim;

      
    
  public:
    
    ChemotaxisSimpleEnergy() :potts(0),sim(0) {}
    float simpleChemotaxisFormula(float _flipNeighborConc,float _conc,double _lambda);

    virtual ~ChemotaxisSimpleEnergy() {}

    void setSimulatorPtr(Simulator * _sim){sim=_sim;}
    


  


  };

  
};

#endif

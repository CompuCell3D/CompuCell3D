#ifndef COMPUCELL3DCONCENTRATION_H
#define COMPUCELL3DCONCENTRATION_H

namespace CompuCell3D {

/**
@author m
*/
class Concentration{
public:
    Concentration():concentration(0),flag(0)
    {
      
    };
   float concentration;
   char flag;
};

};

#endif

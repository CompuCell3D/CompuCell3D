#ifndef CELLMOMENTOFINERTIA_H
#define CELLMOMENTOFINERTIA_H

#include "MomentOfInertiaDLLSpecifier.h"

namespace CompuCell3D {

  class MOMENTOFINERTIA_EXPORT CellMomentOfInertia {
  public:
    /// Total of all X values.  Divide by volume to get the center of mass.
     unsigned int iXX;
     unsigned int iXY;

    /// Total of all Y values.  Divide by volume to get the center of mass.
     unsigned int iYY;
     unsigned int iYZ;

    /// Total of all Z values.  Divide by volume to get the center of mass.
    unsigned int iZZ;
    unsigned int iXZ;
  };
};
#endif

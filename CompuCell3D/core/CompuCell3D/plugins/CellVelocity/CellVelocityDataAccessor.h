#ifndef CELLVELOCITYDATAACCESSOR_H
#define CELLVELOCITYDATAACCESSOR_H

#include "CellVelocityData.h"
#include <BasicUtils/BasicClassAccessor.h>
#include "CellVelocityDataClassFactory.h"

template<class T>
class CellVelocityDataAccessor : public BasicClassAccessor<T> {
public:
    CellVelocityDataAccessor(cldeque<Coordinates3D < float>

    >
    ::size_type _cldequeCapacity,
            cldeque<Coordinates3D < float>
    >
    ::size_type _enoughDataThreshold
    ):

    BasicClassAccessor<T>(),
    cldequeCapacity(_cldequeCapacity),
    enoughDataThreshold(_enoughDataThreshold) {}

protected:

    virtual BasicClassFactoryBase<void> *createClassFactory() {
        return new CellVelocityDataClassFactory<void, T>(cldequeCapacity, enoughDataThreshold);
    }

private:
    cldeque<Coordinates3D < float> >
    ::size_type cldequeCapacity;
    cldeque<Coordinates3D < float> >
    ::size_type enoughDataThreshold;

};

#endif

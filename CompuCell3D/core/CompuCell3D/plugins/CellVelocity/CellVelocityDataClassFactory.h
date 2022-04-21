#ifndef CELLVELOCITYDATACLASSFACTORY_H
#define CELLVELOCITYDATACLASSFACTORY_H

#include <BasicUtils/BasicClassFactory.h>
#include "CellVelocityData.h"


template<class B, class T>
class CellVelocityDataClassFactory : public BasicClassFactory<B, T> {
public:

    CellVelocityDataClassFactory(cldeque<Coordinates3D < float>

    >
    ::size_type _cldequeCapacity,
            cldeque<Coordinates3D < float>
    >
    ::size_type _enoughDataThreshold
    ):

    BasicClassFactory<B, T>(),
    cldequeCapacity(_cldequeCapacity),
    enoughDataThreshold(_enoughDataThreshold) {}

    /**
     * @return A pointer to a newly allocated instance of class T.
     */
    virtual B *create() {
        return new T(cldequeCapacity, enoughDataThreshold);

    }

    /**
     * @param classNode A pointer to the instance of class T to deallocate.
     */
    virtual void destroy(B *classNode) { delete classNode; }

private:
    cldeque<Coordinates3D < float> >
    ::size_type cldequeCapacity;
    cldeque<Coordinates3D < float> >
    ::size_type enoughDataThreshold;

};


#endif

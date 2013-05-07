#ifndef ROADRUNNERADAPTOR_H
#define ROADRUNNERADAPTOR_H

#include <vector>
#include <map>
#include <string>
#include <rrRoadRunner.h>

#include "RoadRunnerAdaptorDLLSpecifier.h"

class ROADRUNNERADAPTOR_EXPORT  RoadRunnerAdaptor{

public:

    RoadRunnerAdaptor();
    virtual ~RoadRunnerAdaptor();

protected:
	rr::RoadRunner *rrPtr;

};

#endif

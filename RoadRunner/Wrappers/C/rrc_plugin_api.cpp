
#pragma hdrstop
//---------------------------------------------------------------------------
#include <sstream>
#include "rrParameter.h"
#include "rrRoadRunner.h"
#include "rrPluginManager.h"
#include "rrPlugin.h"
#include "rrCGenerator.h"
#include "rrLogger.h"           //Might be useful for debugging later on
#include "rrc_api.h"
#include "rrc_support.h"   //Support functions, not exposed as api functions and or data
#include "rrException.h"
#include "rrUtils.h"
#include "rrStringUtils.h"


using namespace rrc;
using namespace rr;
using rr::Exception;
rr::RoadRunner*     gRRHandle;



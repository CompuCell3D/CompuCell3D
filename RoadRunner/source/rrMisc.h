#ifndef rrMiscH
#define rrMiscH
#include <string>
#include <iomanip>
#include <ostream>
#include "rrExporter.h"
#include "rrObject.h"
using std::ostream;

using std::string;
using std::endl;
//---------------------------------------------------------------------------
namespace rr
{

enum TSelectionType
{
        clTime = 0,
        clBoundarySpecies,
        clFloatingSpecies,
        clFlux,
        clRateOfChange,
        clVolume,
        clParameter,
/*7*/   clFloatingAmount,
/*8*/   clBoundaryAmount,
        clElasticity,
        clUnscaledElasticity,
        clEigenValue,
        clUnknown,
        clStoichiometry
};

class RR_DECLSPEC TSelectionRecord : public rrObject
{
    public:
        unsigned int        index;
        string              p1;
        string              p2;
        TSelectionType      selectionType;
                            TSelectionRecord(const int& index = 0, const TSelectionType type = clUnknown, const string& p1 = gEmptyString, const string& p2 = gEmptyString);

};

ostream& operator<< (ostream& stream, const TSelectionRecord& rec);
}


#endif

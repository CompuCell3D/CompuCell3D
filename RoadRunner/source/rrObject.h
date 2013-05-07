#ifndef rrObjectH
#define rrObjectH
#include <string>
#include "rrExporter.h"
#include "rrConstants.h"
namespace rr
{
using namespace std;

//Have all RoadRunner classes descending from a rrObject
class RR_DECLSPEC rrObject
{
    protected:

    public:
                        rrObject();
        virtual        ~rrObject();
};

}
#endif

#ifndef rrIniKeyH
#define rrIniKeyH
#include <vector>
#include <fstream>
#include <string>
#include <complex>
#include "rrObject.h"
#include "rrStringUtils.h"
#include "rrIniSection.h"


using namespace rr;
using std::string;
using std::complex;

namespace rr
{

// This class stores the definition of a key. A key is a named identifier
// that is associated with a value. It may or may not have a comment.  All comments
// must PRECEDE the key on the line in the config file.
class RR_DECLSPEC rrIniKey : public rrObject
{
    protected:
        void SetupKey(const string& key);

	public:
		string	              	mKey;
		string	              	mValue;
		string	              	mComment;

				              	rrIniKey(const string& key = gEmptyString);
				               ~rrIniKey(){}
        void 	              	ReKey(const string& key);
        string 	              	AsString() const;
        int 	              	AsBool() const;
        int 	              	AsInt() const;
        double 	              	AsFloat() const;
        complex<double> 		AsComplex() const;
		RR_DECLSPEC
        friend ostream& 		operator<<(ostream& stream, const rrIniKey& aKey);
};
}

#endif

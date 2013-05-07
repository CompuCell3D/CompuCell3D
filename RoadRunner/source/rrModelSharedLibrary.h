#ifndef rrModelSharedLibraryH
#define rrModelSharedLibraryH
//---------------------------------------------------------------------------
#include "Poco/SharedLibrary.h"
#include "rrObject.h"

namespace rr
{

using Poco::SharedLibrary;

class RR_DECLSPEC ModelSharedLibrary : public rrObject
{
	protected:
		string 							mLibName;
		string							mPathToLib;
		SharedLibrary					mTheLib;

    public:
										ModelSharedLibrary(const string& pathToLib = gEmptyString);
                                       ~ModelSharedLibrary();

		bool							setPath(const string& pathTo);
    	string							createName(const string& name = gEmptyString);
		string							getName();
		string							getFullFileName();

        bool							load();
        bool							load(const string& name);
        bool							unload();
		bool							isLoaded();
        void*							getSymbol(const string& name);
        bool							hasSymbol(const string& name);

};
}
#endif

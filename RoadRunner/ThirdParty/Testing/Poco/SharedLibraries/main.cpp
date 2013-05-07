#pragma hdrstop
#pragma argsused
#if defined(WIN32)
#include <conio.h>
#endif

#include <iostream>
#include "Poco/SharedLibrary.h"

using namespace std;
using namespace Poco;

typedef void (*HelloFunc)();
#pragma comment(lib, "IPHLPAPI.lib")

#if defined(CG_IDE)
#pragma comment(lib, "poco_foundation-static.lib")
#endif

int main(int argc, char* argv[])
{
	string theLib("/usr/local/lib/libSharedLibTest");
	theLib.append(SharedLibrary::suffix());
 	string theFunc("poco_hello");

	cout<<"Trying to load shared library:"<<theLib<<endl;
    SharedLibrary lib(theLib);

    if(lib.isLoaded())
    {
        cout<<"The lib: "<<theLib<<" was loaded\n";
    }
    else
    {
        cout<<"Failed loading lib: "<<theLib<<"\n";
    }

    if(lib.hasSymbol(theFunc))
    {
        HelloFunc func = (HelloFunc) lib.getSymbol(theFunc);
        func();
    }
    else
    {
        cout<<"Could not find symbol: "<<theFunc<<endl;
    }

#if defined(WIN32)
    cout<<"\nPress any key to exit...";
    cin.ignore(0,'\n');
    getch();
#endif

	return 0;
}

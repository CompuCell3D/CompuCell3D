#ifndef CommandLineParametersH
#define CommandLineParametersH
#include <string>
using std::string;

string Usage(const string& prg);
class Args
{
    public:
                                        Args();
        virtual                        ~Args(){}
        string                          SBMLModelsFilePath;                       	//option m:
        string                          TempDataFolder;                           	//option t:
		bool                            EnableLogging;                    	     	//option v:
        int								ModelNumber;								//option i:
};

#endif

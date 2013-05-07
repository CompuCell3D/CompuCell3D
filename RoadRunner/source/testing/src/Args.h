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
//		string                          Compiler;       		                  	//option l:
        string                          ResultOutputFile;                         	//option r:
        string                          TempDataFolder;                           	//option t:
//        string                          DataOutputFolder;                          	//option d:
//		string                          SupportCodeFolder;                        	//option s:
		bool                            EnableLogging;                    	     	//option v:
};

#endif

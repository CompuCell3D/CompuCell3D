#ifndef rrStreamWriterH
#define rrStreamWriterH
#include <fstream>
#include <string>
#include "rrObject.h"

using std::ofstream;
using std::string;

namespace rr
{

class RR_DECLSPEC StreamWriter : public rrObject
{
    protected:
        string             mFilePath;
        ofstream         mFileStream;

    public:
                        StreamWriter(const string& filePath);
        bool             WriteLine(const string& line);
        bool             Write(const string& text);
        bool             Close();

};

} //ns rr
#endif

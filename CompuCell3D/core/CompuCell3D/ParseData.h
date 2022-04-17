#ifndef PARSEDATA_H
#define PARSEDATA_H


#include <string>


namespace CompuCell3D {

    class ParseData {
    public:
        ParseData(std::string _moduleName = "") : moduleName(_moduleName), frequency(1) {}

        std::string moduleName;
        unsigned int frequency;

        void Frequency(unsigned int _frequency) { frequency = _frequency; }

        std::string ModuleName() { return moduleName; }

    };


};
#endif

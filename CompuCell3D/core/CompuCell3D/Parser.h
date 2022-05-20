#ifndef PARSER_H
#define PARSER_H

#include <XMLCereal/XMLSerializable.h>
#include <CompuCell3D/ParseData.h>
#include <vector>

namespace CompuCell3D {


    class Parser {
    public:
        std::vector<ParseData *> pluginParseDataVector;
        std::vector<ParseData *> steppableParseDataVector;
        ParseData *pottsParseData;

        virtual void readXML(XMLPullParser &in) {}

        virtual void writeXML(XMLSerializer &out) {}

    };


};
#endif

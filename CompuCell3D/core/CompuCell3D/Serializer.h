#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <string>

namespace CompuCell3D {

    class Serializer {
    public:
        Serializer() {}

        virtual ~Serializer() {}

        virtual void serialize() {}

        virtual void readFromFile() {}

        std::string fileName;
        std::string auxPath;
        std::string serializedFileExtension;
    };

}


#endif
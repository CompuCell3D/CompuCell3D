#ifndef FIELD3DIO_H
#define FIELD3DIO_H

#include "Field3D.h"
#include "Point3D.h"
#include "Dim3D.h"

#include <fstream>
#include <string>

#if defined(_WIN32)
//#include <netinet/in.h>
#else

#include <netinet/in.h>

#endif


#include <CompuCell3D/CC3DExceptions.h>

namespace CompuCell3D {

    /**
     * Generic operator for writing Field3D data in a platform independent
     * way.
     *
     * The format is as follows:
     * <table>
     * <tr><td><strong>Bytes <td><strong>Type <td><strong>Description
     * <tr><td>2 <td>char <td>Data type string.
     * <tr><td>1 <td>char <td>Number of dimensions. Always 3.
     * <tr><td>4 <td>uint32_t <td>X dimension.
     * <tr><td>4 <td>uint32_t <td>Y dimension.
     * <tr><td>4 <td>uint32_t <td>Z dimension.
     * <tr><td>X*Y*Z*sizeof(data type) <td>data type according to the type string
           <td>The field data
     * </table>
     *
     * The predefined data type strings are as follows:
     * <table>
     * <tr><td>" i" <td>4 byte signed integer
     * <tr><td>"ui" <td>4 byte unsigned integer
     * <tr><td>" d" <td>8 byte double
     * <tr><td>" f" <td>4 byte float
     * <tr><td>" c" <td>1 byte signed character
     * <tr><td>"uc" <td>1 byte unsigned character
     * <tr><td>" l" <td>4 byte signed integer
     * <tr><td>"ul" <td>4 byte unsigned integer
     * </table>
     *
     * If you what to write a field of a data type not listed in this table you
     * must define const char Field3D<T>::typeStr[3] for that type somewhere in
     * your code or you will get link errors.
     *
     * All data is written in network byte order.
     * (ie. Most significant byte first)
     *
     * Long data types and pointers are different on 64 bit machines.
     * Such data fields will be incompatable between 64 bit and 32 bit machines.
     *
     * @param stream The output stream.
     * @param field The field to write.
     *
     * @return A reference to the passed stream.
     */
    template<class T>
    std::ofstream &operator<<(std::ofstream &stream, const Field3D<T> &field) {
        stream.write(field.typeStr, 2);
        Dim3D dim = field.getDim();

        char dims = 3;
        uint32_t x = htonl(dim.x);
        uint32_t y = htonl(dim.y);
        uint32_t z = htonl(dim.z);
        stream.write(&dims, 1);
        stream.write((char *) &x, sizeof(uint32_t));
        stream.write((char *) &y, sizeof(uint32_t));
        stream.write((char *) &z, sizeof(uint32_t));

        T c;
        Point3D pt;
        for (pt.z = 0; pt.z < dim.z; pt.z++)
            for (pt.y = 0; pt.y < dim.y; pt.y++)
                for (pt.x = 0; pt.x < dim.x; pt.x++) {
                    if (stream.eof()) throw CC3DException("Field3D<T> Error while writing field!");
                    c = field.get(pt);
                    stream.write((char *) &c, sizeof(c));
                }

        return stream;
    }

    /**
     * Generic operator for reading Field3D data in a platform independent
     * way.
     *
     * See std::ofstream &operator<<(std::ofstream &, Field3D<T> &)
     */
    template<class T>
    std::ifstream &operator>>(std::ifstream &stream, Field3D<T> &field) {
        char typeStr[3];
        stream.read(typeStr, 2);
        typeStr[2] = '\0';
        if (!(typeStr[0] == field.typeStr[0] && typeStr[1] == field.typeStr[1]))
            throw CC3DException(std::string("Field3D<T> Type string mismatch on read! ") + "Expected '" +
                                std::string(field.typeStr) + "' read '" + typeStr + "'.");

        char dims = 0;
        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t z = 0;
        stream.read(&dims, 1);
        stream.read((char *) &x, sizeof(uint32_t));
        stream.read((char *) &y, sizeof(uint32_t));
        stream.read((char *) &z, sizeof(uint32_t));
        Dim3D readDim(ntohl(x), ntohl(y), ntohl(z));


        if (dims != 3)
            throw CC3DException("Field3D<T> Read wrong number of dimensions! " + std::to_string((int) dims));

        if (readDim != field.getDim())
            throw CC3DException(
                    std::string("Field3D<T> Wrong dimensions on read! ") + "Expected " + field.getDim() + " got " +
                    readDim + ".");

        bool lByteOrder = false;
        bool sByteOrder = false;
        if (typeStr[1] == 'l' || typeStr[1] == 'i') lByteOrder = true;
        if (typeStr[1] == 's') sByteOrder = true;
        uint32_t l;
        uint16_t s;

        T c;
        Point3D pt;
        for (pt.z = 0; pt.z < readDim.z; pt.z++)
            for (pt.y = 0; pt.y < readDim.y; pt.y++)
                for (pt.x = 0; pt.x < readDim.x; pt.x++) {
                    if (stream.eof()) throw CC3DException("Field3D<T> Error reading file!");

                    if (lByteOrder) {
                        stream.read((char *) &l, sizeof(T));
                        c = (T) ntohl(l);

                    } else if (sByteOrder) {
                        stream.read((char *) &s, sizeof(T));
                        c = (T) ntohs(s);

                    } else {
                        stream.read((char *) &c, sizeof(T));
                    }
                    field.set(pt, c);
                }

        return stream;
    }
};
#endif

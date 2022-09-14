#ifndef COMPUCELL3DDIFFUSABLEGRAPH_H
#define COMPUCELL3DDIFFUSABLEGRAPH_H

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Steppable.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include "DiffusableVector.h"
#include <CompuCell3D/Field3D/Array3D.h>

namespace CompuCell3D {

//template <typename Y> class Field3DImpl;

/**
@author m
*/



    template<typename precision>
    class DiffusableGraph : public CompuCell3D::DiffusableVector<precision> {

    public:
        using CompuCell3D::DiffusableVector<precision>::concentrationFieldNameVector;
        using CompuCell3D::DiffusableVector<precision>::concentrationFieldVector;

        DiffusableGraph() : DiffusableVector<precision>(), concentrationFieldMapVector(0) {};

        virtual ~DiffusableGraph() {

            for (unsigned int i = 0; i < concentrationFieldMapVector.size(); ++i) {
                if (concentrationFieldMapVector[i]) {
                    delete concentrationFieldMapVector[i];
                    concentrationFieldVector[i] = 0;
                }
            }


        }
        //Field3DImpl<precision> * getConcentrationField(unsigned int i){return concentrationFieldVector[i];};

        virtual std::map<CompuCell3D::CellG *, precision> *getConcentrationMapField(const std::string &name) {
            using namespace std;
            for (unsigned int i = 0; i < concentrationFieldNameVector.size(); ++i) {
                if (concentrationFieldNameVector[i] == name)
                    return concentrationFieldMapVector[i];
            }
            return 0;

        };

        virtual void allocateDiffusableFieldVector(unsigned int numberOfFields, Dim3D fieldDim = Dim3D(0, 0, 0)) {
            for (unsigned int i = 0; i < numberOfFields; ++i) {
                concentrationFieldMapVector.push_back(new std::map<CompuCell3D::CellG *, precision>());
                precision val = precision();
                concentrationFieldVector.push_back(new Array3DBordersField3DAdapter<precision>(fieldDim, val));

            }
            concentrationFieldNameVector.assign(numberOfFields, std::string());
        }

    protected:
        std::vector<std::map < CompuCell3D::CellG * , precision> * >
        concentrationFieldMapVector;

    };

};

#endif

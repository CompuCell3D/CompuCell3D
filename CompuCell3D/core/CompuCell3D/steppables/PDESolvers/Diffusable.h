#ifndef COMPUCELL3DDIFFUSABLE_H
#define COMPUCELL3DDIFFUSABLE_H

#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Steppable.h>

namespace CompuCell3D {

    template<typename Y>
    class Field3DImpl;

/**
@author m
*/
    template<typename precision>
    class Diffusable : public Steppable {
    public:
        Diffusable() : concentrationField(0) {};

        virtual ~Diffusable() {

            if (concentrationField) delete concentrationField;
            concentrationField = 0;
        }

        Field3D <precision> *getConcentrationField() { return concentrationField; };

        void allocateDiffusableField(Dim3D fieldDim) { concentrationField = new Field3DImpl<precision>(fieldDim, 0.0); }

    protected:
        Field3D <precision> *concentrationField;
    };

};

#endif

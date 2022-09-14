#ifndef FIRSTORDEREXPANSIONACCEPTANCEFUNCTION_H
#define FIRSTORDEREXPANSIONACCEPTANCEFUNCTION_H

#include "AcceptanceFunction.h"

#include <math.h>

// #include <iostream>
namespace CompuCell3D {

    /**
     * The default Boltzman acceptance function.
     */
    class FirstOrderExpansionAcceptanceFunction : public AcceptanceFunction {
        double k;
        double offset;
        double firstOrderTerm;
    public:
        FirstOrderExpansionAcceptanceFunction(const double _k = 1.0, const double _offset = 0.0) : k(_k),
                                                                                                   offset(_offset) {}

        virtual void setOffset(double _offset) { offset = _offset; }

        virtual void setK(double _k) { k = _k; }

        double accept(const double temp, const double change) {
            if (temp <= 0) {
                if (change > 0) return 0.0;
                if (change == 0) return 0.5;
                return 1.0;

            } else {

                if (change <= offset) return 1.0;

        firstOrderTerm=1.0-(change-offset)/(k*temp);

                if (firstOrderTerm < 0.0)return 0.0;
                else return firstOrderTerm;

            }
        }
    };
};
#endif

#ifndef DEFAULTACCEPTANCEFUNCTION_H
#define DEFAULTACCEPTANCEFUNCTION_H

#include "AcceptanceFunction.h"

#include <math.h>

namespace CompuCell3D {

    /**
     * The default Boltzman acceptance function.
     */
    class DefaultAcceptanceFunction : public AcceptanceFunction {
        double k;
        double offset;

    public:
        DefaultAcceptanceFunction(const double _k = 1.0, const double _offset = 0.0) : k(_k), offset(_offset) {}

        virtual void setOffset(double _offset) { offset = _offset; }

        virtual void setK(double _k) { k = _k; }

        double accept(const double temp, const double change) {
            if (temp <= 0) {
                if (change > 0) return 0.0;
                if (change == 0) return 0.5;
                return 1.0;

            } else {
                if (change <= offset) return 1.0;
                return exp(-(change - offset) / (k * temp));
            }
        }
    };
};
#endif

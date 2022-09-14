
#ifndef ACCEPTANCEFUNCTION_H
#define ACCEPTANCEFUNCTION_H

namespace CompuCell3D {

    /**
     * The Potts3D acceptance function interface.
     *
     * See DefaultAcceptanceFunction.
     */
    class AcceptanceFunction {
    public:

        /**
         * Calculates the probability that a change should be accepted
         * based on the current temperature and the energy cost.
         *
         * @param temp The current temperature.
         * @param change The change energy.
         *
         * @return The probability that the change should be accepted as
         *         a number in the range [0,1).
         */
        virtual double accept(const double temp, const double change) = 0;

        virtual void setOffset(double _offset) = 0;

        virtual void setK(double _k) = 0;

    };
};
#endif

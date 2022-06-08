#ifndef BOUNDARYFACTORY_H
#define BOUNDARYFACTORY_H

#include <string>
#include <iostream>
#include "Boundary.h"
#include "NoFluxBoundary.h"
#include "PeriodicBoundary.h"

namespace CompuCell3D {


    /*
     * Factory class for instantiating  boundary conditions for each axis
     */
    class BoundaryFactory {


    public:

        static const std::string no_flux;
        static const std::string periodic;

        static Boundary *createBoundary(std::string boundary) {

            if (boundary == periodic) {
                return new PeriodicBoundary();

            } else {

                return new NoFluxBoundary();

            }

        }

    };

    const std::string BoundaryFactory::no_flux("NoFlux");
    const std::string BoundaryFactory::periodic("Periodic");

};

#endif

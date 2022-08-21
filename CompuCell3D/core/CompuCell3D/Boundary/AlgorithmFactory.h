#ifndef ALGROITHMFACTORY_H
#define ALGORITHMFACTORY_H

#include <string>
#include <iostream>
#include "Algorithm.h"
#include "ChengbangAlgorithm.h"
#include "DefaultAlgorithm.h"

using namespace std;

namespace CompuCell3D {


    /*
     * Factory class for instantiating  boundary conditions for each axis
     */
    class AlgorithmFactory {


    public:

        static const string chengbang;
        static const string Default;

        static Algorithm *createAlgorithm(string algorithm, int index,
                                          int size, string inputfile) {

            if (algorithm == chengbang) {
                Algorithm *ca = new ChengbangAlgorithm();
                ca->readFile(index, size, inputfile);
                return ca;


            } else {

                return new DefaultAlgorithm();

            }

        }

    };

    const string AlgorithmFactory::chengbang("Chengbang");
    const string AlgorithmFactory::Default("Default");
};

#endif

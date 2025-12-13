#ifndef ALGROITHMFACTORY_H
#define ALGORITHMFACTORY_H

#include <string>
#include <iostream>
#include "Algorithm.h"
#include "ChengbangAlgorithm.h"
#include "DefaultAlgorithm.h"

//using namespace std;

namespace CompuCell3D {


    /*
     * Factory class for instantiating  boundary conditions for each axis
     */
    class AlgorithmFactory {


    public:

        static const std::string chengbang;
        static const std::string Default;

        static Algorithm *createAlgorithm(std::string algorithm, int index,
                                          int size, std::string inputfile) {

            if (algorithm == chengbang) {
                Algorithm *ca = new ChengbangAlgorithm();
                ca->readFile(index, size, inputfile);
                return ca;


            } else {

                return new DefaultAlgorithm();

            }

        }

    };

    const std::string AlgorithmFactory::chengbang("Chengbang");
    const std::string AlgorithmFactory::Default("Default");
};

#endif

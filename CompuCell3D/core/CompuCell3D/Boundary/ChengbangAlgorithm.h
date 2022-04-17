#ifndef CHENGBANGALGORITHM_H
#define CHENGBANGALGORITHM_H

#include "Algorithm.h"

#include <vector>

using std::vector;

namespace CompuCell3D {

    /*
     * Chengbang's Algorithm. 
     */
    class ChengbangAlgorithm : public Algorithm {


    public:
        ChengbangAlgorithm() { evolution = -1; }

        void readFile(const int index, const int size, string inputfile);

        bool inGrid(const Point3D &pt);

        int getNumPixels(int x, int y, int z);

        int i;
        int s;
        string filetoread;
        int evolution;
    private:
        vector <vector<vector < float>> >
        dataStructure;

        void readFile(const char *inputFile);
    };

};


#endif

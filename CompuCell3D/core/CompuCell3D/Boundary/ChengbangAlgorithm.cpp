#include "Algorithm.h"
#include "ChengbangAlgorithm.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <cstdlib>
#include <Logger/CC3DLogger.h>

using namespace CompuCell3D;

/*
 * Read the input file and populate
 *
 * @ return void
 */
void ChengbangAlgorithm::readFile(const char *inputfile) {
    ifstream in(inputfile, ios::in);

    // Make sure the file is open
    if (!in.is_open()) {
        CC3D_Log(LOG_DEBUG) << "Error opening file" << inputfile;
        exit(1);
    }

    // Read the dimensions
    int x, y, z;
    float temp;
    string s;
    getline(in, s);
    istringstream is(s);
    is >> x;
    is >> y;
    is >> z;

    //Initialize the datastructure
    // vector< vector< vector<int> > > ds (x, vector< vector<int> > (y,vector<int>(1)));
    dataStructure.resize(x);
    for (int k = 0; k < x; k++) {
        dataStructure[k].resize(y);
    }


    // Populate the data structure
    for (int i = 0; i < y; i++) { // y axis
        for (int j = 0; j < x; j++) { // x axis
            getline(in, s);
            std::string::size_type pos = s.find_last_not_of(" \t\n\r\0");

            if (pos < s.size()) {
                s.erase(pos + 1, s.size());
            }

            istringstream ss(s);

            while (!ss.eof()) {
                ss >> temp;
                dataStructure[j][i].push_back(temp);
            }
        }
    }
}


/*
 * Read the input file and populate
 * our 3D vector.
 * @ return void.
 */
void ChengbangAlgorithm::readFile(const int index, const int size, string
inputfile) {
    i = index;
    s = size;
    string num;
    string currentfile;

    if (i <= size) {
        char ch[60];
        stringstream ss;
        ss << i;
        ss >> num;
        currentfile = inputfile + num + ".dat";
        sprintf(ch, currentfile.c_str(), i);
        readFile(ch);
        filetoread = inputfile;
    }
    i++;
}

/*
 * Apply Chengbang's algorithm.
 * Return 'true' if the passed point is in the grid.
 *
 */
bool ChengbangAlgorithm::inGrid(const Point3D &pt) {
    bool inside = false;


    // if current step is less than 100 use first shape only
    if (currentStep != evolution) {
        evolution = currentStep;
        if (currentStep % 50 == 0) {

            readFile(i, s, filetoread);
        }
    }
    //Determine the length of the vector
    int length = dataStructure[pt.x][pt.y].size();
    //Determine if the vector is empty
    if (dataStructure[pt.x][pt.y][0] == -1) {
        inside = false;
        return inside;
    }
    //Determine the position of the point
    int position = 0;

    for (int i = 0; i < length; i++) {

        if (dataStructure[pt.x][pt.y][i] == pt.z) {
            //boundary point
            inside = true;
            return inside;
        }

        if (dataStructure[pt.x][pt.y][i] > pt.z) {
            break;
        }

        position++;
    }  //  end for

    if (position == 0 || position == length) {

        // point lies outside the boundary
        inside = false;
        return inside;

    } else {

        //Determine if the elements on either side are even or odd
        int pre = position % 2;
        int post = (length - (position + 1)) % 2;

        if (pre == 0 && post == 0) {
            //Point lies outside
            inside = false;
            return inside;

        } else {
            //Point lies inside
            inside = true;

        }

    }
    return inside;
}


/*
 * Return the Number of Cells
 * 
 * @param x int
 * @param y int
 * @param z int
 *
 * @return int
 */
int ChengbangAlgorithm::getNumPixels(int x, int y, int z) {

    float num = 0.0;

    for (int i = 0; i < x; i++) {

        for (int j = 0; j < z; j++) {  // Changed from y to z  TMC

            set<float, less<float> > s;
            for (unsigned int l = 0; l < dataStructure[i][j].size(); l++)
                s.insert(dataStructure[i][j][l]);
            /* for(unsigned int m=0; m<dataStructure2[i][j].size(); m++)
                 s.insert(dataStructure2[i][j][m]);*/

            set < float, less < float > > ::iterator
            p;
            vector<float> dataStructure;
            for (p = s.begin(); p != s.end(); p++) {
                if (*p > -1) dataStructure.push_back(*p);
            }


            for (unsigned int k = 0; k < dataStructure.size(); k += 2) {


                //check to see if the vector is empty
                if (dataStructure[0] == -1) break;

                //calculate the number of pixels
                float y1 = dataStructure[k];
                float y2 = dataStructure[k + 1];

                // case 1: z1 >= z
                if (y1 >= y) break;


                // case 2: z2 >= z
                if (y2 >= y) {
                    num += y - y1;
                    break;
                }

                num += y2 - y1 + 1;


            }

        }

    }

    return (int) num;

}

/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

// // // #define CompuCellLibShared_EXPORTS	// if you dont define this DLL import/export macro  from CompuCellLib you will get error "definition of dllimport static data member not allowed"
									// // // //this is because you define static members in the Simulator class and witohut this macro they will be redefined here as import symbols which is not allowed

// // // #include "Simulator.h"
// // // using namespace CompuCell3D;

// // // #include <BasicUtils/BasicException.h>
// // // #include <BasicUtils/BasicSmartPointer.h>

// // // // #include <XMLCereal/XMLPullParser.h>

// // // // #include <XercesUtils/XercesStr.h>

// // // // #include <xercesc/util/PlatformUtils.hpp>
// // // // XERCES_CPP_NAMESPACE_USE;

// // // #include <iostream>
// // // #include <string>
// // // #include <fstream>
// // // using namespace std;

// // // #include <stdlib.h>

// // // //#include <config.h>
// // // #include <BasicUtils/BasicRandomNumberGenerator.h>

// // // #include <XMLUtils/XMLParserExpat.h>


// // // #if defined(_WIN32)
	// // // #include <windows.h>
// // // #endif


#include <iostream>
#include <string>
#include <Components/CellCM.h>
#include <Components/SimulationBox.h>
#include <Components/CellInventoryCM.h>
#include <Components/CellFactoryCM.h>
#include <time.h>
#include <BasicUtils/BasicRandomNumberGenerator.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <PublicUtilities/NumericalUtils.h>
#include <limits>
#include <fstream>

#if defined(_WIN32)
	#include <windows.h>
#endif


using namespace std;
using namespace  CenterModel;

// // // PluginManager<Plugin> Simulator::pluginManager;
// // // PluginManager<Steppable> Simulator::steppableManager;
// // // BasicPluginManager<PluginBase> Simulator::pluginBaseManager;


void Syntax(const string name) {
  cerr << "Syntax: " << name << " <config>" << endl;
  exit(1);
}




int main(int argc, char *argv[]) {
	cerr<<"Welcome to CC3D command line edition"<<endl;
    CellCM cell;
    cell.grow();
	cerr<<"cell.position="<<cell.position<<endl;

	Vector3 boxDim(21.2,45.7,80.1);
	//Vector3 gridSpacing(4,10,8);
	//Vector3 gridSpacing(2.01,2.01,2.01);
    //Vector3 gridSpacing(3.01,3.01,3.01);
    Vector3 gridSpacing(1.01,1.01,1.01);

	SimulationBox sb;
	//sb.setDim(21.2,45.7,80.1);
	//sb.setBoxSpatialProperties(21.2,45.7,80.1,1.5,5.5,7.1);
	sb.setBoxSpatialProperties(boxDim,gridSpacing);
	cerr<<sb.getDim()<<endl;

	cerr<<sb.getLatticeLookupDim()<<endl;

	CellFactoryCM cf=CellFactoryCM();
	cf.setSimulationBox(&sb);

    CellInventoryCM ci=CellInventoryCM();
	ci.setCellFactory(&cf);

	//CellCM *cell1=cf.createCellCM(19,20,67);
	//ci.addToInventory(cell1);
	//ci.addToInventory(cf.createCellCM(2,34,21));
	//ci.addToInventory(cf.createCellCM(8,1,9));
	//ci.addToInventory(cf.createCellCM(8,1,10));





	BasicRandomNumberGeneratorNonStatic rGen;
	srand(time(0));
	unsigned int randomSeed=(unsigned int)rand()*((std::numeric_limits<unsigned int>::max)()-1);                
	rGen.setSeed(randomSeed);

	//creating many cells
	int N=200;
	double r_min=1.0;
	double r_max=2.0;


	//ofstream out("init.cpp");
	//out<<"for (int i=0;i<N;++i){"<<endl;
	//
	//for (int i=0;i<N;++i){
	//	CellCM * cell=cf.createCellCM(boxDim.fX*rGen.getRatio(),boxDim.fY*rGen.getRatio(),boxDim.fZ*rGen.getRatio());
	//	cell->interactionRadius=r_min+rGen.getRatio()*(r_max-r_min);
	//	ci.addToInventory(cell);
	//	out<<"CellCM * cell=cf.createCellCM("<<cell->position.fX<<","<<cell->position.fY<<","<<cell->position.fZ<<");"<<endl;
	//	out<<"cell->interactionRadius="<<cell->interactionRadius<<";"<<endl;
	//	out<<"ci.addToInventory(cell);"<<endl;
	//}
	//out<<"}"<<endl;

CellCM * cellTmp;
{
    //cellTmp=cf.createCellCM(8.50226,2.74006,49.2544);
    //cellTmp->interactionRadius=1.63363;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(0.957869,34.1792,8.1048);
    //cellTmp->interactionRadius=1.53577;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(20.2145,3.10814,39.3706);
    //cellTmp->interactionRadius=1.69376;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(16.2355,3.90475,29.2961);
    //cellTmp->interactionRadius=1.9051;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(11.8092,10.5344,9.72038);
    //cellTmp->interactionRadius=1.92959;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(12.9936,27.5854,15.7383);
    //cellTmp->interactionRadius=1.92742;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(11.4043,34.531,76.8139);
    //cellTmp->interactionRadius=1.52231;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(14.2928,37.3731,75.133);
    //cellTmp->interactionRadius=1.41037;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(0.165255,28.6444,43.6944);
    //cellTmp->interactionRadius=1.84817;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(14.7564,27.1568,32.1483);
    //cellTmp->interactionRadius=1.56949;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(6.34917,21.3794,31.5174);
    //cellTmp->interactionRadius=1.4943;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(14.7091,0.924015,42.0738);
    //cellTmp->interactionRadius=1.50986;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(3.3689,39.4589,56.4554);
    //cellTmp->interactionRadius=1.99719;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(9.17536,4.00476,38.7437);
    //cellTmp->interactionRadius=1.20454;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(13.2677,35.3836,41.2082);
    //cellTmp->interactionRadius=1.25301;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(17.802,20.6081,27.7588);
    //cellTmp->interactionRadius=1.92203;
    //ci.addToInventory(cellTmp);
    //cellTmp=cf.createCellCM(8.35865,22.1962,54.0304);
    //cellTmp->interactionRadius=1.87145;
    //ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(4.47898,34.0741,44.8702);
    cellTmp->interactionRadius=1.61149;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.3383,31.6978,42.1087);
    cellTmp->interactionRadius=1.19929;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.486,24.005,9.60629);
    cellTmp->interactionRadius=1.75444;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(19.0054,34.9002,19.3155);
    cellTmp->interactionRadius=1.68498;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(12.0979,10.2024,37.7463);
    cellTmp->interactionRadius=1.62347;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(9.62257,23.8601,76.2756);
    cellTmp->interactionRadius=1.5299;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.59423,43.4357,2.72842);
    cellTmp->interactionRadius=1.78198;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(15.3408,27.6888,75.4638);
    cellTmp->interactionRadius=1.32597;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.323048,7.70284,72.1133);
    cellTmp->interactionRadius=1.95038;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.13511,18.062,49.8454);
    cellTmp->interactionRadius=1.79871;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.4384,9.57655,49.3729);
    cellTmp->interactionRadius=1.89099;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.9413,4.74063,65.8425);
    cellTmp->interactionRadius=1.81665;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(11.9527,40.6394,33.3921);
    cellTmp->interactionRadius=1.89255;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(3.13393,36.0296,43.5403);
    cellTmp->interactionRadius=1.2342;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(12.6406,1.95232,67.7097);
    cellTmp->interactionRadius=1.57533;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.08988,17.1579,5.71242);
    cellTmp->interactionRadius=1.32121;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.55567,13.0278,72.8016);
    cellTmp->interactionRadius=1.35015;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(18.7802,36.2285,75.4284);
    cellTmp->interactionRadius=1.65459;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(1.26484,31.1395,64.5901);
    cellTmp->interactionRadius=1.0597;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(8.10004,30.4134,78.3801);
    cellTmp->interactionRadius=1.79986;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(3.93753,14.4183,8.42156);
    cellTmp->interactionRadius=1.36679;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(4.82077,15.9116,42.8188);
    cellTmp->interactionRadius=1.57908;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.5923,33.3686,20.7995);
    cellTmp->interactionRadius=1.27214;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(19.3526,14.0741,36.3295);
    cellTmp->interactionRadius=1.9618;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.6241,43.529,32.2951);
    cellTmp->interactionRadius=1.7623;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(9.21288,6.10629,10.029);
    cellTmp->interactionRadius=1.61702;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(14.9846,11.3478,62.8086);
    cellTmp->interactionRadius=1.17678;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.14435,36.5441,20.3315);
    cellTmp->interactionRadius=1.26624;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(15.8046,39.5801,62.7224);
    cellTmp->interactionRadius=1.79918;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(8.65718,36.131,73.8437);
    cellTmp->interactionRadius=1.94709;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.9992,7.68014,70.6392);
    cellTmp->interactionRadius=1.17938;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(4.66449,34.3774,53.4699);
    cellTmp->interactionRadius=1.18935;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.48114,37.0788,71.8708);
    cellTmp->interactionRadius=1.80175;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.8127,26.6169,32.9752);
    cellTmp->interactionRadius=1.8389;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.1714,14.6078,42.649);
    cellTmp->interactionRadius=1.73539;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(3.61923,19.961,34.1452);
    cellTmp->interactionRadius=1.31258;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(15.9701,30.2429,67.9472);
    cellTmp->interactionRadius=1.27417;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(15.7256,7.34011,51.2982);
    cellTmp->interactionRadius=1.21384;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(8.67962,0.912839,4.42406);
    cellTmp->interactionRadius=1.32346;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.4682,38.943,57.8276);
    cellTmp->interactionRadius=1.94523;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(11.7763,17.6938,57.3407);
    cellTmp->interactionRadius=1.51844;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.02501,26.0502,29.9069);
    cellTmp->interactionRadius=1.47047;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.972514,3.82647,1.42634);
    cellTmp->interactionRadius=1.64772;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(4.35257,11.3218,12.0298);
    cellTmp->interactionRadius=1.66805;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.3232,22.7976,9.03719);
    cellTmp->interactionRadius=1.34408;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.74598,16.812,2.42697);
    cellTmp->interactionRadius=1.95204;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(5.22077,44.133,7.64853);
    cellTmp->interactionRadius=1.89323;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(15.7597,10.1922,16.7454);
    cellTmp->interactionRadius=1.06198;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(14.6177,24.5667,24.1773);
    cellTmp->interactionRadius=1.77856;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.27421,42.18,22.9565);
    cellTmp->interactionRadius=1.18023;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.6893,11.6285,43.9706);
    cellTmp->interactionRadius=1.29635;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.9874,17.0871,10.4383);
    cellTmp->interactionRadius=1.10319;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(1.63653,20.2773,74.0514);
    cellTmp->interactionRadius=1.82871;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(4.27912,27.4633,51.5114);
    cellTmp->interactionRadius=1.65389;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.5733,0.818999,6.83524);
    cellTmp->interactionRadius=1.59481;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.69971,7.88522,22.7026);
    cellTmp->interactionRadius=1.83758;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(3.0423,36.1679,42.8362);
    cellTmp->interactionRadius=1.85384;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.3753,37.9956,9.40588);
    cellTmp->interactionRadius=1.00963;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(5.65801,14.9099,46.0068);
    cellTmp->interactionRadius=1.20159;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(15.9041,34.9344,58.0557);
    cellTmp->interactionRadius=1.4416;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(1.36331,10.4352,70.5182);
    cellTmp->interactionRadius=1.61425;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(14.8142,26.0873,75.2616);
    cellTmp->interactionRadius=1.46426;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.4632,41.0428,0.222676);
    cellTmp->interactionRadius=1.43276;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.64559,12.6069,64.5245);
    cellTmp->interactionRadius=1.53932;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.624519,8.13264,54.4749);
    cellTmp->interactionRadius=1.92872;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.48367,3.84026,51.9934);
    cellTmp->interactionRadius=1.48289;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.2053,38.7926,45.1225);
    cellTmp->interactionRadius=1.02884;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(20.2481,28.837,52.9742);
    cellTmp->interactionRadius=1.62107;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(20.3466,42.5391,1.21281);
    cellTmp->interactionRadius=1.85067;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.111,3.99109,29.3962);
    cellTmp->interactionRadius=1.99547;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.4769,44.1255,8.91783);
    cellTmp->interactionRadius=1.1887;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.9205,33.6149,14.6027);
    cellTmp->interactionRadius=1.09148;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(9.26846,19.1041,61.1106);
    cellTmp->interactionRadius=1.69595;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.3139,5.44672,8.25657);
    cellTmp->interactionRadius=1.25931;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.8755,6.09969,17.0773);
    cellTmp->interactionRadius=1.08893;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(20.9313,27.9434,16.3279);
    cellTmp->interactionRadius=1.00731;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(12.8976,30.5966,63.0194);
    cellTmp->interactionRadius=1.61685;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(18.7131,28.5664,7.49375);
    cellTmp->interactionRadius=1.5886;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.4205,27.9421,33.2438);
    cellTmp->interactionRadius=1.95315;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(21.1233,10.6855,78.5744);
    cellTmp->interactionRadius=1.30402;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.93592,28.8334,50.9158);
    cellTmp->interactionRadius=1.57242;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.4354,15.7047,34.2158);
    cellTmp->interactionRadius=1.00783;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(5.13654,13.2638,11.5405);
    cellTmp->interactionRadius=1.75015;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.39085,38.9296,79.6016);
    cellTmp->interactionRadius=1.52273;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(14.78,9.73209,44.0966);
    cellTmp->interactionRadius=1.22915;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.01609,19.3223,60.4437);
    cellTmp->interactionRadius=1.45891;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(6.81826,36.8462,62.6687);
    cellTmp->interactionRadius=1.53065;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.2619,6.5285,55.4309);
    cellTmp->interactionRadius=1.20537;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.9487,16.6469,79.2507);
    cellTmp->interactionRadius=1.4537;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(18.9642,6.19118,49.4758);
    cellTmp->interactionRadius=1.26403;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(19.4971,18.0708,9.53119);
    cellTmp->interactionRadius=1.8644;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.4623,21.0167,69.7256);
    cellTmp->interactionRadius=1.95612;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.3886,31.3168,66.4064);
    cellTmp->interactionRadius=1.4503;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(14.3834,39.0062,43.398);
    cellTmp->interactionRadius=1.43028;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(15.3867,0.46021,51.3867);
    cellTmp->interactionRadius=1.97347;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.40179,22.771,58.1503);
    cellTmp->interactionRadius=1.88005;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.606,38.7018,13.7168);
    cellTmp->interactionRadius=1.12937;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.619517,29.8371,78.6077);
    cellTmp->interactionRadius=1.5944;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(12.6694,0.543862,30.7514);
    cellTmp->interactionRadius=1.3043;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.5454,11.0398,70.5688);
    cellTmp->interactionRadius=1.3521;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.0632,14.2376,13.195);
    cellTmp->interactionRadius=1.71856;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(1.95209,5.81703,33.3924);
    cellTmp->interactionRadius=1.10935;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.2124,45.6895,15.4609);
    cellTmp->interactionRadius=1.4883;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(4.97793,11.0432,12.3438);
    cellTmp->interactionRadius=1.52459;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(5.65597,41.4854,0.921998);
    cellTmp->interactionRadius=1.56618;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(3.72439,11.5134,46.3679);
    cellTmp->interactionRadius=1.47694;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.6831,14.6237,41.695);
    cellTmp->interactionRadius=1.82324;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.16987,25.1887,34.9751);
    cellTmp->interactionRadius=1.53218;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.42507,33.1789,65.4457);
    cellTmp->interactionRadius=1.23767;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.679,39.8343,27.5678);
    cellTmp->interactionRadius=1.64665;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(18.2868,42.1671,66.9121);
    cellTmp->interactionRadius=1.41519;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(8.87684,35.5221,5.92866);
    cellTmp->interactionRadius=1.90697;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.9681,12.6872,39.3582);
    cellTmp->interactionRadius=1.05777;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.294643,10.4775,64.4748);
    cellTmp->interactionRadius=1.63255;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.80831,3.60359,39.6091);
    cellTmp->interactionRadius=1.17921;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(9.83921,10.0738,62.7279);
    cellTmp->interactionRadius=1.54637;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.2652,32.0143,13.1696);
    cellTmp->interactionRadius=1.73891;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.24454,20.9087,13.4271);
    cellTmp->interactionRadius=1.10454;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(19.2349,45.2052,33.3546);
    cellTmp->interactionRadius=1.52111;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(9.81657,43.251,35.8052);
    cellTmp->interactionRadius=1.88799;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(9.64442,25.8926,66.1224);
    cellTmp->interactionRadius=1.25743;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(1.44301,5.38202,61.5213);
    cellTmp->interactionRadius=1.27068;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(12.9734,29.6916,44.9843);
    cellTmp->interactionRadius=1.60525;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(11.4645,8.5787,56.3918);
    cellTmp->interactionRadius=1.73081;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.7672,39.9149,40.5532);
    cellTmp->interactionRadius=1.55291;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(9.41115,21.5845,26.4556);
    cellTmp->interactionRadius=1.60338;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(12.7802,4.21794,36.2149);
    cellTmp->interactionRadius=1.5475;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(11.9671,40.9531,36.943);
    cellTmp->interactionRadius=1.22381;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(12.026,32.8405,41.4319);
    cellTmp->interactionRadius=1.17787;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.118517,12.6313,41.3649);
    cellTmp->interactionRadius=1.4336;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(4.89775,34.3805,15.5595);
    cellTmp->interactionRadius=1.61472;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(14.0052,28.954,10.1918);
    cellTmp->interactionRadius=1.81303;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(19.4673,37.2459,43.0177);
    cellTmp->interactionRadius=1.89951;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(6.8582,25.9081,38.8645);
    cellTmp->interactionRadius=1.60168;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(14.6315,33.6496,3.95099);
    cellTmp->interactionRadius=1.59019;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(5.61994,36.1794,48.1684);
    cellTmp->interactionRadius=1.128;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.93363,39.2857,36.482);
    cellTmp->interactionRadius=1.0893;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(1.4881,41.4595,4.84442);
    cellTmp->interactionRadius=1.69325;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.4544,31.6646,26.9425);
    cellTmp->interactionRadius=1.43078;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(5.93341,40.0792,79.0673);
    cellTmp->interactionRadius=1.85044;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(18.1755,39.6002,3.44194);
    cellTmp->interactionRadius=1.87102;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(20.3249,35.3176,23.6116);
    cellTmp->interactionRadius=1.38925;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.0371,32.3382,21.0786);
    cellTmp->interactionRadius=1.42712;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.70422,4.49596,17.2949);
    cellTmp->interactionRadius=1.501;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(20.7777,24.5933,33.3174);
    cellTmp->interactionRadius=1.69646;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(11.3029,35.8112,51.7814);
    cellTmp->interactionRadius=1.68661;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.894061,2.36456,61.8402);
    cellTmp->interactionRadius=1.61418;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.6308,25.1663,21.7984);
    cellTmp->interactionRadius=1.66008;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.6749,44.9023,37.9002);
    cellTmp->interactionRadius=1.17423;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.2094,7.61874,46.1179);
    cellTmp->interactionRadius=1.95445;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(2.31531,38.2089,35.54);
    cellTmp->interactionRadius=1.36402;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(3.39392,13.0967,20.8862);
    cellTmp->interactionRadius=1.64975;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(8.23838,26.6385,73.7594);
    cellTmp->interactionRadius=1.37293;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(4.39846,33.3241,48.4498);
    cellTmp->interactionRadius=1.38289;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(1.79076,39.9778,31.5085);
    cellTmp->interactionRadius=1.12054;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.2427,36.1948,15.7845);
    cellTmp->interactionRadius=1.3087;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(6.73963,44.2174,21.1403);
    cellTmp->interactionRadius=1.8519;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.22735,3.51103,67.094);
    cellTmp->interactionRadius=1.7662;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(6.52663,18.1513,49.4462);
    cellTmp->interactionRadius=1.27353;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.1577,36.7412,14.3177);
    cellTmp->interactionRadius=1.38914;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(11.2298,7.64949,52.7359);
    cellTmp->interactionRadius=1.1516;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.750908,38.8885,28.3254);
    cellTmp->interactionRadius=1.16446;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(18.1541,37.4281,57.5616);
    cellTmp->interactionRadius=1.72386;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(13.007,31.4309,26.3596);
    cellTmp->interactionRadius=1.6775;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(1.19053,32.5398,17.5239);
    cellTmp->interactionRadius=1.98722;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(3.61692,16.2749,8.63725);
    cellTmp->interactionRadius=1.5317;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(17.8763,10.1097,74.0543);
    cellTmp->interactionRadius=1.45327;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(7.36646,16.293,30.0829);
    cellTmp->interactionRadius=1.22891;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(5.60142,12.1322,12.5089);
    cellTmp->interactionRadius=1.4732;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(14.7374,42.8158,37.0804);
    cellTmp->interactionRadius=1.58643;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.8697,27.6886,59.9774);
    cellTmp->interactionRadius=1.85041;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(12.6265,45.0048,77.6167);
    cellTmp->interactionRadius=1.08561;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(9.12255,3.95954,37.8601);
    cellTmp->interactionRadius=1.72548;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(0.883361,5.60358,34.3147);
    cellTmp->interactionRadius=1.42947;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(18.8124,12.0754,0.898631);
    cellTmp->interactionRadius=1.88042;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(9.94912,4.5029,31.0809);
    cellTmp->interactionRadius=1.13219;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(19.676,11.4455,5.62409);
    cellTmp->interactionRadius=1.47868;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(1.73641,28.734,57.5735);
    cellTmp->interactionRadius=1.13317;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(11.8065,44.3068,19.1735);
    cellTmp->interactionRadius=1.38235;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(10.3887,2.19327,74.311);
    cellTmp->interactionRadius=1.49505;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(11.9969,26.4597,16.8209);
    cellTmp->interactionRadius=1.49515;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(16.1037,11.3767,77.5765);
    cellTmp->interactionRadius=1.02397;
    ci.addToInventory(cellTmp);
    cellTmp=cf.createCellCM(11.5559,0.62928,15.6405);
    cellTmp->interactionRadius=1.98425;
    ci.addToInventory(cellTmp);
}


	cerr<<"inventory size="<<ci.getSize()<<endl;

	const SimulationBox::LookupField_t & lookupField=sb.getLookupFieldRef();

	CompuCell3D::Dim3D lookupFieldDim=lookupField.getDim();
	CompuCell3D::Point3D pt;

	int cellCounter=0;

	
	//for (pt.x=0;pt.x<lookupFieldDim.x; ++pt.x)
	//	for (pt.y=0;pt.y<lookupFieldDim.y; ++pt.y)
	//		for (pt.z=0;pt.z<lookupFieldDim.z; ++pt.z){

	//			set<CellSorterDataCM> &lookupSet=lookupField.get(pt)->sorterSet;
	//			cellCounter+=lookupSet.size();
	//			cerr<<"THIS IS SET FOR pt="<<pt<<" size="<<lookupSet.size()<<endl;

	//		}
	//cerr<<"cellCounter="<<cellCounter<<endl;

	//visiting interaction pairs
	
	//CellInventoryCM::cellInventoryIterator itr;
	//for (itr=ci.cellInventoryBegin() ; itr!=ci.cellInventoryEnd(); ++itr){
	//	CellCM * cell=itr->second;
	//	//CompuCell3D::Point3D pt=sb.getCellLatticeLocation(cell);

	//	std::pair<std::vector<CompuCell3D::Point3D>,unsigned int> neighborListPair=sb.getLatticeLocationsWithinInteractingRange(cell);
	//	for (int i=0 ; i<neighborListPair.first.size();++i){
	//		cerr<<"neighborListPair.first["<<i<<"]="<<neighborListPair.first[i]<<endl;
	//	}

	//}
	
	//std::set<int> set1;
	//std::set<int> set2;

	//set1.insert(1);
	//set1.insert(2);

	//std::set<int>::iterator sitr1=set1.begin();
	//std::set<int>::iterator sitr2=set2.end();


	//pt=sb.getCellLatticeLocation(cell1);
	//std::pair<std::vector<CompuCell3D::Point3D>,unsigned int> nPair=sb.getLatticeLocationsWithinInteractingRange(cell1);

	//std::vector<CompuCell3D::Point3D> nSitesVec=nPair.first;

	//const SimulationBox::LookupField_t & lookupLatticeRef=sb.getLookupFieldRef();

	//cerr<<"CENTER POINT="<<pt<<endl;
	//for(unsigned int i=0 ; i<nSitesVec.size();++i){
	//	cerr<<"nSite["<<i<<"]="<<nSitesVec[i]<<endl;
	//	std::set<CellSorterDataCM> & currentSorterSet=lookupLatticeRef.get(nSitesVec[i])->sorterSet;
	//	for (std::set<CellSorterDataCM>::iterator sitr = currentSorterSet.begin() ; sitr!= currentSorterSet.end();++sitr){
	//		cerr<<"cell id="<<sitr->cell->id<<endl;

	//	}
	//}


#if defined(_WIN32)
	volatile DWORD dwStart;
	dwStart = GetTickCount();
#endif


	Vector3 bc; //boundary condition vector - initialized with (0.,0.,0.)
	
	int n=0;
	double dist=0.0;

	CellInventoryCM::cellInventoryIterator itr;
	for (itr=ci.cellInventoryBegin() ; itr!=ci.cellInventoryEnd(); ++itr){
		CellCM * cell=itr->second;
		
		InteractionRangeIterator itr = sb.getInteractionRangeIterator(cell);

		itr.begin();
		InteractionRangeIterator endItr = sb.getInteractionRangeIterator(cell).end();

		//bool flag= (itr==endItr);
		//cerr<<"itr==endItr = "<<flag<<endl;
		//

		//
		//cerr<<"itr->cell="<<(*itr)<<endl;
		//
		//++itr;
		//cerr<<"itr->cell="<<(*itr)<<endl;

		//cerr<<"itr!=endItr="<<(itr!=endItr)<<endl;
		////cerr<<"itr==endItr = "<<(itr==endItr)<<endl;

		//cerr<<"***************** NEIGHBORS of cell->id="<<cell->id<<endl;
		CellCM *nCell;

        Vector3 distVector;

		for (itr.begin(); itr!=endItr ;++itr){
			nCell=(*itr);
			if (nCell==cell)//neglecting "self interactions"
				continue; 

			distVector=distanceVectorInvariantCenterModel(cell->position,nCell->position,boxDim ,bc);
            dist=distVector.Mag();
			if (dist<=cell->interactionRadius || dist<=nCell->interactionRadius){
				cerr<<"**********INTERACTION "<<cell->id<<"-"<<nCell->id<<" ********************"<<endl;
				cerr<<"THIS IS cell.id="<<nCell->id<<" distance from center cell="<<dist<<endl;
				cerr<<"nCell->position="<<nCell->position<<" cell->position="<<cell->position<<endl;
                cerr<<"nCellLocation="<<sb.getCellLatticeLocation(nCell)<<" cell location="<<sb.getCellLatticeLocation(cell)<<endl;
				++n;
			}

			
		}
		
		//if (n++>20){
		//	break;
		//}
	


	}

    CellCM *cell104 = ci.getCellById(104);
    CellCM *cell169 = ci.getCellById(169);

    cerr<<"location 104="<<sb.getCellLatticeLocation(cell104);
    cerr<<"location 169="<<sb.getCellLatticeLocation(cell169);

	cerr<<"FOUND "<<n<<" interactions of cells"<<endl;
#if defined(_WIN32)
	cerr<<"DISTANCE CALCULATION FOR  "<<N<<" cells too "<<GetTickCount()-dwStart<<" miliseconds to complete"<<endl;
	dwStart = GetTickCount();
#endif	
	
	
	
	
  return 1;
}

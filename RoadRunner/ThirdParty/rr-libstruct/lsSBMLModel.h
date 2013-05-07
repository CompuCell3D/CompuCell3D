#ifndef ls_SBML_MODEL_H
#define ls_SBML_MODEL_H
#ifndef NO_SBML

#include <string>
#include <sbml/SBMLDocument.h>
#include <sbml/Model.h>
#include <sbml/Species.h>

namespace ls
{

using namespace libsbml;

class SBMLmodel
{
	private:
    	SBMLDocument*			    _Document;
    	Model* 						_Model;

	public:
        static SBMLmodel* 			FromFile(std::string &sFileName);
        static SBMLmodel* 			FromSBML(std::string &sSBML);
                                    SBMLmodel(std::string &sSBML);
                                    SBMLmodel();
                                   ~SBMLmodel(void);

        void 						InitializeFromSBML(std::string &sSBML);
        void 						InitializeFromFile(std::string &sFileName);

        Model* 						getModel();
        int 						numFloatingSpecies();
        int 						numReactions();
        const Species* 				getNthFloatingSpecies(int n);
        const Species* 				getNthBoundarySpecies(int n);
        const Reaction* 			getNthReaction(int n);
};

}
#endif

#endif //ls_SBML_MODEL_H

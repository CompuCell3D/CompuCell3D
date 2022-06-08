#ifndef REACTION_KERNELS_H
#define REACTION_KERNELS_H

#include <string>
#include <vector>

typedef std::pair <std::string, std::string> fieldNameAddTerm_t;

std::string genReactionKernels(std::vector < fieldNameAddTerm_t >
const &fieldNameAddTerms);

#endif
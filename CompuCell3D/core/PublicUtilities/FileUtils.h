#ifndef FILEUTILS_H
#define FILEUTILS_H

#if defined(_WIN32) || defined(__MINGW32__)
#include "WindowsGlob.h"
#else
#include <glob.h>
#endif

#include <string>
#include <vector>

std::vector<std::string> filesInDir(const std::string& dirpath, const std::string& glob_expr);

#endif //FILEUTILS_H
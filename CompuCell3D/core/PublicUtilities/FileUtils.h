#ifndef FILEUTILS_H
#define FILEUTILS_H

#include <experimental/filesystem>
#include <regex>
#include <string>
#include <vector>

template<bool recursive>
std::vector<std::string> filesInDir(const std::experimental::filesystem::path& dirpath, const std::regex& regex_expr);
template<bool recursive>
std::vector<std::string> filesInDir(const std::string& dirpath, const std::string& regex_expr);

#endif //FILEUTILS_H
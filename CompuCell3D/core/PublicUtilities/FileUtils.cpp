#include "FileUtils.h"

namespace filesystem = std::experimental::filesystem;

template<bool recursive>
std::vector<std::string> filesInDir(const filesystem::path& dirpath, const std::regex& regex_expr) {
    std::vector<std::string> o;
    typedef typename std::conditional<recursive, filesystem::recursive_directory_iterator, filesystem::directory_iterator>::type itr_t;
    for (auto& itr: itr_t(dirpath))
        if (filesystem::is_regular_file(itr) && std::regex_match(itr.path().extension().string(), regex_expr))
            o.push_back(itr.path().string());
    return o;
}

template<bool recursive>
std::vector<std::string> filesInDir(const std::string& dirpath, const std::string& regex_expr) {
    return filesInDir<recursive>(filesystem::path(dirpath), std::regex(regex_expr));
}

template std::vector<std::string> filesInDir<true>(const filesystem::path&, const std::regex&);
template std::vector<std::string> filesInDir<false>(const filesystem::path&, const std::regex&);
template std::vector<std::string> filesInDir<true>(const std::string&, const std::string&);
template std::vector<std::string> filesInDir<false>(const std::string&, const std::string&);

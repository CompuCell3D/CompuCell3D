#include "FileUtils.h"
#include <iostream>

std::vector<std::string> filesInDir(const std::string& dirpath, const std::string& glob_expr) {
    std::vector<std::string> result;

#if defined(_WIN32) || defined(__MINGW32__)

    result = get_all_files_names_within_folder(dirpath, glob_expr);

#else
    std::string pathGlob = dirpath + "/" + glob_expr;
    glob_t globbuf;

    if (glob(pathGlob.c_str(), 0, NULL, &globbuf) == 0) {
        for (unsigned int i = 0; i < globbuf.gl_pathc; i++)
            result.push_back(globbuf.gl_pathv[i]);
    }

    globfree(&globbuf);

#endif

    return result;

}
//#include "FileUtils.h"
//
//#include <glob.h>
//#include <vector>
//
//
//std::vector<std::string> filesInDir(const std::string& dirpath, const std::string& pattern){
//    glob_t glob_result;
//    std::string full_glob_str = dirpath+"/"+pattern;
//    glob(full_glob_str.c_str(),GLOB_TILDE,NULL,&glob_result);
//    std::vector<std::string> files;
//    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
//        files.emplace_back(glob_result.gl_pathv[i]);
//    }
//    globfree(&glob_result);
//    return files;
//}
//
////namespace filesystem = std::filesystem;
////
////template<bool recursive>
////std::vector<std::string> filesInDir(const filesystem::path& dirpath, const std::regex& regex_expr) {
////    std::vector<std::string> o;
////    typedef typename std::conditional<recursive, filesystem::recursive_directory_iterator, filesystem::directory_iterator>::type itr_t;
////    for (auto& itr: itr_t(dirpath))
////        if (filesystem::is_regular_file(itr) && std::regex_match(itr.path().extension().string(), regex_expr))
////            o.push_back(itr.path().string());
////    return o;
////}
////
////template<bool recursive>
////std::vector<std::string> filesInDir(const std::string& dirpath, const std::string& regex_expr) {
////    return filesInDir<recursive>(filesystem::path(dirpath), std::regex(regex_expr));
////}
////
////template std::vector<std::string> filesInDir<true>(const filesystem::path&, const std::regex&);
////template std::vector<std::string> filesInDir<false>(const filesystem::path&, const std::regex&);
////template std::vector<std::string> filesInDir<true>(const std::string&, const std::string&);
////template std::vector<std::string> filesInDir<false>(const std::string&, const std::string&);

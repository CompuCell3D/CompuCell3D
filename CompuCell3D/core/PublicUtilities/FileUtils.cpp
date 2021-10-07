#include "FileUtils.h"

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

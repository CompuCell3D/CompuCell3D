//
// Created by m on 10/9/21.
//

#ifndef WINDOWSGLOB_H
#define WINDOWSGLOB_H

#include <io.h>
#include <windows.h>
#include <vector>
#include <string>
typedef struct
{
    size_t gl_pathc;
    char **gl_pathv;
    size_t gl_offs;
} glob_t;

#define GLOB_ERR  (1 << 0)
#define GLOB_MARK       (1 << 1)	/* Append a slash to each name.  */
#define GLOB_NOSORT     (1 << 2)	/* Don't sort the names.  */
#define GLOB_DOOFFS     (1 << 3)	/* Insert PGLOB->gl_offs NULLs.  */
#define GLOB_NOCHECK    (1 << 4)	/* If nothing matches, return the pattern.  */
#define GLOB_APPEND     (1 << 5)	/* Append to results of a previous call.  */
#define GLOB_NOESCAPE   (1 << 6)	/* Backslashes don't quote metacharacters.  */
#define GLOB_PERIOD     (1 << 7)	/* Leading `.' can be matched by metachars.  */
#define GLOB_MAGCHAR    (1 << 8)	/* Set in gl_flags if any metachars seen.  */
#define GLOB_ALTDIRFUNC (1 << 9)	/* Use gl_opendir et al functions.  */
#define GLOB_BRACE      (1 << 10)	/* Expand "{a,b}" to "a" "b".  */
#define GLOB_NOMAGIC    (1 << 11)	/* If no magic chars, return the pattern.  */
#define GLOB_TILDE      (1 << 12)	/* Expand ~user and ~ to home directories. */
#define GLOB_ONLYDIR    (1 << 13)	/* Match only directories.  */
#define GLOB_TILDE_CHECK (1 << 14)	/* Like GLOB_TILDE but return an error
									  if the user name is not available.  */
/* Error returns from `glob'.  */
#define GLOB_NOSPACE    1       /* Ran out of memory.  */
#define GLOB_ABORTED    2       /* Read error.  */
#define GLOB_NOMATCH    3       /* No matches found.  */
#define GLOB_NOSYS      4       /* Not implemented.  */
int glob(const char *pattern, int flags, void* unused, glob_t*);
void globfree(glob_t*);
std::vector<std::string> get_all_files_names_within_folder(std::string folder, std::string pattern);
char* getfname(const char*);
static int glob_compare(const void*, const void*);


#endif //COMPUCELL3D_WINDOWSGLOB_H

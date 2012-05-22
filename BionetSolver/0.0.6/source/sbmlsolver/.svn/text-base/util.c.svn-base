/**
 * Filename    : util.c
 * Revision    : $Id: util.c,v 1.4 2008/09/12 20:02:49 raimc Exp $
 * Source      : $Source: /cvsroot/sbmlsolver/SBML_odeSolver/src/util.c,v $
 */
/* 
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. The software and
 * documentation provided hereunder is on an "as is" basis, and the
 * authors have no obligations to provide maintenance, support,
 * updates, enhancements or modifications.  In no event shall the
 * authors be liable to any party for direct, indirect, special,
 * incidental or consequential damages, including lost profits, arising
 * out of the use of this software and its documentation, even if the
 * authors have been advised of the possibility of such damage.  See
 * the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 *
 * The original code contained here was initially developed by:
 *
 *     Christoph Flamm
 *
 * Contributor(s):
 *     
 */

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>

/* own header files */
#include "sbmlsolver/util.h"

/*-------------------------------------------------------------------------*/
void nrerror(const char message[])
{
  fprintf(stderr, "\n%s\n", message);
  exit(EXIT_FAILURE);
}

/*-------------------------------------------------------------------------*/
void *space(unsigned size) {
  void *pointer;
  
  if ( (pointer = (void *) calloc(1, (size_t) size)) == NULL) {
#ifdef EINVAL
    if (errno==EINVAL) {
      fprintf(stderr,"SPACE: requested size: %d\n", size);
      nrerror("SPACE allocation failure -> EINVAL");
    }
    if (errno==ENOMEM)
#endif
      nrerror("SPACE allocation failure -> no memory");
  }
  return  pointer;
}

#ifdef WITH_DMALLOC
#include "dmalloc.h"
#define space(S) calloc(1,(S))
#endif

#undef xrealloc
/* dmalloc.h #define's xrealloc */
void *xrealloc (void *p, unsigned size) {
  if (p == 0)
    return space(size);
  p = (void *) realloc(p, size);
  if (p == NULL) {
#ifdef EINVAL
    if (errno==EINVAL) {
      fprintf(stderr,"xrealloc: requested size: %d\n", size);
      nrerror("xrealloc allocation failure -> EINVAL");
    }
    if (errno==ENOMEM)
#endif
      nrerror("xrealloc allocation failure -> no memory");  
  }
  return p;
}

/*-------------------------------------------------------------------------*/
char *get_line(FILE *fp)
{
  char s[512], *line, *cp;
  
  line = NULL;
  do {
    if (fgets(s, 512, fp)==NULL) break;
    cp = strchr(s, '\n');
    if (cp != NULL) *cp = '\0';
    if (line==NULL)
      line = space(strlen(s)+1); /*!!! TODO: valgrind: "Use of
                                       uninitialised value of size
                                       8" in adjsenstest_ContDiscData*/
    else
      line = (char *) xrealloc(line, strlen(s)+strlen(line)+1);
    strcat(line, s);
  } while(cp==NULL);
  
  return line;
}

/*-------------------------------------------------------------------------*/

char*
concat (char *a, char *b)
{
  char *tmp = NULL;
  tmp = (char *)xalloc(strlen(a)+strlen(b)+2, sizeof(char));
  strcpy(tmp, a);
  if (tmp[strlen(a)-1] != '/') strcat(tmp, "/");
  strcat(tmp, b);
  return (tmp);
}

void fatal (FILE *hdl, char *fmt, ...)
{
  va_list args;
  if ( hdl == NULL ) hdl = stderr;
  va_start(args, fmt);
  if ( errno != 0 )
    fprintf( hdl, "FATAL ERROR: %s: ", strerror(errno));
  else fprintf(hdl, "FATAL ERROR: ");
  vfprintf(hdl, fmt, args);
  fprintf(hdl,"\n");
  fflush(hdl);
  va_end(args);
  exit(EXIT_FAILURE);
}
                                                                               
void Warn (FILE *hdl, char *fmt, ...)
{
  va_list args;
  if ( hdl == NULL ) hdl = stderr;
  va_start(args, fmt);
  fprintf(hdl, "WARNING: ");
  vfprintf(hdl, fmt, args);
  fprintf(hdl,"\n");
  fflush(hdl);
  va_end(args);
}

void *xalloc (size_t nmemb, size_t size)
{
  void *tmp;
  if ( (tmp = calloc(nmemb, size)) == NULL )
    fatal(stderr, "xalloc(): %s\n", strerror(ENOMEM));
  return (tmp);
}

/*
 * @return void
 */
void xfree (void *ptr)
{
  if ( ptr == NULL ) {
#ifdef MEMDEBUG
    Warn(stderr, "xfree(): arg 1 is NULL");
#endif
    return;
  }
  free (ptr);
  ptr = NULL;
}

/* End of file */

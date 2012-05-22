/*
  Last changed Time-stamp: <2006-06-12 11:04:54 raim>
  $Id: charBuffer.h,v 1.4 2009/03/27 15:55:03 fbergmann Exp $
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
 *     Andrew Finney
 *
 * Contributor(s):
 */

#ifndef _CHARBUFFER_
#define _CHARBUFFER_

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct charBuffer charBuffer_t;

  charBuffer_t *CharBuffer_create();
  void CharBuffer_free(charBuffer_t *);
  void CharBuffer_append(charBuffer_t *, const char *);
  void CharBuffer_appendInt(charBuffer_t *, int);
  void CharBuffer_appendDouble(charBuffer_t *, double);
  const char *CharBuffer_getBuffer(charBuffer_t *);

#ifdef __cplusplus
};
#endif

#endif

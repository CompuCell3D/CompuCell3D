/*
  Last changed Time-stamp: <2007-09-07 13:05:00 raim>
  $Id: interpol.h,v 1.6 2008/01/28 19:25:27 stefan_tbi Exp $
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
 *     Stefan Müller
 *
 * Contributor(s):
 *     
 */

#ifndef _INTERPOL_H_
#define _INTERPOL_H_

typedef struct ts time_series_t;

  /** Stores Interpolation Data */
  struct ts {
    int    n_var;   /**< number of variables in the list */
    char   **var;   /**< list of variables */

    int    n_data;  /**< number of variables for which data is stored */
    double **data;  /**< time series data for variables */
    int    type;    /**< interpolation type */
    double **data2; /**< interpolation data for variables */
    
    int    n_time;  /**< number of time points */
    double *time;   /**< time points */
    
    int    last;    /**< last interpolation interval */

    char   **mess;  /**< list of warning messages */
    int    *warn;   /**< number of warnings */
  } ;


  int read_header_line(char *file, int n_var, char **var,
			    int *col, int *index);
  int read_columns(char *file, int n_col, int *col, int *index,
			time_series_t *ts);

  void free_data(time_series_t *ts);
  void print_data(time_series_t *ts);
  void test_interpol(time_series_t *ts);

  time_series_t *read_data(char *file, int num, char **var);

  double call(int i, double x, time_series_t *ts);

  int spline(int n, double *x, double *y, double *y2);
  void splint(int n, double *x, double *y, double *y2,
	      double x_, double *y_, int *j);

  void linint(int n, double *x, double *y,
	      double x_, double *y_, int *j);
	    
  int  bisection(int n, double *x, double x_);
  void hunt(int n, double *x, double x_, int *low);
  
#endif

/* end of file */

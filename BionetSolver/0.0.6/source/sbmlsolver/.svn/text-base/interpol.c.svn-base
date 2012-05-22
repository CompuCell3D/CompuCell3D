/*
  Last changed Time-stamp: <2007-09-07 13:18:03 raim>
  $Id: interpol.c,v 1.10 2007/12/18 14:05:34 stefan_tbi Exp $
*/

/* written 2005 by stefan mueller */
/* revised 2007 by sm */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "sbmlsolver/solverError.h"

#define PRIVATE static
#define PUBLIC SBML_ODESOLVER_API

#undef VERBOSE
/* #define VERBOSE */

#include "interpol.h"

/* ------------------------------------------------------------------------ */

void free_data(time_series_t* ts)
{
    int i;

    /* free data */
    for ( i=0; i<ts->n_var; i++ )
	{
	    free(ts->var[i]);
	    if (ts->data[i] != NULL) free(ts->data[i]);
	    if (ts->data2[i] != NULL) free(ts->data2[i]);
	}
    free(ts->time);

    /* free index lists */
    free(ts->var);
    free(ts->data);
    free(ts->data2);
    
    /* free warnings */
    for ( i=0; i<2; i++ )
	if ( ts->warn[i] != 0 ) /* ??? use SolverError here ??? */
	    Warn(stderr, "call(): %s: %d times\n", ts->mess[i], ts->warn[i]);
  
    free(ts->mess);
    free(ts->warn); 

    /* free */
    free(ts);

}

/* ------------------------------------------------------------------------ */

void print_data(time_series_t* ts)
{
    int i, j;

    /* print */
    fprintf(stderr, "\n");
    fprintf(stderr, "n_var : %i\n", ts->n_var);
    fprintf(stderr, "n_data: %i\n", ts->n_data);
    fprintf(stderr, "n_time: %i\n", ts->n_time);
    fprintf(stderr, "\n");

    /* print variable list */
    fprintf(stderr, "#t ");
    for (j=0; j<ts->n_var; j++) fprintf(stderr, "%s ", ts->var[j]);
    fprintf(stderr, "\n");
  
    /* print corresponding data */
    for ( i=0; i<ts->n_time; i++ )
	{
	    fprintf(stderr, "%g ", ts->time[i]);
	    for ( j=0; j<ts->n_var; j++ )
		{
		    if (ts->data[j] != NULL)
			fprintf(stderr, "%g ", ts->data[j][i]);
		    else fprintf(stderr, "x ");
		}
	    fprintf(stderr, "\n");
	}    	    
    fprintf(stderr, "\n");
    
    /* print */
    fprintf(stderr, "last  : %i\n", ts->last);
    fprintf(stderr, "\n");
    for ( i=0; i<2; i++ )
	fprintf(stderr, "%s: %d times\n", ts->mess[i], ts->warn[i]);  
    fprintf(stderr, "\n");

}
    
/* ------------------------------------------------------------------------ */

void test_interpol(time_series_t* ts)
{
    double *xs, *ys;
    double x, y_spl, y_lin;
    double prec;
    int i, i_, j;
    int nt, last_spl, last_lin;

    prec = 10;
    last_spl = 0;
    last_lin = 0;
    
    xs = ts->time;
    nt = ts->n_time;
    for ( i=0; i<nt-1; i++ )
	{
	    for ( i_=0; i_<prec; i_++ )
		{
		    x = xs[i] + (xs[i+1] - xs[i]) * i_ / prec;
		    printf("%g ", x);
		    for ( j=0; j<ts->n_var; j++ )
			{
			    ys = ts->data[j];
			    if ( ys != NULL )
				{
				    linint(nt, xs, ys, x, &y_lin, &last_lin);
				    splint(nt, xs, ys, ts->data2[j], x, &y_spl, &last_spl);
				    printf("%g %g ", y_lin, y_spl);
				}
			}
		    printf("\n");
		}
	}

}

/* ------------------------------------------------------------------------ */

/** given a file name and a variable list, read_data reads time series
    data from the file, (but only for variables in the list) stores
    the data according to the index in the variable list, calculates
    the second derivatives for spline interpolation, and returns a
    pointer to the created data structure. */

time_series_t *read_data(char *file, int n_var, char **var)
{
    int i;
    char *name;

    int n_data;      /* number of relevant data columns */
    int *col;        /* positions of relevant columns in data file */
    int *index;      /* corresponding indices in variable list */

    int n_time;      /* number of data rows */

    time_series_t *ts;

    /* alloc mem */
    ASSIGN_NEW_MEMORY_BLOCK(ts, 1, time_series_t, NULL);

    /* alloc mem for index lists */
    ts->n_var = n_var;
    ASSIGN_NEW_MEMORY_BLOCK(ts->var,   n_var, char *,   NULL);
    ASSIGN_NEW_MEMORY_BLOCK(ts->data,  n_var, double *, NULL); 
    ASSIGN_NEW_MEMORY_BLOCK(ts->data2, n_var, double *, NULL);    

    /* initialize index lists */
    for ( i=0; i<n_var; i++ )
	{
	    ASSIGN_NEW_MEMORY_BLOCK(name, strlen(var[i])+1, char , NULL);
	    strcpy(name, var[i]);
	    ts->var[i]   = name;
	    ts->data[i]  = NULL;
	    ts->data2[i] = NULL;
	}

    /* alloc temp mem for column info */
    ASSIGN_NEW_MEMORY_BLOCK(col,   n_var, int, NULL);
    ASSIGN_NEW_MEMORY_BLOCK(index, n_var, int, NULL);

    /* read header line */
    n_data = read_header_line(file, n_var, var, col, index);
    ts->n_data = n_data;
	
    /* count number of lines */
    n_time = read_columns(file, 0, NULL, NULL, NULL);
    ts->n_time = n_time;

    /* alloc mem for data */
    for ( i=0; i<n_data; i++ )
	{
	    ASSIGN_NEW_MEMORY_BLOCK(ts->data[index[i]],  n_time, double, NULL);
	    ASSIGN_NEW_MEMORY_BLOCK(ts->data2[index[i]], n_time, double, NULL);
	}
    ASSIGN_NEW_MEMORY_BLOCK(ts->time,  n_time, double, NULL);

    /* read data */
    read_columns(file, n_data, col, index, ts);

    /* free temp mem */
    free(col);
    free(index);

    /* initialize interpolation type */
    ts->type = 3;
    /* calculate second derivatives */
    for ( i=0; i<n_var; i++ )
	if ( ts->data[i] != NULL )
	    {
		if ( spline(ts->n_time, ts->time, ts->data[i], ts->data2[i]) != 1 )
		    return NULL; /* ran out of memory during spline routine */
	    }

    ts->last = 0;
    
    /* alloc mem for warnings */
    ASSIGN_NEW_MEMORY_BLOCK(ts->mess, 2, char *, NULL);
    ASSIGN_NEW_MEMORY_BLOCK(ts->warn, 2, int,    NULL);   

    /* initialize warnings */
    ts->mess[0] = "argument out of range (left) ";
    ts->mess[1] = "argument out of range (right)";
    for ( i=0; i<2; i++ )
	ts->warn[i] = 0;  

    return ts;
    
}

/* ------------------------------------------------------------------------ */

/* given a file name and a variable list, */
/* read_header_line finds the header line in the file, */
/* finds the columns (in the header line) of the variables (in the list), */
/* and returns the list of found columns (their indices) */
/* and the corresponding variables (their indices). */

int read_header_line(char *file, int n_var, char **var,
		     int *col, int *index)
{    
    FILE *fp;
    char *line, *token;
    
    int i, j;
    int count; /* counter for relevant columns */
    int *flag; /* flag for found variables */
    
    /* open file */
    if ( (fp = fopen(file, "r")) == NULL )
	fatal(stderr, "read_data(): read_header_line(): file not found");
    
    /* find header line */
    for ( i=0; (line = get_line(fp)) != NULL; i++ ) {
	
	/* read column 0 */
	token = strtok(line, " ");
	
	/* header line found */	
	if (token != NULL && strcmp(token, "#t") == 0)
	    break;
	
	/* skip empty lines and comment lines */
	if ( token == NULL || *token == '#' ) {
	    free(line);
	    continue;
	}	
	/* exit otherwise */
	else
	    fatal(stderr, "read_data(): read_header_line(): no header line found");
    }
    
    /* reset counter and flag */
    count = 0;
    flag = (int *) space(n_var * sizeof(int));
    for ( i=0; i<n_var; i++)
	flag[i] = 0;
    
    /* read other columns */
    for ( i=1; (token = strtok(NULL, " ")) != NULL; i++ ) {
	
	/* find column name in variable list */
	for ( j=0; j<n_var; j++ )
	    if ( strcmp(token, var[j]) == 0 )
		break;
	
	/* column name found */
	if ( j != n_var ) {
	    col[count]   = i;
	    index[count] = j;
	    count++;
	    flag[j] = 1;
	}
    }

    for ( i=0; i<n_var; i++)
	if ( flag[i] == 0 )
	    Warn(stderr, "read_data(): read_header_line(): no column for variable %s found", var[i]);   
	
    /* free */
    free(line);
    free(flag);
    fclose(fp);
	
    return count;
	
}

/* ------------------------------------------------------------------------ */

/* given a file name, */
/* a column list, a corresponding index list, */
/* and a pointer to the resulting data structure */
/* read_values reads the columns from the file into the data structure */
/* (at the right index) */
/* and returns the number of read lines. */
/* (or only counts the number of lines, if the pointer is NULL.) */

int read_columns(char *file, int n_col, int *col, int *index,
		 time_series_t *ts)
{    
    FILE *fp;
    char *line, *token;
    int i, j, k;
    int curr; /* current column */
    
    /* open file */
    if ( (fp = fopen(file, "r")) == NULL )
	fatal(stderr, "read_columns(): file not found");

    /* find data lines */
    for ( i=0; (line = get_line(fp)) != NULL; i++ ) {
	
	/* column 0 */
	token = strtok(line, " ");
	
	/* skip empty lines and comment lines (including header line) */
	if ( token == NULL || *token == '#' ) {
	    free(line);
	    i--;
	    continue;
	}
	
	/* do not read line - only count line */
	if ( ts == NULL ) {
	    free(line);
	    continue;
	}
	
	/* read value from column 0 into ts->time[i] */
	sscanf(token, "%lf", ts->time+i);
	
	/* read other columns */
	curr = 1;
	for ( j=0; j<n_col; j++ ) {
	    for ( k=curr; k<=col[j]; k++ )
		token = strtok(NULL, " ");
	    
	    /* read value from column col[j] into ts->data[index[j]][i] */
	    sscanf(token, "%lf", ts->data[index[j]]+i);
	    curr = k;
	}
	free(line);
    }
    
    /* free */
    fclose(fp);

    return i;

}
 
/* ------------------------------------------------------------------------ */

/* given a variable index i, a time point x, */
/* and time series data (for several variables), */
/* call returns the cubic-spline interpolation i(x) */
/* of the variable at the time point */

double call(int i, double x, time_series_t *ts)
{
    int nt;          /* number of x and y values */
    double *xs, *ys; /* x values, y values */
    double y;        /* result */

    /* introduce abbreviations */
    nt = ts->n_time;
    xs = ts->time;
    ys = ts->data[i];
    
    /* check if data is available */
    if ( i < 0 || i >= ts->n_var )
	fatal(stderr, "call(): variable index out of range");
    if ( ys == NULL )
	fatal(stderr, "call(): no data stored for variable");

    /* check if x is out of range (and warn) */
    if ( x < xs[0] )
	{
	    /*  fprintf(stderr, "left out range: %g\n", x); */
	    y = ys[0];
	    ts->last = -1;
	    ts->warn[0]++;
	}
    else if ( x >= xs[nt-1] )
	{
	    /* fprintf(stderr, "right out range: %g\n", x); */
	    y = ys[nt-1];
	    ts->last = nt-1;
	    ts->warn[1]++;
	}
    /* interpolate (and store last interpolation interval) */
    else
	splint(nt, xs, ys, ts->data2[i], x, &y, &ts->last);

#ifdef VERBOSE
    fprintf(stderr, "call %s(%g)\n", ts->var[i], x);
    fprintf(stderr, "function %s has index %d\n", ts->var[i], i);
    l = ts->last;
    if ( l == -1 )
	/*  fprintf(stderr, "argument out of range (left) \n"); */
	else if ( l == nt-1 )
	    /*  fprintf(stderr, "argument out of range (right)\n"); */
	    else
		{
		    fprintf(stderr, "argument %g found in interval ", x);
		    fprintf(stderr, "[ t[%d] = %g, t[%d] = %g ]\n", l, xs[l], l+1, xs[l+1]);
		}
    fprintf(stderr, "%s(%g) = %g\n", ts->var[i], x, y);
    fprintf(stderr, "\n");
#endif

    return y;

}

/* ------------------------------------------------------------------------ */

/* given arrays x[0..n-1] and y[0..n-1] */
/* tabulating a function f, i.e. y[i] = f(x[i]) */
/* spline returns y2[0..n-1] */
/* containing the second derivatives of the cubic-spline interpolation */

int spline(int n, double *x, double *y, double *y2)
{    
    int i;
    double p, sig, *u;

    ASSIGN_NEW_MEMORY_BLOCK(u, n-1, double, 0);

    y2[0] = u[0] = 0.0;
    for ( i=1; i<=n-2; i++ )
	{
	    sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
	    p = sig * y2[i-1] + 2.0;
	    y2[i] = (sig - 1.0) / p;
	    u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1]);
	    u[i] = (6.0 * u[i] / (x[i+1]-x[i-1]) - sig * u[i-1]) / p;
	}

    y2[n-1] = 0.0;
    for (i=n-2; i>=0; i--)
	y2[i] = y2[i] * y2[i+1] + u[i];

    free(u);
    return 1;
}

/* ------------------------------------------------------------------------ */

/* given arrays x[0..n-1] and y[0..n-1] */
/* tabulating a function f, i.e. y[i] = f(x[i]) */
/* and y2[0..n-1] */
/* containing the second derivatives of the cubic-spline interpolation */
/* and an argument x_ */
/* spline returns the interpolated value y_ = f(x_) */
/* and the left interval boundary j, i.e. x[j] <= x_ < x[j+1] */

void splint(int n, double *x, double *y, double *y2,
	    double x_, double *y_, int *j)
{

    double h, b, a;

    hunt(n, x, x_, j);
    
    h = x[*j+1] - x[*j];
    a = (x[*j+1] - x_) / h;
    b = (x_ - x[*j]) / h;
    *y_ = a * y[*j] + b * y[*j+1]
	+ ((a*a*a-a) * y2[*j] + (b*b*b-b) * y2[*j+1]) * (h*h) / 6.0;

}

/* ------------------------------------------------------------------------ */

void linint(int n, double *x, double *y, double x_, double *y_, int *j)
{
    double h, b, a;

    hunt(n, x, x_, j);
    
    h = x[*j+1] - x[*j];
    a = (x[*j+1] - x_) / h;
    b = (x_ - x[*j]) / h;
    *y_ = a * y[*j] + b * y[*j+1];

}
    
/* ------------------------------------------------------------------------ */

/* given an array x[0..n-1] and a value x_ */
/* bisection returns an index low */
/* such that x_ is in the interval [x[low], x[low+1]). */
/* low = -1 or low = n-1 indicates that x_ is out of range. */

int bisection(int n, double *x, double x_)
{
    int low, high, med;

    low = -1;
    high = n;
    
    while ( high - low > 1 )
	{
	    med = (high + low) >> 1;
	    if ( x_ >= x[med] ) low = med;
	    else high = med;
	}
    
    return low;

}

/* ------------------------------------------------------------------------ */

/* given an array x[0..n-1], a value x_, and an initial index low, */
/* hunt returns an index low */
/* such that x_ is in the interval [x[low], x[low+1]). */
/* low = -1 or low = n-1 indicates that x_ is out of range. */

void hunt(int n, double *x, double x_, int *low)
{
    int high, med, inc;

    inc = 1;
    if ( x_ >= x[*low] )
	{
	    /* hunt up */
	    high = *low + inc;
	    while ( x_ >= x[high] )
		{
		    inc <<= 1;
		    *low = high;
		    high += inc;
		    if ( high >= n )
			{
			    /* end of array */
			    high = n;
			    break;
			}
		}
	}
    else
	{
	    /* hunt down */
	    high = *low;
	    *low -= inc;
	    while ( x_ < x[*low] )
		{
		    inc <<= 1;
		    high = *low;
		    *low -= inc;
		    if ( *low <= -1 )
			{
			    /* end of array */
			    *low = -1;
			    break;
			}
		}
	}

    /* bisection */
    while ( high - *low > 1 )
	{
	    med = (high + *low) >> 1;
	    if (x_ >= x[med]) *low = med;
	    else high = med;
	}
    
}

/* end of file */

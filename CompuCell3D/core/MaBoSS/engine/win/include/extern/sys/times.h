#ifndef _TIMES_H
#define _TIMES_H

#if defined(_WIN32) || defined(__MINGW32__)

	#include <time.h>

	// Returns information about a process' consumption of processor time
	struct tms
	  {
		// Total processor time used in executing its instructions
		clock_t tms_utime;
		// Total processor time used
		clock_t tms_stime;
		// Sum of tms_utime and tms_cutime of all terminated child processes
		clock_t tms_cutime;
		// Like tms_cutime but for tms_stime
		clock_t tms_cstime;
	  };

	// Stores information about the processor time in the passed calling process
	clock_t times(struct tms *_buffer);

#else
	#include <sys/times.h>
#endif

#endif

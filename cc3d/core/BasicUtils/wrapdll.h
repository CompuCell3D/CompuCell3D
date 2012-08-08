/* wrapdll.h */

/* Defines DLLEXPORT and DLLCALL for cross-platform development wrappers */

/* $Id: wrapdll.h,v 1.4 2002/07/21 23:03:20 rswindell Exp $ */

/****************************************************************************
 * @format.tab-size 4		(Plain Text/Source Code File Header)			*
 * @format.use-tabs true	(see http://www.synchro.net/ptsc_hdr.html)		*
 *																			*
 * Copyright 2002 Rob Swindell - http://www.synchro.net/copyright.html		*
 *																			*
 * This library is free software; you can redistribute it and/or			*
 * modify it under the terms of the GNU Lesser General Public License		*
 * as published by the Free Software Foundation; either version 2			*
 * of the License, or (at your option) any later version.					*
 * See the GNU Lesser General Public License for more details: lgpl.txt or	*
 * http://www.fsf.org/copyleft/lesser.html									*
 *																			*
 * Anonymous FTP access to the most recent released source is available at	*
 * ftp://vert.synchro.net, ftp://cvs.synchro.net and ftp://ftp.synchro.net	*
 *																			*
 * Anonymous CVS access to the development source and modification history	*
 * is available at cvs.synchro.net:/cvsroot/sbbs, example:					*
 * cvs -d :pserver:anonymous@cvs.synchro.net:/cvsroot/sbbs login			*
 *     (just hit return, no password is necessary)							*
 * cvs -d :pserver:anonymous@cvs.synchro.net:/cvsroot/sbbs checkout src		*
 *																			*
 * For Synchronet coding style and modification guidelines, see				*
 * http://www.synchro.net/source.html										*
 *																			*
 * You are encouraged to submit any modifications (preferably in Unix diff	*
 * format) via e-mail to mods@synchro.net									*
 *																			*
 * Note: If this box doesn't appear square, then you need to fix your tabs.	*
 ****************************************************************************/

#ifndef _WRAPDLL_H
#define _WRAPDLL_H

#if defined(DLLEXPORT)
	#undef DLLEXPORT
#endif
#if defined(DLLCALL)
	#undef DLLCALL
#endif

#if defined(_WIN32) && (defined(WRAPPER_IMPORTS) || defined(WRAPPER_EXPORTS))
	#if defined(WRAPPER_IMPORTS)
		#define DLLEXPORT	__declspec(dllimport)
	#else
		#define DLLEXPORT	__declspec(dllexport)
	#endif
	#if defined(__BORLANDC__)
		#define DLLCALL __stdcall
	#else
		#define DLLCALL
	#endif
#else	/* !_WIN32 || !_DLL*/
	#define DLLEXPORT
	#define DLLCALL
#endif

#endif


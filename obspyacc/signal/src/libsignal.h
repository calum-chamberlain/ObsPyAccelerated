/*
 * =====================================================================================
 *
 *       Filename:  libsignal.h
 *
 *        Purpose:  Signal processing routines for ObsPy Accelerated
 *
 *        Created:  19 May 2021
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Calum Chamberlain
 *   Organization:  VUW
 *      Copyright:  Calum Chamberlain
 *        License:  GNU Lesser General Public License, Version 3
 *                  (https://www.gnu.org/copyleft/lesser.html)
 *
 * =====================================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if (defined(_MSC_VER))
    #include <float.h>
    #define isnanf(x) _isnan(x)
    #define inline __inline
#endif
#if (defined(__APPLE__) && !isnanf)
    #define isnanf isnan
#endif
#include <fftw3.h>
#if defined(__linux__) || defined(__linux) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    #include <omp.h>
    #ifndef N_THREADS
        #define N_THREADS omp_get_max_threads()
    #endif
#endif
#ifndef OUTER_SAFE
    #if defined(__linux__) || defined(__linux)
        #define OUTER_SAFE 1
    #else
        #define OUTER_SAFE 0
    #endif
#else
    #define OUTER_SAFE 1
#endif

// Function headers
int interpolate(float*, float*, float*, float*, long, long);

int resample();
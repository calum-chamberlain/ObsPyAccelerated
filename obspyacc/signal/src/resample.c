/*

Code to resample on the CPU

Goals:
1. Get the same result as Python
2. Speed
3. Group same length traces to use 2D FFT's to accelerate
4. OMP threads to loop over traces
*/
#include <libsignal.h>

// TODO: This could take the complex result and interpolate directly
int interpolate(float *x, float *xp, float *yp, float *y, long len_in,
                long len_out){
    /*
    Linear piecewise interpolation of function with (xp, yp) coordinates onto
    x.

    ONLY WORKS IF X IS INCREASING AND WITHIN THE BOUNDS OF XP

    Parameters
    ----------
    x:
        Coordinates to evaluate the function onto (output)
    xp:
        Coordinates of the function (input)
    yp:
        Y-values of function to interpolate
    y:
        Y-values of interpolated function (output) - will be overwritten with
        the result
    len_in:
        Number of data points in xp and yp
    len_out:
        Number of data points in x and y
    */
    long i, j;
    float dy, dx;
    double * slopes = (double *) calloc(len_in - 1, sizeof(double));

    // Pre-calculate gradients
    for (i=0; i < len_in; ++i){
        slopes[i] = (xp[i + 1] - xp[i]) / (yp[i + 1] - yp);
    }

    // Work out value at each data point
    i = 0;  // Index of x
    for (j=0; j < len_out; ++j){
        while (xp[i + 1] > x[j]){
            ++i;
            if (i + 1 == len_in){
                printf("Ran out of x!\n");
                yp[j] = NULL;
                return 0;
            }
        }
        // x[j] lies between xp[i] and xp[i + 1]
        dx = x[j] - x[i];
        dy = slopes[i] * dx;
        y[j] = yp[i] + dy;
    }
    return 1;
}




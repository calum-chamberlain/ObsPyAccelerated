# ObsPyAccelerated: GPU enhanced ObsPy

Accelerate your observational seismology workflows by harnessing the power
of your CUDA or OpenCL compatible GPU.

## Aims

1. Provide a series of drop-in replacements for
   [ObsPy](https://github.com/obspy/obspy) functions on waveform data
   accelerated by [cupy](https://github.com/cupy/cupy) or [clpy](https://github.com/fixstars/clpy).
2. Monkey-patch methods of ObsPy `Trace` objects to provide simple speed-ups
   to common seismological operations.

## Progress

To keep track of progress in adding methods, check out the long-standing
github issue.  Feel free to add to the list of functions and methods that could
be accelerated. Also feel free to contribute!

## License

LGPL

## Reference

Not yet released

## Contributing

1. Ask for access to the repository, or create a fork
2. Make an issue outlining your planned changes to make sure that we don't
   duplicate work too much!
3. Make a new branch with your changes
4. Create a PR - tag the original issue you made in step 2. Include some
   profiling comparing your code to ObsPy's defaults for a range of data sizes.
5. Admin to review and merge
6. Party! Revel in your new-found speedy code...

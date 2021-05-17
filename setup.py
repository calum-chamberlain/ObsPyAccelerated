try:
    # use setuptools if we can
    from setuptools import setup
    using_setuptools = True
except ImportError:
    from distutils.core import setup
    using_setuptools = False

import glob
import os
import sys
import shutil
import obspyacc

VERSION = obspyacc.__version__

long_description = '''
GPU accelerated ObsPy
'''

scriptfiles = [f for f in glob.glob("scripts/*") if os.path.isfile(f)]


def setup_package():

    # Figure out whether to add ``*_requires = ['numpy']``.
    build_requires = []
    try:
        import numpy
    except ImportError:
        build_requires = ['numpy>=1.6, <2.0']

    install_requires = [
        'numpy>=1.12', 'matplotlib>=1.3.0', 'obspy>=1.0.3', 'numba']

    setup_args = {
        'name': 'ObsCuPy',
        'version': VERSION,
        'description': 'ObsPyAccelerated: GPU accelerated ObsPy',
        'long_description': long_description,
        'url': 'https://github.com/calum-chamberlain/ObsPyAccelerated',
        'author': 'Calum Chamberlain',
        'author_email': 'calum.chamberlain@vuw.ac.nz',
        'license': 'LGPL',
        'classifiers': [
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'License :: OSI Approved :: GNU Library or Lesser General Public '
            'License (LGPL)',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        'keywords': 'CUDA OpenCL GPU ObsPy Observational Seismology',
        'scripts': scriptfiles,
        'install_requires': install_requires,
        'setup_requires': ['pytest-runner'],
        'tests_require': ['pytest>=2.0.0', 'pytest-cov', 'pytest-pep8',
                          'pytest-xdist', 'pytest-rerunfailures',
                          'obspy>=1.1.0', 'numba'],
    }

    if using_setuptools:
        setup_args['setup_requires'] = build_requires
        setup_args['install_requires'] = install_requires

    if len(sys.argv) >= 2 and (
        '--help' in sys.argv[1:] or
        sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                        'clean')):
        # For these actions, NumPy is not required.
        pass
    else:
        setup_args['packages'] = [
            'obspyacc', 'obspyacc.core', 'obspyacc.signal']
    if os.path.isdir("build"):
        shutil.rmtree("build")
    setup(**setup_args)


if __name__ == '__main__':
    setup_package()
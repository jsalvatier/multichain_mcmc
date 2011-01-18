'''
Created on Oct 24, 2009

# Author: John Salvatier <jsalvati@u.washington.edu>, 2009.
'''
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


DISTNAME            = 'multichain_mcmc'
DESCRIPTION         = "Multichain MCMC framework and algorithms"
LONG_DESCRIPTION    ="""
                    A simple framework based on PyMC for multichain MCMC algorithms. 
                    
                    Contains working implementations of:
                        * DREAM/DREAM_ZS sampler : multichain_mcmc.dream.DreamSampler
                        * Adaptive Metropolis Adjusted Langevin Algorithm (AMALA) sampler : multichain_mcmc.amala.AmalaSampler
                        
                    See the sampler classes for details. AMALA sampler requires PyMC branch with gradient information support to function.
                        http://github.com/pymc-devs/pymc/tree/gradientBranch
                    """
MAINTAINER          = 'John Salvatier'
MAINTAINER_EMAIL    = "jsalvati@u.washington.edu"
URL                 = "pypi.python.org/pypi/multichain_mcmc"
LICENSE             = "BSD"
VERSION             = "0.3"

classifiers =  ['Development Status :: 3 - Alpha',
                'Programming Language :: Python',
                'License :: OSI Approved :: BSD License',
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Operating System :: OS Independent']

if __name__ == "__main__":

    setup(name = DISTNAME,
          version = VERSION,
        maintainer  = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        description = DESCRIPTION,
        license = LICENSE,
        url = URL,
        long_description = LONG_DESCRIPTION,
        packages = ['multichain_mcmc'], 
        classifiers =classifiers,
        install_requires=['pymc', 'numpy','scipy', 'cython'],
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("multichain_mcmc.rand_no_replace", ["multichain_mcmc/rand_no_replace.pyx"])])


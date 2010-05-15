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
                        1) DREAM/DREAM_ZS sampler
                        2) Adaptive Metropolis Adjusted Langevin Algorithm (AMALA) sampler
                    
                    1) DREAM_ZSimplementation based on the algorithms presented in the following two papers:
                    
                        C.J.F. ter Braak, and J.A. Vrugt, Differential evolution Markov chain with
                        snooker updater and fewer chains, Statistics and Computing, 18(4),
                        435-446, doi:10.1007/s11222-008-9104-9, 2008.
                        
                        J.A. Vrugt, C.J.F. ter Braak, C.G.H. Diks, D. Higdon, B.A. Robinson, and
                        J.M. Hyman, Accelerating Markov chain Monte Carlo simulation by
                        differential evolution with self-adaptive randomized subspace sampling,
                        International Journal of Nonlinear Sciences and Numerical
                        Simulation, 10(3), 273-290, 2009.
                    2) AMALA implementation based on 
                    
                        AMALA sampler requires PyMC branch with gradient information support to function.
                        http://github.com/pymc-devs/pymc/tree/gradientBranch
                    """
MAINTAINER          = 'John Salvatier'
MAINTAINER_EMAIL    = "jsalvati@u.washington.edu"
URL                 = "pypi.python.org/pypi/multichain_mcmc"
LICENSE             = "BSD"
VERSION             = "0.2.2"

classifiers =  ['Development Status :: 2 - Pre-Alpha',
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
        install_requires=["pymc >= 2.1alpha", "numpy >= 1.2",'scipy >= 0.7', 'cython'],
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("multichain_mcmc.rand_no_replace", ["multichain_mcmc/rand_no_replace.pyx"])])


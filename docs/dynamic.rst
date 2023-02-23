dynamic
==========

This module implements algorithms for optimization and sampling for consensus based
particles systems. All the dynamics inherit from the base class ``ParticleDynamic``.

.. currentmodule:: polarcbo.dynamic

.. autoclass:: ParticleDynamic
   :members:
   :undoc-members:
   :show-inheritance:




Standard Consensus Based Schemes
--------------------------------

The following classes implement standard consensus based schemes.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   CBO

Polarized Consensus Based Schemes
---------------------------------

The following classes implement polarized consensus based schemes as described in
[1]_.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   PolarCBO
   PolarCBS

Cluster Consensus Based Schemes
---------------------------------

These classes implement cluster consensus based schemes as described in [1]_. 


.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   CCBO
   CCBS

Not maintained
--------------

The following classes implement dynmaics that are currently not maintained.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:
   :template: classtemplate.rst

   EMCBO
   KMeansCBO


References
----------

.. [1] Bungert, L., Wacker, P., & Roith, T. (2022). Polarized consensus-based 
   dynamics for optimization and sampling. arXiv preprint arXiv:2211.05238.




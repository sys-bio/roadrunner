.. currentmodule:: RoadRunner

***********************
Stoichiometric Analysis
***********************

Preliminaries
-------------

A network of :math:`m` chemical species and :math:`n` reactions can be described by the
:math:`m` by :math:`n` stochiometry matrix :math:`\mathbf{N}`. :math:`\mathbf{N}_{i,j}` is the net number of  
species :math:`i` produced or consumed in reaction :math:`j`.  The dynamics of the network are
described by 

.. math::

   \frac{d}{dt}\mathbf{s}(t) = \mathbf{N} \mathbf{v}(\mathbf{s}(t),\mathbf{p},t), 

where :math:`\mathbf{s}` is the vector of species concentrations, :math:`\mathbf{p}` 
is a vector of time independent parameters, and :math:`t` is time.

Each structural conservation, or interchangably, *conserved sum* (e.g. conserved moiety) in 
the network coresponds to a lineraly dependent row in the stoichiometry matrix :math:`\mathbf{N}`.

If there are conserved sums, then the row rank, :math:`r` of :math:`N` is :math:`< m`, and 
the stochiometry matrix :math:`N` may first be re-ordered such that the first :math:`r` are linearly
independent, and the remaining :math:`m-r` rows are linear combinations of the first :math:`r`
rows. The *reduced stochiometry matrix* :math:`\mathbf{N_R}` is then formed from the first
:math:`r` rows of :math:`N`. Finally, :math:`N` may be expressed as a product of the :math:`m \times r`
*link matrix* :math:`\mathbf{L}` and the :math:`r \times n` :math:`\mathbf{N_R}` matrix: 

.. math::

   \mathbf{N} = \mathbf{L}\mathbf{N_R}.

The link matrix :math:`\mathbf{L}` has the form

.. math::

   \mathbf{L} = \left[ \begin{array}{c}
                       \mathbf{I}_{r} \\
                       \mathbf{L}_0    \end{array} \right],

where :math:`\mathbf{I}_{r}` is the :math:`r \times r` identity matrix and :math:`\mathbf{L}_0` is a :math:`(m-r) \times r`
matrix.

Full vs. Extended Stoichiometry Matrix
--------------------------------------

The "full" stoichiometric matrix includes any conserved quantities (as opposed to the reduced stoichiometric matrix,
which does not). The *extended* stoichiometric matrix is equal to the full stoichiometric matrix plus additional
rows representing boundary species and sources / sinks. For example, consider the following reaction system:

::

    reaction1:  => C
    reaction2: C =>
    reaction3: C =>
    reaction4: MI => M
    reaction5: M => MI
    reaction6: XI => X
    reaction7: X => XI

The extended stoichiometry matrix for this system is:

     >>>  rr.getExtendedStoichiometryMatrix()
                          reaction1, reaction2, reaction3, reaction4, reaction5, reaction6, reaction7
      C                [[         1,        -1,        -1,         0,         0,         0,         0],
      M                 [         0,         0,         0,         1,        -1,         0,         0],
      X                 [         0,         0,         0,         0,         0,         1,        -1],
      MI                [         0,         0,         0,        -1,         1,         0,         0],
      XI                [         0,         0,         0,         0,         0,        -1,         1],
      reaction1_source  [        -1,         0,         0,         0,         0,         0,         0],
      reaction2_sink    [         0,         1,         0,         0,         0,         0,         0],
      reaction3_sink    [         0,         0,         1,         0,         0,         0,         0]]

Methods
-------

The following methods are related to the analysis of the stoichiometric matrix.

.. autosummary::

   RoadRunner.getLinkMatrix
   RoadRunner.getNrMatrix
   RoadRunner.getKMatrix
   RoadRunner.getConservationMatrix
   RoadRunner.getL0Matrix
   RoadRunner.getFullStoichiometryMatrix
   RoadRunner.getExtendedStoichiometryMatrix
   RoadRunner.getReducedStoichiometryMatrix
   ExecutableModel.getNumConservedMoieties
   ExecutableModel.getConservedMoietyIds
   ExecutableModel.getConservedMoietyValues
   ExecutableModel.setConservedMoietyValues

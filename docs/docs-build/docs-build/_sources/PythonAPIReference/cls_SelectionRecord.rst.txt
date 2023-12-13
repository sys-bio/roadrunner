SelectionRecord
_______________

.. class:: SelectionRecord(str)
   :module: roadrunner

   RoadRunner provides a range of flexible ways of selecting values from
   a simulation. These values can not only be calculated directly via
   RoadRunner.getSelectionValue, but any of these selections can be
   used as columns in the simulate result matrix.

   The SectionRecord.selectionType should be one of the constants listed
   here.

   p1 and p2 may be required along with the attribute to set certain selection types that use one or more arguments.
   
   Below is the list of attributes which are FOR ADVANCED USERS ONLY. For simplified usage, refer to :ref:`Selecting Values <selecting-values>`.
   

.. method::SelectionRecord.__init__(str)
   Create a new selection record. This constructor really should not be called,
   SelectionRecords should be created by the RoadRunner.createSelection

.. attribute:: SelectionRecord.ALL

.. attribute:: SelectionRecord.ALL_DEPENDENT

.. attribute:: SelectionRecord.ALL_DEPENDENT_AMOUNT

.. attribute:: SelectionRecord.ALL_DEPENDENT_CONCENTRATION

.. attribute:: SelectionRecord.ALL_INDEPENDENT

.. attribute:: SelectionRecord.ALL_INDEPENDENT_AMOUNT

.. attribute:: SelectionRecord.ALL_INDEPENDENT_CONCENTRATION

.. attribute:: SelectionRecord.AMOUNT

   species must have either a CONCENTRATION or AMOUNT
   modifer to distinguish it.

.. attribute:: SelectionRecord.BOUNDARY

   species must have either a BOUNDARY or FLOATING modifiers.

.. attribute:: SelectionRecord.BOUNDARY_AMOUNT   

   boundary species amount

.. attribute:: SelectionRecord.BOUNDARY_CONCENTRATION

   boundary species concentration
   
.. attribute:: SelectionRecord._COMPARTMENT

   Compartments and parameters can be either current or initial values. These values with and underscore, '_' are intended to be used with either an CURRENT or INITIAL value modifier.

.. attribute:: SelectionRecord.COMPARTMENT

   the current compartment value   
   
.. attribute:: SelectionRecord.CONCENTRATION

   species must have either a CONCENTRATION or AMOUNT
   modifer to distinguish it.

.. attribute:: SelectionRecord.CONSREVED_MOIETY   
   
.. attribute:: SelectionRecord.CONTROL   

   scaled control coefficient of designated arguments.
   
.. attribute:: SelectionRecord.CURRENT

.. attribute:: SelectionRecord.DEPENDENT

.. attribute:: SelectionRecord.EIGENVALUE

   real part of eigenvalue of designated identifier.

.. attribute:: SelectionRecord.EIGENVALUE_COMPLEX

   complex part of eigenvalue of designated identifier.

.. attribute:: SelectionRecord.ELASTICITY

   scaled elasticity coefficient of designated arguments.

.. attribute:: SelectionRecord.ELEMENT

.. attribute:: SelectionRecord.FLOATING
   
   species must have either a BOUNDARY or FLOATING modifiers.

.. attribute:: SelectionRecord.FLOATING_AMOUNT

   floating species current amounts.

.. attribute:: SelectionRecord.FLOATING_AMOUNT_RATE   

   floating species amount rates (value, not reaction rates)

.. attribute:: SelectionRecord.FLOATING_CONCENTRATION

   floating species current concentrations.

.. attribute:: SelectionRecord.FLOATING_CONCENTRATION_RATE

.. attribute:: SelectionRecord.GLOBAL_PARAMETER

   the current global parameter value

.. attribute:: SelectionRecord.INDEPENDENT

.. attribute:: SelectionRecord.INITIAL

.. attribute:: SelectionRecord.INITIAL_FLOATING_AMOUNT

   initial amount of designated identifier.

.. attribute:: SelectionRecord.INITIAL_FLOATING_CONCENTRATION

   initial concentration of designated identifier.

.. attribute:: SelectionRecord.RATE

.. attribute:: SelectionRecord.REACTION

.. attribute:: SelectionRecord.REACTION_RATE

   current reaction rate

.. attribute:: SelectionRecord.STOICHIOMETRY

   stoichiometric coefficient of designated identifier and reaction.

.. attribute:: SelectionRecord.TIME

.. attribute:: SelectionRecord.UNKNOWN

.. attribute:: SelectionRecord.UNKNOWN_CONCENTRATION

.. attribute:: SelectionRecord.UNKNOWN_ELEMENT

.. attribute:: SelectionRecord.UNSCALED

.. attribute:: SelectionRecord.UNSCALED_CONTROL

   unscaled control coefficient of designated arguments.

.. attribute:: SelectionRecord.UNSCALED_ELASTICITY

   unscaled elasticity coefficient of designated arguments.

.. attribute:: SelectionRecord.index
   :module: RoadRunner
   :annotation: int


.. attribute:: SelectionRecord.p1
   :module: RoadRunner
   :annotation: str

   first of the arguments

.. attribute:: SelectionRecord.p2
   :module: RoadRunner
   :annotation: str
   
   second of the arguments


.. attribute:: SelectionRecord.selectionType
   :module: RoadRunner
   :annotation: int


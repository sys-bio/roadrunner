#ifndef rrSelectionRecordH
#define rrSelectionRecordH
#include "rrExporter.h"
#include <ostream>
#include <string>

namespace rr
{
using std::string;
using std::ostream;

/**
 * a way to find sbml model elements using the RoadRunner syntax.
 */
class RR_DECLSPEC SelectionRecord
{
public:
    enum SelectionType
    {
        /**
        * SelectionType for time.
        */
        TIME =                              (0x1 << 0),  // => 0x00000001,

        /**
         * SelectionType for concentrations. Species must have either a CONCENTRATION or AMOUNT
         * modifer to distinguish it.
         */
        CONCENTRATION =                     (0x1 << 1),

        /**
        * SelectionType for amounts. Species must have either a CONCENTRATION or AMOUNT
        * modifer to distinguish it.
        */
        AMOUNT =                            (0x1 << 2),

        /**
        * SelectionType for rates.
        */
        RATE =                              (0x1 << 3),

        /**
         * SelectionType for bounary species. Species must have either a BOUNDARY or FLOATING
         * modifiers.
         */
        BOUNDARY =                          (0x1 << 4),
        /**
        * SelectionType for floating species. Species must have either a BOUNDARY or FLOATING
        * modifiers.
        */
        FLOATING =                          (0x1 << 5),

        /**
         * SelectionType for comparments. Compartments and parameters can be either current
         * or initial values. These values with and underscore, '_'
         * are intended to be used with either an CURRENT or
         * INITIAL value modifier.
         */
        _COMPARTMENT =                      (0x1 << 6),

        /**
        * SelectionType for global parameters. Compartments and parameters can be either current
        * or initial values. These values with and underscore, '_'
        * are intended to be used with either an CURRENT or
        * INITIAL value modifier.
        */
        _GLOBAL_PARAMETER =                 (0x1 << 7),

        /**
        * SelectionType for reactions.
        */
        REACTION =                          (0x1 << 8),

        /**
         * SelectionType for initial values. SBML elements can be accessed as either a initial value, or
         * a current value, they must have one or the other.
         */
        INITIAL =                           (0x1 << 9),

        /**
        * SelectionType for current values. SBML elements can be accessed as either a initial value, or
        * a current value, they must have one or the other.
        */
        CURRENT =                           (0x1 << 10),

        /**
        * SelectionType for unscaled values.
        */
        UNSCALED =                          (0x1 << 11),

        /**
        * SelectionType for elasticities. 
        */
        ELASTICITY =                        (0x1 << 12),

        /**
        * SelectionType for control coefficients. 
        */
        CONTROL =                           (0x1 << 13),

        /**
         * SelectionType for eigenvalues. Eigenvalue without complex is real part of eigenvalue.
         */
        EIGENVALUE =                        (0x1 << 14),

        /**
        * SelectionType for elements. (Unused)
        */
        ELEMENT =                           (0x1 << 15),

        /**
        * SelectionType for stoichiometries.
        */
        STOICHIOMETRY =                     (0x1 << 16),

        /**
        * SelectionType for unknown values.
        */
        UNKNOWN =                           (0x1 << 17),

        /**
        * SelectionType for dependent species. 
        */
        DEPENDENT =                         (0x1 << 18),

        /**
        * SelectionType for independent species. 
        */
        INDEPENDENT =                       (0x1 << 19),

        /**
        * SelectionType for conserved moieties. 
        */
        CONSERVED_MOIETY =                  (0x1 << 20),

        /**
        * SelectionType for state vectors. 
        */
        STATE_VECTOR =                      (0x1 << 21),

        /**
        * SelectionType for events. 
        */
        EVENT =                             (0x1 << 22),

        /**
         * SelectionType for complex values. The complex bit, off to get real values, on to get imaginary parts.
         *
         * Only used with EIGENVALUE currently.
         */
        COMPLEX =                           (0x1 << 23),

        /**
         * SelectionType for complex eigenvalues.
         */
        EIGENVALUE_COMPLEX =                EIGENVALUE | COMPLEX,

        /**
        * SelectionType for unknown concentrations.
        */
        UNKNOWN_CONCENTRATION =             UNKNOWN | CONCENTRATION,

        /**
         * SelectionType for the current compartment value.
         */
        COMPARTMENT =                       _COMPARTMENT | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * SelectionType for the current global parameter value.
         */
        GLOBAL_PARAMETER =                  _GLOBAL_PARAMETER | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * SelectionType for current floating species amounts.
         */
        FLOATING_AMOUNT =                   FLOATING | AMOUNT | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * SelectionType for independent floating species amounts.
         */
        INDEPENDENT_FLOATING_AMOUNT =       FLOATING | AMOUNT | INDEPENDENT | CURRENT,

        /**
         * SelectionType for dependent floating species amounts. (Unused)
         */
        DEPENDENT_FLOATING_AMOUNT =         FLOATING | AMOUNT | DEPENDENT   | CURRENT,

        /**
         * SelectionType for boundary species amounts.
         */
        BOUNDARY_AMOUNT =                   BOUNDARY | AMOUNT | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * SelectionType for current boundary species concentrations.
         */
        BOUNDARY_CONCENTRATION =            BOUNDARY | CONCENTRATION | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * SelectionType for current floating species concentrations.
         */
        FLOATING_CONCENTRATION =            FLOATING | CONCENTRATION | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * SelectionType for floating species amount rates (value, not reaction rates),
         * these are always current.
         */
        FLOATING_AMOUNT_RATE =              FLOATING | RATE | DEPENDENT,

        /**
         * SelectionType for floating species concentration rates (value, not reaction rates),
         * these are always current. (Unused)
         */
        FLOATING_CONCENTRATION_RATE =       FLOATING | CONCENTRATION | RATE | DEPENDENT,


        /**
        * SelectionType for global parameter rates.
        */
        GLOBAL_PARAMETER_RATE =             _GLOBAL_PARAMETER | RATE,

        /**
         * SelectionType for reaction rate, always current
         */
        REACTION_RATE =                     REACTION | RATE | DEPENDENT,

        /**
         * SelectionType for initial species amounts.
         */
        INITIAL_AMOUNT =                    INITIAL | AMOUNT | INDEPENDENT | DEPENDENT,

        /**
         * SelectionType for initial floating species amounts.
         */
        INITIAL_FLOATING_AMOUNT =           INITIAL | FLOATING | AMOUNT | INDEPENDENT | DEPENDENT,

        /**
         * SelectionType for initial species concentrations.
         */
        INITIAL_CONCENTRATION =             INITIAL | CONCENTRATION | INDEPENDENT | DEPENDENT,

        /**
         * SelectionType for initial floating species concentrations.
         */
        INITIAL_FLOATING_CONCENTRATION =    INITIAL | FLOATING | CONCENTRATION | INDEPENDENT | DEPENDENT,

        /**
         * SelectionType for initial compartment values.
         */
        INITIAL_COMPARTMENT =               INITIAL | _COMPARTMENT | INDEPENDENT | DEPENDENT,

        /**
         * SelectionType for initial global parameter values.
         */
        INITIAL_GLOBAL_PARAMETER =          INITIAL | _GLOBAL_PARAMETER | INDEPENDENT | DEPENDENT,


        /**
         * SelectionType for global parameters that have initial assignment rules.
         */
        DEPENDENT_INITIAL_GLOBAL_PARAMETER =  INITIAL | _GLOBAL_PARAMETER | DEPENDENT,

        /**
        * SelectionType for unscaled elasticity.
        */
        UNSCALED_ELASTICITY =               UNSCALED | ELASTICITY,

        /**
        * SelectionType for unscaled control coefficient.
        */
        UNSCALED_CONTROL =                  UNSCALED | CONTROL,

        /**
        * SelectionType for unscaled control coefficient.
        */
        UNKNOWN_ELEMENT =                   UNKNOWN | ELEMENT,

        /**
        * SelectionType for everything.
        */
        ALL =                               (0xffffffff),

        /**
        * SelectionType for all values related to independent species. (Unused)
        */
        ALL_INDEPENDENT =                   ~DEPENDENT,

        /**
        * SelectionType for all values related to dependent species. (Unused)
        */
        ALL_DEPENDENT =                     ~INDEPENDENT,

        /**
        * SelectionType for all independent species amount. (Unused)
        */
        ALL_INDEPENDENT_AMOUNT =            ~DEPENDENT & ~CONCENTRATION,

        /**
        * SelectionType for all dependent species amount. (Unused)
        */
        ALL_DEPENDENT_AMOUNT =              ~INDEPENDENT & ~CONCENTRATION,

        /**
        * SelectionType for all independent species concentration. (Unused)
        */
        ALL_INDEPENDENT_CONCENTRATION =     ~DEPENDENT & ~AMOUNT,

        /**
        * SelectionType for all dependent species concentration. (Unused)
        */
        ALL_DEPENDENT_CONCENTRATION =       ~INDEPENDENT & ~AMOUNT,


        /**
         * The combination of values that uniquely define the current state of
         * the sbml model.
         *
         * Note, any sbml variable can be defined by a rule, the model state
         * is defined as the set of variabls that define the current state of the
         * model. These do not include the initial values as the initial values
         * can not be changed by advancing the model with the integrator. (Unused)
         */
        MODEL_STATE =                       ALL_INDEPENDENT_AMOUNT & ~INITIAL,


        /**
         * If this bit is set, then the reset function evaluates all of the
         * SBML model init conditions (init values and all init assignment rules),
         * and assigns them to the model state variable. This setting brings
         * the model back the originally loaded state.
         */
        SBML_INITIALIZE =                   STATE_VECTOR   // the STATE_VECTOR options
                                                           // is used here because it is
                                                           // not used anywhere else in
                                                           // the reset options, and gives
                                                           // the indication that everything is set,
                                                           // mostly it saves a not wasting a bit value.
    };

    /**
     * After the selection record is created by the roadrunner object, this
     * is an index into the variables list that the ExecutableModel has.
     */
    int index;

    /**
     * For single argument selections, i.e. ee( S1 ), this is the argument, for
     * two arg selections, this is the first arg.
     */
    string p1;

    /**
     * the second argument of two arg selections.
     */
    string p2;

    /**
     * the type of selection.
     */
    SelectionType selectionType;

    /**
     * Selections really should be constructed by the RoadRunner::createSelection
     * method.
     *
     * This just creates a new ebmpy selection.
     */
    SelectionRecord(const int& index = 0,
            const SelectionType type = UNKNOWN,
            const string& p1 = "", const string& p2 = "");

    /**
     * creates a empty selection.
     */
    SelectionRecord(const std::string str);

    /**
     * get a short description of this selection, this returns a string
     * compatable with RoadRunner::createSelection.
     */
    std::string to_string() const;

    /**
     * gets a longer description of this selection, this is intended
     * to called as a python __repr__
     */
    std::string to_repr() const;
};

ostream& operator<< (ostream& stream, const SelectionRecord& rec);
}


#endif

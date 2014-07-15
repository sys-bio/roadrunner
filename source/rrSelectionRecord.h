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
        TIME =                              (0x1 << 0),  // => 0x00000001,

        /**
         * species must have either a CONCENTRATION or AMOUNT
         * modifer to distinguish it.
         */
        CONCENTRATION =                     (0x1 << 1),
        AMOUNT =                            (0x1 << 2),

        RATE =                              (0x1 << 3),

        /**
         * species must have either a BOUNDARY or FLOATING
         * modifiers.
         */
        BOUNDARY =                          (0x1 << 4),
        FLOATING =                          (0x1 << 5),

        /**
         * Compartments and parameters can be either current
         * or initial values. These values with and underscore, '_'
         * are intended to be used with either an CURRENT or
         * INITIAL value modifier.
         */
        _COMPARTMENT =                      (0x1 << 6),
        _GLOBAL_PARAMETER =                 (0x1 << 7),


        REACTION =                          (0x1 << 8),

        /**
         * sbml elements can be accessed as either a initial value, or
         * a current value, they must have one or the other.
         */
        INITIAL =                           (0x1 << 9),
        CURRENT =                           (0x1 << 10),
        UNSCALED =                          (0x1 << 11),
        ELASTICITY =                        (0x1 << 12),
        CONTROL =                           (0x1 << 13),
        EIGENVALUE =                        (0x1 << 14),
        ELEMENT =                           (0x1 << 15),
        STOICHIOMETRY =                     (0x1 << 16),
        UNKNOWN =                           (0x1 << 17),
        DEPENDENT =                         (0x1 << 18),
        INDEPENDENT =                       (0x1 << 19),
        CONSREVED_MOIETY =                  (0x1 << 20),
        STATE_VECTOR =                      (0x1 << 21),
        EVENT =                             (0x1 << 22),

        UNKNOWN_CONCENTRATION =             UNKNOWN | CONCENTRATION,

        /**
         * the current compartment value
         */
        COMPARTMENT =                       _COMPARTMENT | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * the current global parameter value
         */
        GLOBAL_PARAMETER =                  _GLOBAL_PARAMETER | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * current amounts
         */
        FLOATING_AMOUNT =                   FLOATING | AMOUNT | INDEPENDENT | DEPENDENT | CURRENT,
        BOUNDARY_AMOUNT =                   BOUNDARY | AMOUNT | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * current concentrations
         */
        BOUNDARY_CONCENTRATION =            BOUNDARY | CONCENTRATION | INDEPENDENT | DEPENDENT | CURRENT,
        FLOATING_CONCENTRATION =            FLOATING | CONCENTRATION | INDEPENDENT | DEPENDENT | CURRENT,

        /**
         * floating species value rates (value, not reaction rates),
         * these are always current
         */
        FLOATING_AMOUNT_RATE =              FLOATING | RATE | DEPENDENT,
        FLOATING_CONCENTRATION_RATE =       FLOATING | CONCENTRATION | RATE | DEPENDENT,

        GLOBAL_PARAMETER_RATE =             _GLOBAL_PARAMETER | RATE,

        /**
         * reaction rate, always current
         */
        REACTION_RATE =                     REACTION | RATE | DEPENDENT,

        /**
         * initial species values
         */
        INITIAL_AMOUNT =                    INITIAL | AMOUNT | INDEPENDENT | DEPENDENT,
        INITIAL_FLOATING_AMOUNT =           INITIAL | FLOATING | AMOUNT | INDEPENDENT | DEPENDENT,


        INITIAL_CONCENTRATION =             INITIAL | CONCENTRATION | INDEPENDENT | DEPENDENT,
        INITIAL_FLOATING_CONCENTRATION =    INITIAL | FLOATING | CONCENTRATION | INDEPENDENT | DEPENDENT,


        INITIAL_COMPARTMENT =               INITIAL | _COMPARTMENT | INDEPENDENT | DEPENDENT,

        INITIAL_GLOBAL_PARAMETER =          INITIAL | _GLOBAL_PARAMETER | INDEPENDENT | DEPENDENT,


        UNSCALED_ELASTICITY =               UNSCALED | ELASTICITY,
        UNSCALED_CONTROL =                  UNSCALED | CONTROL,
        UNKNOWN_ELEMENT =                   UNKNOWN | ELEMENT,
        ALL =                               (0xffffffff),

        ALL_INDEPENDENT =                   ~DEPENDENT,
        ALL_DEPENDENT =                     ~INDEPENDENT,

        ALL_INDEPENDENT_AMOUNT =            ~DEPENDENT & ~CONCENTRATION,
        ALL_DEPENDENT_AMOUNT =              ~INDEPENDENT & ~CONCENTRATION,

        ALL_INDEPENDENT_CONCENTRATION =     ~DEPENDENT & ~AMOUNT,
        ALL_DEPENDENT_CONCENTRATION =       ~INDEPENDENT & ~AMOUNT,


        /**
         * The combination of values that uniquely define the current state of
         * the sbml model.
         *
         * Note, any sbml variable can be defined by a rule, the model state
         * is defined as the set of variabls that define the current state of the
         * model. These do not include the initial values as the initial values
         * can not be changed by advancing the model with the integrator.
         */
        MODEL_STATE =                       ALL_INDEPENDENT_AMOUNT & ~INITIAL
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

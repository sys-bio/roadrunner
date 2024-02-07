#pragma hdrstop
#include "rrSelectionRecord.h"

#include <vector>
#include <sstream>

#include <iostream>


// TODO When we have gcc 4.4 as minimal compiler, drop poco and use C++ standard regex
#include <Poco/RegularExpression.h>

using Poco::RegularExpression;

// NOTE: The regular expressions need to be file scope static as they cause problems when
// created as function scope static.

static const Poco::RegularExpression is_time_re("^\\s*time\\s*$", RegularExpression::RE_CASELESS);
static bool is_time(const std::string& str)
{
    return is_time_re.match(str);
}

static const Poco::RegularExpression is_uec_re("^\\s*uec\\s*\\(\\s*(\\w*)\\s*,\\s*(\\w*)\\s*\\)\\s*$", RegularExpression::RE_CASELESS);
static bool is_uec(const std::string& str, std::string& p1, std::string& p2)
{
    std::vector<std::string> matches;

    int nmatch = is_uec_re.split(str, matches);

    if (nmatch == 3)
    {
        p1 = matches[1];
        p2 = matches[2];
        return true;
    }
    else
    {
        return false;
    }
}

static const Poco::RegularExpression is_ec_re("^\\s*ec\\s*\\(\\s*(\\w*)\\s*,\\s*(\\w*)\\s*\\)\\s*$", RegularExpression::RE_CASELESS);
static bool is_ec(const std::string& str, std::string& p1, std::string& p2)
{
    std::vector<std::string> matches;

    int nmatch = is_ec_re.split(str, matches);

    if (nmatch == 3)
    {
        p1 = matches[1];
        p2 = matches[2];
        return true;
    }
    else
    {
        return false;
    }
}

static Poco::RegularExpression is_ucc_re("^\\s*ucc\\s*\\(\\s*(\\w*)\\s*,\\s*(\\w*)\\s*\\)\\s*$", RegularExpression::RE_CASELESS);
static bool is_ucc(const std::string& str, std::string& p1, std::string& p2)
{
    std::vector<std::string> matches;

    int nmatch = is_ucc_re.split(str, matches);

    if (nmatch == 3)
    {
        p1 = matches[1];
        p2 = matches[2];
        return true;
    }
    else
    {
        return false;
    }
}

static const Poco::RegularExpression is_cc_re("^\\s*cc\\s*\\(\\s*(\\w*)\\s*,\\s*(\\w*)\\s*\\)\\s*$", RegularExpression::RE_CASELESS);
static bool is_cc(const std::string& str, std::string& p1, std::string& p2)
{
    std::vector<std::string> matches;

    int nmatch = is_cc_re.split(str, matches);

    if (nmatch == 3)
    {
        p1 = matches[1];
        p2 = matches[2];
        return true;
    }
    else
    {
        return false;
    }
}

static const Poco::RegularExpression is_stoich_re("^\\s*stoich\\s*\\(\\s*(\\w*)\\s*,\\s*(\\w*)\\s*\\)\\s*$", RegularExpression::RE_CASELESS);
static bool is_stoich(const std::string& str, std::string& p1, std::string& p2)
{
    std::vector<std::string> matches;

    int nmatch = is_stoich_re.split(str, matches);

    if (nmatch == 3)
    {
        p1 = matches[1];
        p2 = matches[2];
        return true;
    }
    else
    {
        return false;
    }
}

static const Poco::RegularExpression is_concentration_re("^\\s*\\[\\s*(\\w*)\\s*\\]\\s*$", RegularExpression::RE_CASELESS);
static bool is_concentration(const std::string& str, std::string& p1)
{
    std::vector<std::string> matches;

    int nmatch = is_concentration_re.split(str, matches);

    if (nmatch == 2)
    {
        p1 = matches[1];
        return true;
    }
    else
    {
        return false;
    }
}

static const Poco::RegularExpression is_concentration_rate_re("^\\s*\\[\\s*(\\w*)\\s*\\]\\'\\s*$", RegularExpression::RE_CASELESS);
static bool is_concentration_rate(const std::string& str, std::string& p1)
{
    std::vector<std::string> matches;

    int nmatch = is_concentration_rate_re.split(str, matches);

    if (nmatch == 2)
    {
        p1 = matches[1];
        return true;
    }
    else
    {
        return false;
    }
}

static const Poco::RegularExpression is_symbol_re("^\\s*(\\w*)\\s*$", RegularExpression::RE_CASELESS);
static bool is_symbol(const std::string& str, std::string& p1)
{
    std::vector<std::string> matches;

    int nmatch = is_symbol_re.split(str, matches);

    if (nmatch == 2)
    {
        p1 = matches[1];
        return true;
    }
    else
    {
        return false;
    }
}

static const Poco::RegularExpression is_amount_rate_re("^\\s*(\\w*)\\s*'\\s*$", RegularExpression::RE_CASELESS);
static bool is_amount_rate(const std::string& str, std::string& p1)
{


    std::vector<std::string> matches;

    int nmatch = is_amount_rate_re.split(str, matches);

    if (nmatch == 2)
    {
        p1 = matches[1];
        return true;
    }
    else
    {
        return false;
    }
}

static const Poco::RegularExpression is_eigen_re("^\\s*(eigen|eigenReal|eigenImag)\\s*\\(\\s*(\\w*)\\s*\\)\\s*$", RegularExpression::RE_CASELESS);
static bool is_eigen(const std::string& str, std::string& p1, int& complex)
{
    std::vector<std::string> matches;

    int nmatch = is_eigen_re.split(str, matches);

    if (nmatch == 3)
    {
        p1 = matches[2];

        if (matches[1] == "eigen")
        {
            complex = 1;
        }
        else if (matches[1] == "eigenReal")
        {
            complex = 2;
        }
        else
        {
            complex = 3;
        }
        return true;
    }
    else
    {
        return false;
    }
}


static const Poco::RegularExpression is_init_value_re("^\\s*init\\s*\\(\\s*(\\w*)\\s*\\)\\s*$", RegularExpression::RE_CASELESS);
static bool is_init_value(const std::string& str, std::string& p1)
{
    std::vector<std::string> matches;

    int nmatch = is_init_value_re.split(str, matches);

    if (nmatch == 2)
    {
        p1 = matches[1];
        return true;
    }
    else
    {
        return false;
    }
}


static const Poco::RegularExpression is_init_conc_re("^\\s*init\\s*\\(\\s*\\[\\s*(\\w*)\\s*\\]\\s*\\)\\s*$", RegularExpression::RE_CASELESS);
static bool is_init_conc(const std::string& str, std::string& p1)
{
    std::vector<std::string> matches;

    int nmatch = is_init_conc_re.split(str, matches);

    if (nmatch == 2)
    {
        p1 = matches[1];
        return true;
    }
    else
    {
        return false;
    }
}


namespace rr
{





SelectionRecord::SelectionRecord(const int& _index,
        const SelectionType _type, const std::string& _p1,
        const std::string& _p2) :
        index(_index), p1(_p1), p2(_p2), selectionType(_type)
{
}

std::ostream& operator<<(std::ostream& stream, const SelectionRecord& rec)
{
    stream << "A Selection Record --" << std::endl;
    stream << "Index: " << rec.index << std::endl;
    stream << "p1: " << rec.p1 << std::endl;
    stream << "p2: " << rec.p2 << std::endl;
    stream << "SelectionType: " << rec.selectionType << std::endl;
    return stream;
}

std::string SelectionRecord::to_repr() const
{
    std::string selType;

    switch(selectionType) {
    case SelectionRecord::TIME: selType = "TIME"; break;
    case SelectionRecord::BOUNDARY_CONCENTRATION: selType = "BOUNDARY_CONCENTRATION"; break;
    case SelectionRecord::FLOATING_CONCENTRATION: selType = "FLOATING_CONCENTRATION"; break;
    case SelectionRecord::REACTION_RATE: selType = "REACTION_RATE"; break;
    case SelectionRecord::FLOATING_AMOUNT_RATE: selType = "FLOATING_AMOUNT_RATE"; break;
    case SelectionRecord::COMPARTMENT: selType = "COMPARTMENT"; break;
    case SelectionRecord::GLOBAL_PARAMETER: selType = "GLOBAL_PARAMETER"; break;
    case SelectionRecord::FLOATING_AMOUNT: selType = "FLOATING_AMOUNT"; break;
    case SelectionRecord::BOUNDARY_AMOUNT: selType = "BOUNDARY_AMOUNT"; break;
    case SelectionRecord::ELASTICITY: selType = "ELASTICITY"; break;
    case SelectionRecord::UNSCALED_ELASTICITY: selType = "UNSCALED_ELASTICITY"; break;
    case SelectionRecord::CONTROL: selType = "CONTROL"; break;
    case SelectionRecord::UNSCALED_CONTROL: selType = "UNSCALED_CONTROL"; break;
    case SelectionRecord::EIGENVALUE_COMPLEX: selType = "EIGENVALUE_COMPLEX"; break;
    case SelectionRecord::EIGENVALUE_REAL: selType = "EIGENVALUE_REAL"; break;
    case SelectionRecord::EIGENVALUE_IMAG: selType = "EIGENVALUE_IMAG"; break;
    case SelectionRecord::INITIAL_AMOUNT: selType = "INITIAL_AMOUNT"; break;
    case SelectionRecord::INITIAL_CONCENTRATION: selType = "INITIAL_CONCENTRATION"; break;
    case SelectionRecord::STOICHIOMETRY: selType = "STOICHIOMETRY"; break;
    case SelectionRecord::UNKNOWN_ELEMENT: selType = "UNKNOWN_ELEMENT"; break;
    case SelectionRecord::UNKNOWN_CONCENTRATION: selType = "UNKNOWN_CONCENTRATION"; break;
    default: selType = "UNKNOWN"; break;
    }
    std::stringstream s;
    s << "SelectionRecord({'index' : " << index << ", ";
    s << "'p1' : '" << p1 << "', ";
    s << "'p2' : '" << p2 << "', ";
    s << "'selectionType' : " << selType << "})";

    return s.str();
}

rr::SelectionRecord::SelectionRecord(const std::string str) :
        index(-1), selectionType(UNKNOWN)
{
    int complex;
    if (is_ec(str, p1, p2))
    {
        selectionType = ELASTICITY;
    }
    else if(is_uec(str, p1, p2))
    {
        selectionType = UNSCALED_ELASTICITY;
    }
    else if (is_cc(str, p1, p2))
    {
        selectionType = CONTROL;
    }
    else if(is_ucc(str, p1, p2))
    {
        selectionType = UNSCALED_CONTROL;
    }
    else if(is_concentration(str, p1))
    {
        selectionType = UNKNOWN_CONCENTRATION;
    }
    else if (is_concentration_rate(str, p1))
    {
        selectionType = FLOATING_CONCENTRATION_RATE;
    }
    else if(is_amount_rate(str, p1))
    {
        selectionType = FLOATING_AMOUNT_RATE;
    }
    else if(is_eigen(str, p1, complex))
    {
        if(complex == 1)
        {
            selectionType = EIGENVALUE_COMPLEX;
        }
        else if(complex == 2)
        {
            selectionType = EIGENVALUE_REAL;
        }
        else
        {
            selectionType = EIGENVALUE_IMAG;
        }
    }
    else if(is_init_value(str, p1))
    {
        selectionType = INITIAL_AMOUNT;
    }
    else if(is_init_conc(str, p1))
    {
        selectionType = INITIAL_CONCENTRATION;
    }
    else if(is_stoich(str, p1, p2))
    {
        selectionType = STOICHIOMETRY;
    }
    else if (is_symbol(str, p1))
    {
        //This allows 'p1' to be set even for TIME.
        if (is_time(str))
        {
            selectionType = TIME;
        }
        else
        {
            selectionType = UNKNOWN_ELEMENT;
        }
    }
}

std::string rr::SelectionRecord::to_string() const
{
    std::string result;
    switch(selectionType)
    {
    case TIME:
        result = "time";
        break;
    case BOUNDARY_CONCENTRATION:
    case FLOATING_CONCENTRATION:
    case UNKNOWN_CONCENTRATION:
        result = "[" + p1 + "]";
        break;
    case FLOATING_CONCENTRATION_RATE:
        result = "[" + p1 + "]'";
        break;
    case FLOATING_AMOUNT_RATE:
    case GLOBAL_PARAMETER_RATE:
    case COMPARTMENT_RATE:
        result = p1 + "'";
        break;
    case COMPARTMENT:
    case GLOBAL_PARAMETER:
    case FLOATING_AMOUNT:
    case BOUNDARY_AMOUNT:
    case UNKNOWN_ELEMENT:
    case REACTION_RATE:
        result = p1;
        break;
    case ELASTICITY:
        result = "ec(" + p1 + ", " + p2 + ")";
        break;
    case UNSCALED_ELASTICITY:
        result = "uec(" + p1 + ", " + p2 + ")";
        break;
    case CONTROL:
        result = "cc(" + p1 + ", " + p2 + ")";
        break;
    case UNSCALED_CONTROL:
        result = "ucc(" + p1 + ", " + p2 + ")";
        break;
    case EIGENVALUE_COMPLEX:
        result = "eigen(" + p1 + ")";
        break;
    case EIGENVALUE_REAL:
        result = "eigenReal(" + p1 + ")";
        break;
    case EIGENVALUE_IMAG:
        result = "eigenImag(" + p1 + ")";
        break;
    case INITIAL_AMOUNT:
    case INITIAL_FLOATING_AMOUNT:
    case INITIAL_BOUNDARY_AMOUNT:
    case INITIAL_GLOBAL_PARAMETER:
    case INITIAL_COMPARTMENT:
        result = "init(" + p1 + ")";
        break;
    case INITIAL_CONCENTRATION:
    case INITIAL_FLOATING_CONCENTRATION:
    case INITIAL_BOUNDARY_CONCENTRATION:
        result = "init([" + p1 + "])";
        break;
    case STOICHIOMETRY:
        result = "stoich(" + p1 + ", " + p2 + ")";
        break;
    case UNKNOWN:
        result = "UNKNOWN";
        break;
    default:
        result = "ERROR";
        break;
    }
    return result;
}

}

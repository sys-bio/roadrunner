// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file Solver.h
* @author JKM
* @date 09/01/2015
* @copyright Apache License, Version 2.0
* @brief Contains the base class for RoadRunner solvers
**/

# ifndef RR_SOLVER_H_
# define RR_SOLVER_H_


#include "rrLogger.h"
#include "rrOSSpecifics.h"
#include "Dictionary.h"
#include "rrException.h"
#include "Registrable.h"

#include "tr1proxy/rr_memory.h"
#include "tr1proxy/rr_unordered_map.h"
#include <stdexcept>


namespace rr
{
    // fwd decl
    class ExecutableModel;

    /**
     * @author JKM
     * @brief Base class for all integrators and steady state solvers
     */
    class RR_DECLSPEC Solver : public Registrable
    {
    public:

        using Registrable::getName;
        using Registrable::getHint;
        using Registrable::getDescription;

        Solver() = default;

        explicit Solver(ExecutableModel* model);

        ~Solver() override = default;

        /**
         * @brief Update settings values
         * @details update the values of keys in @param inputSettings
         * with the values. Keys that are not in Solver are ignored
         */
         void updateSettings(Dictionary * inputSettings);

        /**
        * @author JKM
        * @brief Get a list of all settings for this solver
        */
        std::vector<std::string> getSettings() const;

        /**
         * @brief get settings for this solver
         * @return mapping of keys which are setting names
         * to values stored as Variants.
         * @see getSettings
         */
        std::unordered_map<std::string, Setting>& getSettingsMap();

        /**
        * @brief Reset all settings to their respective default values
        */
        virtual void resetSettings();

        /**
        * @author JKM, WBC
        * @brief Get the value of an integrator setting
        * @note Use one of the type-concrete versions like @ref getValueAsInt
        * to avoid type conversion gotchas
        */
        virtual Setting getValue(const std::string& key) const;

        /**
        * @author JKM
        * @brief Return true if this setting is supported by the integrator
        */
        virtual Setting hasValue(const std::string& key) const;

        /**
        * @author JKM
        * @brief Get the number of parameters
        */
        virtual size_t getNumParams() const;

        /**
        * @author JKM
        * @brief Get the name of the parameter at index n
        */
        virtual std::string getParamName(size_t n) const;

        /**
        * @author JKM
        * @brief Get the display name of the parameter at index n
        */
        virtual std::string getParamDisplayName(int n) const;

        /**
        * @author JKM
        * @brief Get the hint of the parameter at index n
        */
        virtual std::string getParamHint(int n) const;

        /**
        * @author JKM
        * @brief Get the description of the parameter at index n
        */
        virtual std::string getParamDesc(int n) const;

        /**
        * @author WBC, JKM
        * @brief Wrapper for @ref getValue which converts output to a specific type
        */
        virtual std::string getValueAsString(const std::string& key);

        virtual void setValue(const std::string& key, Setting value);

        /**
        * @author JKM
        * @brief Get the solver settings as a std::string
        */
        virtual std::string getSettingsRepr() const;

        /**
        * @author JKM
        * @brief Python dictionary-style std::string representation of settings
        */
        virtual std::string settingsPyDictRepr() const;

        /**
        * @author JKM
        * @brief Return a std::string representation of the solver
        */
        virtual std::string toString() const;

        /**
        * @author JKM
        * @brief Return std::string representation a la Python __repr__ method
        */
        virtual std::string toRepr() const;

        /**
        * @author WBC
        * @brief Gets the hint associated with a given key
        */
        const std::string& getDisplayName(const std::string& key) const;

        /**
        * @author WBC
        * @brief Gets the hint associated with a given key
        */
        const std::string& getHint(const std::string& key) const;

        /**
        * @author WBC
        * @brief Gets the description associated with a given key
        */
        const std::string& getDescription(const std::string& key) const;

        /**
        * @author WBC
        * @brief Gets the type associated with a given key
        */
        Setting::TypeId getType(const std::string& key) const;

        /**
        * @author JKM
        * @brief Called whenever a new model is loaded to allow integrator
        * to reset internal state
        */
        virtual void syncWithModel(ExecutableModel* m) = 0;


        /**
         * @brief returns the pointer to the ExecutableModel
         */
        virtual ExecutableModel *getModel() const;


        using SettingsList     =  std::vector<std::string> ;
        using SettingsMap      =  std::unordered_map<std::string, Setting> ;
        using DisplayNameMap   =  std::unordered_map<std::string, std::string> ;
        using HintMap          =  std::unordered_map<std::string, std::string> ;
        using DescriptionMap   =  std::unordered_map<std::string, std::string> ;

        SettingsList sorted_settings;
        SettingsMap settings;
        DisplayNameMap display_names_;
        HintMap hints;
        DescriptionMap descriptions;

    protected:
        /**
         * non-owning pointer to model
         */
        ExecutableModel* mModel = nullptr;

        void addSetting(const std::string& name, Setting val, std::string display_name, std::string hint, std::string description);

    };

}

# endif /* RR_INTEGRATOR_H_ */

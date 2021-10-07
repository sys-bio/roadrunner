// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file Integrator.cpp
* @author ETS, WBC, JKM
* @date Apr 25, 2014
* @copyright Apache License, Version 2.0
* @brief Contains the base class for RoadRunner integrators
**/

// == INCLUDES ================================================

#include "Solver.h"
#include "rrConfig.h"
#include <iomanip>
#include <utility>
#include <iostream>

// == CODE ====================================================

namespace rr
{
    Solver::Solver(ExecutableModel* model)
        : mModel(model){}

    void Solver::addSetting(const std::string& name, const Setting& val, const std::string& display_name, const std::string& hint, const std::string& description) {
        if (settings.find(name) == settings.end()) {
            // only add the new setting if its not already present
            sorted_settings.push_back(name);
            settings[name] = val;
            display_names_[name] = display_name;
            hints[name] = hint;
            descriptions[name] = description;
        }
    }

    void Solver::updateSettings(Dictionary * inputSettings)
    {
        if (!inputSettings)
            return;
        const std::vector<std::string>& thisSolversSettings = getSettings();
        for (const auto& setting: thisSolversSettings){
            if (inputSettings->hasKey(setting)){
                setValue(setting, inputSettings->getItem(setting));
            }
        }
    }

    std::unordered_map<std::string, Setting>& Solver::getSettingsMap()
    {
        return settings;
    }

    std::vector<std::string> Solver::getSettings() const
    {
        std::vector<std::string> keys;
        for (const auto & sorted_setting : sorted_settings)
        {
            keys.push_back(sorted_setting);
        }
        return keys;
    }

    void Solver::resetSettings()
    {
        sorted_settings.clear();
        settings.clear();
        hints.clear();
        descriptions.clear();
    }

    size_t Solver::getNumParams() const
    {
        if (sorted_settings.size() != settings.size())
            throw std::runtime_error("Setting count inconsistency");
        return settings.size();
    }

    std::string Solver::getParamName(size_t n) const
    {
        if (sorted_settings.size() != settings.size())
            throw std::runtime_error("Setting count inconsistency");
        return sorted_settings.at(n);
    }

    std::string Solver::getParamDisplayName(int n) const
    {
        return getDisplayName(getParamName(n));
    }

    std::string Solver::getParamHint(int n) const
    {
        return getHint(getParamName(n));
    }

    std::string Solver::getParamDesc(int n) const
    {
        return getDescription(getParamName(n));
    }

    Setting Solver::getValue(const std::string& key) const
    {
        auto option = settings.find(key);
        if (option == settings.end())
        {
            throw std::invalid_argument("Solver::getValue: invalid key: " + key);
        }
        return option->second;
    }

    Setting Solver::hasValue(const std::string& key) const
    {
        return Setting(settings.find(key) != settings.end());
    }


    std::string Solver::getValueAsString(const std::string& key)
    {
        return getValue(key).get<std::string>();
    }

    void Solver::setValue(const std::string& key, Setting value)
    {
        if (settings.find(key) ==  settings.end()){
            std::string name = getName();
            throw std::invalid_argument(name + " invalid key: " + key);
        }
        settings[key] =  std::move(value);
    }


    const std::string& Solver::getDisplayName(const std::string& key) const
    {
        auto option = Solver::display_names_.find(key);
        if (option == display_names_.end())
        {
            throw std::invalid_argument("invalid key: " + key);
        }
        return option->second;
    }

    const std::string& Solver::getHint(const std::string& key) const
    {
        auto option = Solver::hints.find(key);
        if (option == hints.end())
        {
            throw std::invalid_argument("invalid key: " + key);
        }
        return option->second;
    }

    const std::string& Solver::getDescription(const std::string& key) const
    {
        auto option = Solver::descriptions.find(key);
        if (option == descriptions.end())
        {
            throw std::invalid_argument("invalid key: " + key);
        }
        return option->second;
    }

    Setting::TypeId Solver::getType(const std::string& key) const
    {
        return getValue(key).type();
    }

    std::string Solver::getSettingsRepr() const
    {
        std::stringstream ss;
        for(size_t n=0; n < getNumParams(); ++n){
            ss << "    " << std::setw(20) << getParamName(n) << ": " << getValue(getParamName(n)).toString() << "\n";
        }
        return ss.str();
    }

    std::string Solver::settingsPyDictRepr() const
    {
        std::stringstream ss;
        for(size_t n=0; n < getNumParams(); ++n)
            ss << (n ? ", " : "") << "'" << getParamName(n) << "': " << getValue(getParamName(n)).toString();
        return ss.str();
    }

    std::string Solver::toString() const
    {
        std::stringstream ss;
        ss << "< roadrunner.Solver() >\n";
        ss << "  name: " << getName() << "\n";
        ss << "  settings:\n";
        ss << getSettingsRepr();
        return ss.str();
    }

    std::string Solver::toRepr() const
    {
        std::stringstream ss;
        ss << "< roadrunner.Solver() \"" << getName() << "\" " << settingsPyDictRepr() << " >\n";
        return ss.str();
    }

    ExecutableModel *Solver::getModel() const {
        if (!mModel){
            throw NullPointerException("Solver::getModel(): mModel pointer is null");
        }
        return mModel;
    }
}



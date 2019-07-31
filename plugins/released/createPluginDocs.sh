#! /usr/bin/bash

plugins="add_noise test_model chisquare levenberg_marquardt nelder_mead monte_carlo_bs auto2000"
#Create PDF and HTML docs for each plugin

thisFolder=`pwd`
for plugin in $plugins
do
    echo "Creating docs for Plugin $plugin"
    cd ./$plugin/docs
    echo "In folder: " `pwd`
    makeDocs.sh
    cd $thisFolder
done

echo "Done.."



#! /usr/bin/bash

scripts="./../../../../scripts"
plugin=monte_carlo_bs

$scripts/makePDFDocs.sh $plugin
$scripts/makeHTMLDocs.sh $plugin

$scripts/cleanDocsFolder.sh

#! /usr/bin/bash

scripts="./../../../../scripts"
plugin=levenberg_marquardt

$scripts/makePDFDocs.sh $plugin
$scripts/makeHTMLDocs.sh $plugin

$scripts/cleanDocsFolder.sh

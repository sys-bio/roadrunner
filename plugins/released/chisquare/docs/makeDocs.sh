#! /usr/bin/bash

scripts="./../../../../scripts"
plugin=chisquare

$scripts/makePDFDocs.sh $plugin
$scripts/makeHTMLDocs.sh $plugin

$scripts/cleanDocsFolder.sh

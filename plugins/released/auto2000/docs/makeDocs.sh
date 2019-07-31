#! /usr/bin/bash

scripts="./../../../../scripts"
plugin=auto2000

$scripts/makePDFDocs.sh $plugin
$scripts/makeHTMLDocs.sh $plugin

$scripts/cleanDocsFolder.sh

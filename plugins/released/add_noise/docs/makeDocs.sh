#! /usr/bin/bash

scripts="./../../../../scripts"
plugin=add_noise

$scripts/makePDFDocs.sh $plugin
$scripts/makeHTMLDocs.sh $plugin

$scripts/cleanDocsFolder.sh

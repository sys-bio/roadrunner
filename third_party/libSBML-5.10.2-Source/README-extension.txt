                        l i b S B M L   5.9.0
		   SBML Level 3 extension packages

This directory contains beta releases of libSBML extensions to
implement support for certain SBML Level 3 packages.

To integrate one of these packages into your copy of libSBML 5, simply
extract the package archive into the existing libSBML source directory.
Then, re-run CMake, and enable the package you just added using the
appropriate check-box in the list of options presented by CMake.  For
example, to enable the "groups" package, select ENABLE_GROUPS in the
CMake GUI.  Alternatively, you can run the command line version of
CMake with the flag -DENABLE_GROUPS=ON.

For more information about libSBML, please visit the libSBML website at
                   http://sbml.org/Software/libSBML


Thank you for your interest in libSBML!

The LibSBML Team.


     .-://///:`  .:/+++++/-`      .--.             `---`  `--
  -/++//:---:.`://+syyyssoo+`    ohhy`            /hhh.  -hy`
`/++/-`       ::/ohhyyssssoss-   ohhh+           .yhhh.  .hy`          
:++/.        `:::sysoo+++++oss.  ohoyh-         `ohoyh.  .hy`          
++//`        `--:/oo+///://+os:  oh//hs`        :hs.yh.  .hy`          
/+//.       `..--:////:--:/oos.  oh/`sh/       `yh-`yh.  .hy`          
`////:-.......---::://///++oo-   oh/ -hy.      +h+ `yh.  .hy`          
  .:///:::::--::::://///++oo:    oh/  +hs     -hy` `yh.  .hy`          
`::-``..--::::::://osyyysoooo.   oh/  `yh:   `sh:  `yh.  .hy`          
:o+/`      .:////oyhyyyyyyssss`  oh/   :hy`  /ho`  `yh.  .hy`          
/oo/        .///oyysoo+++oosyy-  oh/    oho .hh.   `yh.  .hy`          
.sso:       `+++oso+//////syyy`  oh/    .hh-oh/    `yh.  .hy`          
 :sss+-`   ./oooooo//:::+syyy.   oh/     /hhhs`    `yh.  -hy`          
  `/syssooossssssssssssyyyy/`    oh/      shh-     `yh.  -hhooooooooooo
    `-/+oso+/-.-:/osyyso/-`      -:.      .:-      `--`  `:::::::::::::

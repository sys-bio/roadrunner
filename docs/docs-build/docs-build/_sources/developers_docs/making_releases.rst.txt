Making roadrunner releases
-----------------------------

Most of building the roadrunner release is handled automatically
on the `azure pipelines ci platform <https://dev.azure.com/TheRoadrunnerProject/roadrunner/_build>`_.
In roadrunner we use both an experimental release, for new changes
to be tested by power users before they are released to the general public
in the release branch. The difference between an experimental and an
actual release, in terms of source code changes, is the
`EXPERIMENTAL <https://github.com/sys-bio/roadrunner/blob/develop/wrappers/Python/setup.py#L24>`_
flag in the Python setup.py.

To make a release:

    #. Push changes to `develop` branch in the form of a github pull request
    #. Push the changes to the `experimental-release`.
    #. If making an actual release, push changes to the `release` branch.

Each push to `develop`, `experimental-release` or `release` triggers a
new CI build with that branch. If passed, you may download the artefacts
as zip files by navigating to the `pipeline <https://dev.azure.com/TheRoadrunnerProject/roadrunner/_build?definitionId=8>`_,
clicking on your specific build.

Now there are two jobs left: uploading the zip files directly to the
github release, with a description on what the release contains
and uploading pip wheels to pypi. The latter is done by unzipping
each of the downloaded artefacts and copying the pip wheels
into a new directory. Once all pip wheels are together and in
a new directory you can run the command:

.. code-block:: bash

    twine upload *

to upload all of the pip wheels in the directory.

A caveat is regarding the linux pip wheel which must be built on
centos which is the common denominator for most modern linux distributions.
This is handled automatically by the CI system but one must take care
that the name of the pip wheel has `manylinux2014` as its
`platform tag <https://www.python.org/dev/peps/pep-0425/#id13>`_.

.. note::

    Do not try to upload pip wheels built on ubuntu to pip as
    this will not be accepted.













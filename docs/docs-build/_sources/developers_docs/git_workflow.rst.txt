Git Workflow
=============

In roadrunner we use a familiar and common github workflow that makes use of feature branches,
a develop branch (which is the default branch), an experimental release branch and a release
branch. All new features, bug fixes, small changes etc. are conducted on a new branch from the
develop branch.

.. code-block:: bash

    git checkout develop # make sure were on develop
    git checkout -b feature-branch

.. note::

    If you have accidentally made changes to the develop branch, move them to a feature
    branch without committing. Do ``git switch -c feature-branch``.

Now that you are in a new branch ``feature-branch`` any changes are completely isolated
from the main ``develop`` branch, thus minimizing inclusion of accidental
bugs into roadrunner. To push this new ``feature-branch` to github do

.. code-block:: bash

    # ensure changes are committed
    git add -A
    git commit -m "awesome new feature - tests pass"

    # push to github. The ``feature-branch`` does not yet exist on github (the remote) so we need to create it
    git push -u origin feature-branch

Whilst developing your feature on the ``feature-branch`` consider committing often, preferably at
useful milestones, such as when you complete a class or even a set of methods. Be sure to
use useful commit messages to aid yourself and others in understanding your commit.

Once complete - i.e. all your tests pass locally, you need to integrate these changes
back into the develop branch. In roadrunner, we have disabled direct pushes to the develop,
experimental-release and release branches, even for administrators. We do this because it is
very easy to accidentally incorporate changes that break roadrunner. Even when all tests pass
locally on your system, its possible that you've unintentionally broken something for another
operating system. For this reason, we force all changes to be pulled through our azure-pipelines
CI system before they can be merged.

To integrate your changes from ``feature-branch`` into develop, push your most recent changes
and then create a pull request using the github browser. In depth instructions for creating
a pull request can be found `here <https://opensource.com/article/19/7/create-pull-request-github>`_.
You should select ``feature-branch`` on the right and ``develop`` on the left in order to request
that the contents of ``feature-branch`` be merged into ``develop``.

This will automatically push your changes to azure-pipelines and begin building roadrunner on all
systems in the cloud. If something breaks, you will not be given the option to merge ``feature-branch``
into ``develop``. Instead you should follow the links to the roadrunner azure-pipelines interface
to find out what broke your build. Fix the problems locally and commit and push them, like before.
You do not need to do anything with the pull request, the changes will automatically be
pulled once again through the azure-pipeline CI system.

.. note::

    It is critical that your code is tested. If you do not test your code, then the azure pipelines
    build will not break even when roadrunner is broken. The integrity of roadrunner depends on
    your ability to test your code.


























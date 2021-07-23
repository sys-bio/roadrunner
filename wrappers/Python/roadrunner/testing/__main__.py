import os, sys, glob


testing_directory = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(testing_directory, "test_data")
roadrunner_directory = os.path.dirname(testing_directory)
rr_site_packages_dir = os.path.dirname(roadrunner_directory)

# add local site-packages dir to path for running tests
# without installing roadrunner.
sys.path.append(rr_site_packages_dir)

# some debugging messages
print(f"Executing __main__.py located at \"{__file__}\"")
print("Adding roadrunner site-packages to sys.path: ", rr_site_packages_dir)

import roadrunner
import tester
print(f"Using roadrunner at {os.path.abspath(roadrunner.__file__)}")
print(f"Running tester at {os.path.abspath(tester.__file__)}")

roadrunner.sigtrap()

tester.runTester(test_data_dir)








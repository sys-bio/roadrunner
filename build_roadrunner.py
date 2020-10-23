import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("dependencies", help="Path to install prefix of libroadrunner-deps", type=str)
args = parser.parse_args()


def do_check_call(commands: list, error_message=None):
    try:
        cmake_found = subprocess.check_call(commands)
    except subprocess.CalledProcessError as e:
        print(e.output)
        if error_message:
            print(error_message)
        exit(1)


BASE_DIRECTORY = os.getcwd()

# will error if cmake not available
do_check_call(["cmake", "--version"],
              "Make sure cmake is available and your environment variables are correctly configured to allow the 'cmake' command to be run from shell")

# clone repo
roadrunner_github = r"https://github.com/CiaranWelsh/roadrunner.git"
ROADRUNNER_DIR = os.path.join(BASE_DIRECTORY, "libroadrunner")
do_check_call(["git", "clone", roadrunner_github, ROADRUNNER_DIR])

# make build directory and cd
os.chdir(ROADRUNNER_DIR)
ROADRUNNER_BUILD_DIR = os.path.join(ROADRUNNER_DIR, "build")
os.makedirs(ROADRUNNER_BUILD_DIR)
os.chdir(ROADRUNNER_BUILD_DIR)

INSTALL_LOCATION = os.path.join(ROADRUNNER_DIR, "roadrunner-install")
# cmake command
do_check_call(
    ["cmake", f"-DCMAKE_INSTALL_PREFIX={INSTALL_LOCATION}", f"-DRR_DEPENDENCIES_INSTALL_PREFIX={args['dependencies']}",
     ROADRUNNER_DIR])

# build command
do_check_call(["cmake", "--build", ROADRUNNER_BUILD_DIR, "-j", "12"])

# install command
do_check_call(["cmake" "--build", "install"])

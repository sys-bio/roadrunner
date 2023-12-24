"""
Get LLVM from url.

Usage:

    python getLLVM.py <llvm_download_link> <llvm_install_prefix?

For example:

    python getLLVM.py https://github.com/sys-bio/llvm-13.x/releases/download/llvmorg-13.0.0/llvm13-ubuntu-gcc10-rel.tar.gz /mnt/d/roadrunner/roadrunner/llvm-download-test/ubuntu

"""


import glob
import argparse
import os
from os.path import join, splitext, isdir, isfile
import zipfile
import tarfile
import requests
import io
import shutil

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("llvm_download_link", type=str, help="Url to compressed llvm-13.x link to download")
parser.add_argument("llvm_install_prefix", type=str, help="Where to install llvm (directory that will contain bin, include, lib etc")
args = parser.parse_args()

# filename of downloaded zip, llvm-13.x-Darwin-Release.zip
compressed_llvm_filename = args.llvm_download_link.split('/')[-1]

# absolute path to compressed_llvm_filename
abs_compressed_llvm_filename = join(args.llvm_install_prefix, compressed_llvm_filename)

# e.g. ('llvm-13.x-Darwin-Release', '.zip')
downloaded_llvm_folder, ext = splitext(compressed_llvm_filename)
# when link is tar.gz, get rid of extra .tar
if ".tar" in downloaded_llvm_folder:
    downloaded_llvm_folder = downloaded_llvm_folder.replace(".tar", "")

# make install prefix if not exists
if not isdir(args.llvm_install_prefix):
    os.makedirs(args.llvm_install_prefix)

print("compressed_llvm_filename".ljust(20), compressed_llvm_filename)
print("abs_compressed_llvm_filename".ljust(20), abs_compressed_llvm_filename)
print("downloaded_llvm_folder".ljust(20), downloaded_llvm_folder)
print("args.llvm_download_link".ljust(20), args.llvm_download_link)
print("downloaded_llvm_folder, ext".ljust(20), downloaded_llvm_folder, ext)

# we expect a folder called bin in
abs_downloaded_llvm_folder = join(args.llvm_install_prefix, downloaded_llvm_folder)

# Don't download if we already have it
if not isdir(join(abs_downloaded_llvm_folder, "bin")):
    print("downloading llvm-13.x from {}".format(args.llvm_download_link))
    r = requests.get(args.llvm_download_link, stream=True)
    if ext == ".zip":
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(args.llvm_install_prefix)
    elif ext == ".gz":
        z = tarfile.open(fileobj=r.raw, mode="r|gz")
        z.extractall(args.llvm_install_prefix)
    else:
        raise ValueError("Unsupported extension")
else :
    print("Found existing llvm, not downloading llvm")

# move from unzip dir to llvm_install_prefix
folders = glob.glob(join(abs_downloaded_llvm_folder, "*"))
print(folders)
for f in folders:
    shutil.move(f, args.llvm_install_prefix)


if isfile(abs_compressed_llvm_filename):
    os.remove(abs_compressed_llvm_filename)






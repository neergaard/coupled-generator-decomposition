# import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
# def read(fname):
#    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="projects",
    version="0.1",
    author="Jesper Duemose Nielsen",
    author_email="jdue@dtu.dk",
    description="Tools for various projects.",
    # license = "BSD",
    keywords="EEG MEG",
    url="https://github.com/jdue/projects-python",
    packages=["projects"],
    # long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        #    "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        "mne",
        "numpy",
        "pandas",
        "pyvista",
        "scipy",
    ],
)

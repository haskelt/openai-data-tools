import setuptools
import glob

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openai_data_tools",
    version="0.22.0-alpha",
    author="Todd Haskell",
    author_email="todd@craggypeak.com",
    description="A set of classes for processing data using the OpenAI API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haskelt/openai-data-tools",
    license="GNU General Public License v3",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.6',
    install_requires=[
        'openai>=1.1.1',
        'numpy>=1.26.1'
    ]
)

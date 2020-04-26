import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ntropy",
    version="0.0.2",
    author="Ilia Zintchenko",
    author_email="ilia@ntropy.network",
    description="Run, train and benchmark machine learning models across data silos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ntropy-network/ntropy-client",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

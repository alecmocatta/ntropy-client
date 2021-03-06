import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'pytorch-dp',
    'requests',
    'torch',
]

setuptools.setup(
    name="ntropy",
    version="0.0.4",
    author="Ntropy Network Inc.",
    author_email="dev@ntropy.network",
    description="Train machine learning models on data across multiple data silos.",
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
    install_requires=requirements,
)

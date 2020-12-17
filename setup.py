from setuptools import setup, find_packages

setup(
    name="modelmiezelb",
    version="0.2.0",
    packages=find_packages(),
    author="Lukas Beddrich",
    author_email="lukas.beddrich@frm2.tum.de",
    description="package to model intermediate scattering functions for MIEZE measurements",
    long_description="file:README.md",
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=[
        "numpy",
        "matplotlib",
        "iminuit",
        "pytest",
        "scipy"
    ],
    python_requires=">=3.7.3",
    project_urls={"Source Code" : "https://github.com/LukasBeddrich/modelmiezelb"}
)
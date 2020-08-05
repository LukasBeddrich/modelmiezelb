from setuptools import setup, find_packages

setup(
    name="modelmiezelb",
    version="0.1.1",
    packages=find_packages(),
    author="Lukas Beddrich",
    author_email="lukas.beddrich@frm2.tum.de",
    description="package to model intermediate scattering functions for MIEZE measurements",
    install_requires=["numpy", "scipy", "skipi", "matplotlib", "pytest"]
    )
from setuptools import setup, find_packages

with open("README.md", "r") as f:
        long_description = f.read()
        
setup(
    name="SymbolicKinDyn",
    author="Underactuated Lab DFKI Robotics Innovation Center Bremen",
    maintainer="Hannah Isermann",
    maintainer_email="hannah.isermann@dfki.de",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url="",
    packages=find_packages(exclude="unit_testing"),
    setup_requires=["numpy"],
    install_requires=[
        "numpy",
        "sympy>=1.8",
        "urdfpy",
        "regex",
        "PyYAML",
        "pylatex"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Academic Usage',
        'Programming Language :: Python',
    ],
)
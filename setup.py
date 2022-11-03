from setuptools import setup, find_packages

setup(
    name="SymbolicKinDyn",
    author="Underactuated Lab DFKI Robotics Innovation Center Bremen",
    version="0.0.0",
    # url="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sympy>=1.8",
        "urdfpy",
        "regex"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Academic Usage',
        'Programming Language :: Python',
    ],
)
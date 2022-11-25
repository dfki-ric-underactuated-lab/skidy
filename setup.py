from setuptools import setup, find_packages

setup(
    name="SymbolicKinDyn",
    author="Underactuated Lab DFKI Robotics Innovation Center Bremen",
    version="0.0.0",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # url="",
    packages=find_packages(exclude="unit_testing"),
    setup_requires=["numpy"],
    install_requires=[
        "numpy",
        "sympy>=1.8",
        "urdfpy",
        "regex",
        "PyYAML"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Academic Usage',
        'Programming Language :: Python',
    ],
)
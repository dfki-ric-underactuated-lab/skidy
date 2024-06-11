from setuptools import setup, find_packages

with open("README.md", "r") as f:
        long_description = f.read()
        
setup(
    name="skidy",
    author="Underactuated Lab DFKI Robotics Innovation Center Bremen",
    maintainer="Hannah Isermann",
    maintainer_email="hannah.isermann@dfki.de",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url="",
    packages=find_packages("src",exclude="test"),
    package_dir={"": "src"},
    setup_requires=["numpy"],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "sympy>=1.8",
        "urdf_parser_py>=0.0.4",
        "regex",
        "PyYAML",
        "pylatex",
        "pydot",
    ],
    extras_require={"testing": ["cython","oct2py","kinpy","pin"],
                    "testing_required": ["kinpy"]},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Programming Language :: Python',
        "Operating System :: OS Independent",
    ],
)
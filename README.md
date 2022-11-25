# python symbolic kinematics and dynamics

Symbolic kinematics and dynamics model generation using Equations of Motion in closed form. 
This python file is almost a copy of the Matlab symbolic kinematics and dynamics generation tool.


## Install
The project requires the following packages:
- sympy (Version >= 1.8)
```
python3 -m pip install --upgrade sympy
```
- numpy
```
python3 -m pip install numpy
```
- urdfpy
```
python3 -m pip install urdfpy
```
- regex
```
python3 -m pip install regex
```
- PyYAML
```
python3 -m pip install PyYAML
```

To install all dependencies use:
```
python3 setup.py install
```
## Usage
The script kinematics generator contains one class `SymbolicKinDyn` which contains all needed functions.
This class has the attributes `gravity_vector`, `ee`, `A`, `B`, `X`, `Y` and `Mb`, which have to be set as described in https://git.hb.dfki.de/underactuated-robotics/symbolic_kinematics_and_dynamics/-/blob/master/documentation/main.pdf.

The in this documentation described example of a 2R kinematic chain robot is coded after the `if __name__ == "__main__:"` part in the kinematics_generator.py script.


If you have set all needed parameters (you have to either set `A` and `Y` or `B`, not all of them) the forward kinematics can be calculated with 
`closed_form_kinematics_body_fixed(q,dq,ddq)`.

The inverse dynamics is calculated using:
`closed_form_inv_dyn_body_fixed(q,dq,ddq)`.


### Code generation
There exists a code generation function, which can generate python, Matlab/Octave and C (C99) code. 

This can be done with:
`generateCode()`
See docstring for more options.

## Unit testing
To run the unit tests use:
```
python3 ./unit_testing/unit_testing.py
```

## Benchmarking
## Shortcomings
Currently, the expression simplification takes ages for higher dimension system with 3 or 4 rotational degrees of freedom. 

# python symbolic kinematics and dynamics

Symbolic kinematics and dynamics model generation using Equations of Motion in closed form. 
This python file is almost a copy of the matlab symbolic kinematics and dynamics generation tool.


# Requirements
Sympy (Version >= 1.8)
```
python3 -m pip install --upgrade sympy
```
urdfpy
```
python3 -m pip install urdfpy
```

# Usage
The script kinematics generator contains one class `SymbolicKinDyn` which contains all needed functions.
This class has the atributes `gravity_vector`, `ee`, `A`, `B`, `X`, `Y` and `Mb`, which have to be set as described in https://git.hb.dfki.de/underactuated-robotics/symbolic_kinematics_and_dynamics/-/blob/master/documentation/main.pdf.

The in this doku described example of a 2R kinematic chain robot is coded after the `if __name__ == "__main__:"` part in the kinematics_generator.py script.


If you have set all needed parameters (you have to eighter set `A` and `Y` or `B`, not all of them) the forward kinematics can be calculated with 
`closed_form_kinematics_body_fixed(q,dq,ddq)`
or 
`closed_form_kinematics_body_fixed_parallel(q,dq,ddq)`
for paralell expression simplification.

The inverse dynamics is calcualted using:
`closed_form_inv_dyn_body_fixed(q,dq,ddq)`
or 
`closed_form_inv_dyn_body_fixed_parallel(q,dq,ddq)`
for parallel expression simplificaiton.


## Code generation
There esists a code generation function, which can generate  python, Matlab/Octave and C (C99) code. 

This can be done with:
`generateCode()`
See docstring for more options.

# Shortcommings
Currently, the expression simplification takes ages for higher dimension system with 3 or 4 rotational degrees of freedom. 

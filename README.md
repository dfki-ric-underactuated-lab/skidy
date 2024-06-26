# skidy - symbolic kinematics and dynamics generator

- [skidy - symbolic kinematics and dynamics generator](#skidy---symbolic-kinematics-and-dynamics-generator)
  - [1. Installation](#1-installation)
    - [Install package from source](#install-package-from-source)
    - [Dependencies](#dependencies)
  - [2. Usage](#2-usage)
    - [2.1. YAML and JSON](#21-yaml-and-json)
      - [2.1.1. Create robot model as YAML file](#211-create-robot-model-as-yaml-file)
      - [2.1.2. Code generation using YAML](#212-code-generation-using-yaml)
    - [2.2. Python](#22-python)
    - [2.3. URDF](#23-urdf)
  - [3. Unit testing](#3-unit-testing)
  - [4. Benchmarking](#4-benchmarking)
  - [5. License](#5-license)

Symbolic kinematics and dynamics model generation using Equations of Motion in closed form.

## 1. Installation

```bash
python3 -m pip install skidy
```

For installing the test dependencies use:

```bash
python3 -m pip install skidy[testing]
```

### Install package from source

To install the package from source go to the root folder of the package:

```bash
cd /path/to/project
```

then use:

```bash
python3 -m pip install -e .
```

to install the package in editable mode or:

```bash
python3 -m pip install .
```

to install the package uneditable.

**Note:** The `.` is part of the command.  

### Dependencies

The project has the following dependencies:

- sympy (Version >= 1.8)

    ```bash
    python3 -m pip install --upgrade sympy
    ```

- numpy

    ```bash
    python3 -m pip install numpy
    ```

- urdf_parser_py (Version >= 0.0.4)

    ```bash
    python3 -m pip install --upgrade urdf_parser_py
    ```

- regex

    ```bash
    python3 -m pip install regex
    ```

- PyYAML

    ```bash
    python3 -m pip install PyYAML
    ```

- pylatex

    ```bash
    python3 -m pip install pylatex
    ```

- pydot

    ```bash
    python3 -m pip install pydot
    ```

For unit testing the following packages are additionally recommended:

- oct2py

    ```bash
    python3 -m pip install oct2py
    ```

    This requires a working octave installation on your system:
    Install on Ubuntu with:

    ```bash
    sudo apt install octave
    ```

- cython

    ```bash
    python3 -m pip install cython
    ```

- kinpy

    ```bash
    python3 -m pip install kinpy
    ```

- pinocchio

    ```bash
    python3 -m pip install pin
    ```

## 2. Usage

There are four ways to load your robot into the library:

1. using a *YAML* file
2. using a *JSON* file
3. directly in *python*
4. using *URDF*

For defining a robot with one of the first three options (YAML, JSON, python) the following parameters are required:

- 6D joint screw coordinates for any joint
- 4x4 body reference coordinates for any link
- 4x4 end-effector configuration w.r.t. last link body fixed frame in the chain

For generating the inverse dynamics the following two parameters are required:

- 3D gravity vector
- 6x6 Mass-inertia matrix for any link

Tree-like robot structures require the following graph description parameters additionally:

- parent link for any joint
- support graph for any joint
- (child links for any joint) -> currently not used

Using URDF the following three parameters are required:

- path to URDF file
- 4x4 end-effector configuration w.r.t. last link body fixed frame in the chain
- 3D gravity vector

### 2.1. YAML and JSON

**NOTE:** As *JSON* and *YAML* files represent the same data structure this documentation covers only *YAML* files. Use *JSON* accordingly and just replace `yaml` with `json` in all commands.

#### 2.1.1. Create robot model as YAML file

There is a function to generate a **template YAML file** in which it is easy to modify the parameters for your robot.
To generate your robot template use

```bash
python3 -m skidy --template [options] new_filename.yaml
```

or the python function `skidy.generate_template_yaml(path, structure)`.

For [options] the option `--structure` is highly recommended. There you can define which joint types to use in the template. E.g. use `--structure 'rrp'` for a robot which has two revolute joints followed by one prismatic joint.

The command `python3 -m skidy --template --structure 'rp' my_robot_template.yaml` creates the following output file:

```yaml
---
parent:
  - 0
  - 1

child:
  - [1]
  - []

support:
  - [1]
  - [1,2]

ee_parent: 2

gravity: [0,0,-g]

representation: spatial

joint_screw_coord:
  - type: revolute
    axis: [0,0,1]
    vec: [0,0,0]

  - type: prismatic
    axis: [0,0,1]

body_ref_config:
  - rotation:
      axis: [0,0,1]
      angle: 0
    translation: [0,0,0]

  - rotation:
      axis: [0,0,1]
      angle: 0
    translation: [0,0,0]

ee:
  - rotation:
      axis: [0,0,1]
      angle: 0
    translation: [0,0,0]

mass_inertia:
  - mass: m1
    frame: body
    inertia:
      Ixx: Ixx1
      Ixy: Ixy1
      Ixz: Ixz1
      Iyy: Iyy1
      Iyz: Iyz1
      Izz: Izz1
    com: [0,0,0]

  - mass: m2
    frame: body
    inertia:
      Ixx: Ixx2
      Ixy: Ixy2
      Ixz: Ixz2
      Iyy: Iyy2
      Iyz: Iyz2
      Izz: Izz2
    com: [0,0,0]

q: [q1,q2]
qd: [dq1,dq2]
q2d: [ddq1,ddq2]
WEE: [0,0,0,0,0,0]

assignments:
  g: 9.81

```

The code explained:

```yaml
parent:
  - 0
  - 1

child:
  - [1]
  - []

support:
  - [1]
  - [1,2]

ee_parent: 2
```

The parameters are generated to represent a serial robot by default. Modify parameters for tree-like structures. For serial robots these parameters are optional.

- **parent**: list of parent links for any joint. Use 0 for World.
- **child**: list of lists with child links for any link. Use empty list if no child is present.
- **support**: list of lists with all support links beginning with first link including current link for any link.
- **ee_parent**: parent joint index for end-effector link. Defaults to last link. For robots with more than one end-effector you can use a list of indices instead e.g.: `ee_parent: [1,2]`. Note, that this requires a list of end_effector transforms `ee`.

---

```yaml
gravity: [0,0,-g]
```

Gravity vector.
**Note:** you can always use symbolic variables instead of numeric values.

---

```yaml
representation: spatial
```

Define whether the representation of the joint screw coordinates and the body reference configuration is w.r.t. world frame (`representation: spatial`) or in body fixed coordinates (`representation: body_fixed`)

---

```yaml
joint_screw_coord:
  - type: revolute
    axis: [0,0,1]
    vec: [0,0,0]

  - type: prismatic
    axis: [0,0,1]
```

The joint screw coordinates can be defined either using the syntax which is used above, where `type` is the joint type (`revolute` or `prismatic`), `axis` is the joint axis and `vec` is a vector from the origin to the joint axis.
Alternatively, you can directly use the 6D joint screw vectors instead:

```yaml
joint_screw_coord:
  - [0,0,1,0,0,0]
  - [0,0,0,0,0,1]
```

---

```yaml
body_ref_config:
  - rotation:
      axis: [0,0,1]
      angle: 0
    translation: [0,0,0]

  - rotation:
      axis: [0,0,1]
      angle: 0
    translation: [0,0,0]
```

The body reference configuration is a list of SE(3) transformation matrices. To define them you have several options:

1. Write down the whole matrix e.g.:

    ```yaml
    body_ref_config:
      - [[cos(pi),-sin(pi),0, 0],
         [sin(pi), cos(pi),0, 0],
         [      0,       0,1,L1],
         [      0,       0,0, 1]]
    ```

2. Write rotation and translation separately:

    ```yaml
    body_ref_config:
      - rotation:
          [[1,0,0],
           [0,1,0],
           [0,0,1]]
        translation: [0,0,L1]
    ```

3. Use axis angle representation for rotation:
    -> See code above.

4. For zero rotations or translations it is possible to omit the option:

    ```yaml
    body_ref_config:
      - translation: [0,0,L1]
    ```

5. Use xyz_rpy coordinates to define Pose:

    ```yaml
    body_ref_config:
      - xyzrpy: [0, 0, L1, 0, pi/2, 0]
    ```

6. Use roll pitch yaw (rpy) euler angles to define rotation:

    ```yaml
    body_ref_config:
      - rotation:
          rpy: [0, pi/2, 0]
        translation: [0,0,L1]
    ```

7. Use quaternion [w,x,y,z] to define rotation:

    ```yaml
    body_ref_config:
      - rotation:
          Q: [1, 0, 0, 0]
        translation: [0,0,L1]
    ```

---

```yaml
ee:
  - rotation:
      axis: [0,0,1]
      angle: 0
    translation: [0,0,0]
```

End-effector representation w.r.t. last link body frame in the chain as SE(3) transformation matrix. Here you have the same syntax options as for the body reference configuration. For robots with more than one end-effector you can define more than one transform here. Note: in this case you have to define a list of parent joints in `ee_parent`.
If you have just one end-effector, you can omit the top level list (indicated by the trailing `-`).

---

```yaml
mass_inertia:
  - mass: m1
    frame: body
    inertia:
      Ixx: Ixx1
      Ixy: Ixy1
      Ixz: Ixz1
      Iyy: Iyy1
      Iyz: Iyz1
      Izz: Izz1
    com: [0,0,0]

  - mass: m2
    frame: body
    inertia:
      Ixx: Ixx2
      Ixy: Ixy2
      Ixz: Ixz2
      Iyy: Iyy2
      Iyz: Iyz2
      Izz: Izz2
    com: [0,0,0]
```

Mass-inertia matrices of all links. The keyword `frame` defines the frame where the `inertia` is defined in and accepts two values: `frame: body` and `frame: com`. When the inertia is w.r.t. the com frame, there is another keyword `rotation` available, which defines the rotation between the com frame and the body frame. The rotation can be defined like all other rotations above.

For the definition you have the following syntax options:

1. Write down whole matrix:

    ```yaml
    mass_inertia:
      - [[   Ixx1,    Ixy1,    Ixz1,       0, -cz1*m1,  cy1*m1],
         [   Ixy1,    Iyy1,    Iyz1,  cz1*m1,       0, -cx1*m1],
         [   Ixz1,    Iyz1,    Izz1, -cy1*m1,  cx1*m1,       0],
         [      0,  cz1*m1, -cy1*m1,      m1,       0,       0],
         [-cz1*m1,       0,  cx1*m1,       0,      m1,       0],
         [ cy1*m1, -cx1*m1,       0,       0,       0,      m1]]
    ```

2. Define mass, inertia and center of mass (com) separately:

    ```yaml
    mass_inertia:
      - mass: m1
        frame: body
        inertia:
          [[Ixx1,Ixy1,Ixz1],
           [Ixy1,Iyy1,Iyz1],
           [Ixz1,Iyz1,Izz1]]
        com: [cx1,cy1,cz1]

    ```

3. Only define the 6 independent inertia parameters:

    ```yaml
    mass_inertia:
      - mass: m1
        frame: body
        inertia: [Ixx1,Ixy1,Ixz1,Iyy1,Iyz1,Izz1]
        com: [cx1,cy1,cz1]

    ```

4. Define only necessary inertia parameters:

    ```yaml
    mass_inertia:
      - mass: m1
        frame: body
        inertia: 
          Ixx: Ixx1
          Iyy: Iyy1
          Izz: Izz1
        com: [cx1,cy1,cz1]
    ```

    Supports the parameters `Ixx`, `Ixy`, `Ixz`, `Iyy`, `Iyz` and `Izz`. All parameters default to 0.

5. Define only one value which is internally multiplied by an identity matrix:

    ```yaml
    mass_inertia:
      - mass: m1
        frame: com
        inertia: m1*L1**2
        com: [cx1,cy1,cz1]
    ```

6. Automatically generate symbols in inertia matrix:

    ```yaml
    mass_inertia:
      - mass: m1
        frame: body
        inertia:
          index: 1
          pointmass: False
        com: [0,0,0]

    ```

    Here the parameter index is appended to the names `Ixx` etc. and generates an inertia matrix following the naming scheme used in the examples above.
    With the parameter `pointmass: True` the resulting inertia matrix looks like this:

    ```yaml
    [[I1, 0, 0],
     [ 0,I1, 0],
     [ 0, 0,I1]]
    ```

---

```yaml
q: [q1,q2]
qd: [dq1,dq2]
q2d: [ddq1,ddq2]
WEE: [0,0,0,0,0,0]
```

Define symbolic generalized vectors (q, qd, q2d) and the time varying wrench on the end-effector link (WEE). Where the generalized vectors are list of length n and the wrench is a list of length 6 with symbolic values e.g.: `[Mx,My,Mz,Fx,Fy,Fz]`.
If the robot has more than one end-effector, one wrench per end_effector has to be defined in a list instead, e.g.:

```YAML
WEE: 
  - [Mx1,My1,Mz1,Fx1,Fy1,Fz1]
  - [Mx2,My2,Mz2,Fx2,Fy2,Fz2]
```

Add jerk and jounce here, to activate the calculation of the 1st and 2nd order derivatives of the Equation of Motion. Optionally, you can define the time derivatives of the external wrench `WDEE` and `W2DEE` here accordingly. E.g.:

```yaml
q: [q1,q2]
qd: [dq1,dq2]
q2d: [ddq1,ddq2]
q3d: [dddq1,dddq2]
q4d: [ddddq1,ddddq2]
WEE: [Mx,My,Mz,Fx,Fy,Fz]
WDEE: [dMx,dMy,dMz,dFx,dFy,dFz]
W2DEE: [ddMx,ddMy,ddMz,ddFx,ddFy,ddFz]
```

___

```yaml
assignments:
  g: 9.81
```

Under the optional `assignments` keyword you can create a dictionary which maps the symbolic values to numeric values. These numeric values will be used as default values for the parameters in the generated code.
#### 2.1.2. Code generation using YAML

To start the code generation process use:

```bash
python3 -m skidy [options] path/to/robot.yaml
```

In the options you have to specify what kind of code (python `-p`, Matlab `-m`, C++ `-c`, C `-C`, julia `-j`, cython `--cython`, latex `-l`) you'd like to generate and whether the equations should be simplified `-s`.

Use

```bash
python3 -m skidy -h
```

to get a description of all available options.

### 2.2. Python

As for YAML and JSON there is a function to auto-generate a **python template file** which makes it easier to define your own robot.
To generate your robot template use

```bash
python3 -m skidy --template [options] new_filename.py
```

or the python function `skidy.generate_template_python(path, structure)`.

For [options] the option `--structure` is highly recommended. There you can define which joint types to use in the template. E.g. use `--structure 'rrp'` for a robot which has two revolute joints followed by one prismatic joint.

The command `python3 -m skidy --template --structure 'rp' my_robot_template.py` creates the following output file:

```python
from skidy import (SymbolicKinDyn,
                   transformation_matrix,
                   mass_matrix_mixed_data,
                   joint_screw,
                   SO3Exp,
                   inertia_matrix,
                   generalized_vectors)
from skidy.symbols import g, pi
import sympy

# Define symbols:
m1, m2 = sympy.symbols('m1 m2', real=True, const=True)
Ixx1, Ixx2 = sympy.symbols('Ixx1 Ixx2', real=True, const=True)
Ixy1, Ixy2 = sympy.symbols('Ixy1 Ixy2', real=True, const=True)
Ixz1, Ixz2 = sympy.symbols('Ixz1 Ixz2', real=True, const=True)
Iyy1, Iyy2 = sympy.symbols('Iyy1 Iyy2', real=True, const=True)
Iyz1, Iyz2 = sympy.symbols('Iyz1 Iyz2', real=True, const=True)
Izz1, Izz2 = sympy.symbols('Izz1 Izz2', real=True, const=True)

# Define connectivity graph
parent = [0,
          1]

child = [[1],
         []]

support = [[1],
           [1,2]]

ee_parent = 2

# gravity vector
gravity = sympy.Matrix([0,0,g])

# representation of joint screw coordinates and body reference configurations
representation = 'spatial' # alternative: 'body_fixed'

# joint screw coordinates (6x1 sympy.Matrix per joint)
joint_screw_coord = []
joint_screw_coord.append(joint_screw(axis=[0,0,1], vec=[0,0,0], revolute=True))
joint_screw_coord.append(joint_screw(axis=[0,0,1], revolute=False))

# body reference configurations (4x4 SE3 Pose (sympy.Matrix) per link)
body_ref_config = []
body_ref_config.append(transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0]))
body_ref_config.append(transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0]))

# end-effector configuration w.r.t. last link body fixed frame in the chain (4x4 SE3 Pose (sympy.Matrix))
ee = transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0])

# mass_inertia parameters (6x6 sympy.Matrix per link)
Mb = []
Mb.append(mass_matrix_mixed_data(m1, inertia_matrix(Ixx1,Ixy1,Ixz1,Iyy1,Iyz1,Izz1), sympy.Matrix([0,0,0]), frame="body"))
Mb.append(mass_matrix_mixed_data(m2, inertia_matrix(Ixx2,Ixy2,Ixz2,Iyy2,Iyz2,Izz2), sympy.Matrix([0,0,0]), frame="body"))

q, qd, q2d = generalized_vectors(len(body_ref_config), startindex=1)
WEE = sympy.zeros(6,1)

skd = SymbolicKinDyn(gravity_vector=gravity,
                     ee=ee,
                     body_ref_config=body_ref_config,
                     joint_screw_coord=joint_screw_coord,
                     config_representation=representation,
                     Mb=Mb,
                     parent=parent,
                     child=child,
                     support=support,
                     ee_parent=ee_parent,
                     )

# run Calculations
skd.closed_form_kinematics_body_fixed(q, qd, q2d, simplify=True)
skd.closed_form_inv_dyn_body_fixed(q, qd, q2d, WEE=WEE, simplify=True)

# Generate Code
skd.generate_code(python=True, cpp=False, C=False, Matlab=False, latex=False,
                 folder="./generated_code", use_global_vars=True,
                 name="plant", project="Project")
```

The code explained:

```python
from skidy import (SymbolicKinDyn,
                   transformation_matrix,
                   mass_matrix_mixed_data,
                   joint_screw,
                   SO3Exp,
                   inertia_matrix,
                   generalized_vectors)
```

The class `SymbolicKinDyn` is the main object for calculating the kinematic and dynamic equations of your robot and generate the code.
Additionally, we import several helper functions for defining the matrices which are useful for the robot definition:

- `transformation_matrix`: Create SE(3) transformation matrix from SO(3) rotation and translation vector.
- `mass_matrix_mixed_data`: Create 6x6 mass-inertia matrix from mass, 3x3 inertia matrix and 3x1 center of mass vector.
- `joint_screw`: create 6x1 joint screw vector from joint axis and vector from origin to joint axis.
- `SO3Exp`: Exponential mapping of SO(3) to generate rotation matrix from rotation angle and rotation axis.
- `inertia_matrix`: generate 3x3 inertia matrix from 6 independent parameters (Ixx, Ixy, ...).
- `generalized_vectors`: generate symbolic generalized vectors q, qd and q2d of predefined length n.

```python
from skidy.symbols import g, pi
```

The package `skidy.symbols` includes the most common used symbolic variables, which can be used for defining your robot.

```python
import sympy
```

The whole library used sympy objects for all symbolic equations etc. Hence, we need `sympy` to create additional symbolic variables and matrices later.

---

```python
# Define symbols:
m1, m2 = sympy.symbols('m1 m2', real=True, const=True)
Ixx1, Ixx2 = sympy.symbols('Ixx1 Ixx2', real=True, const=True)
Ixy1, Ixy2 = sympy.symbols('Ixy1 Ixy2', real=True, const=True)
Ixz1, Ixz2 = sympy.symbols('Ixz1 Ixz2', real=True, const=True)
Iyy1, Iyy2 = sympy.symbols('Iyy1 Iyy2', real=True, const=True)
Iyz1, Iyz2 = sympy.symbols('Iyz1 Iyz2', real=True, const=True)
Izz1, Izz2 = sympy.symbols('Izz1 Izz2', real=True, const=True)
```

Create symbolic variables which can be used in the equations for the robot definition later. The most common symbols are also already present in the `skidy.symbols` package and may be imported from there instead.

---

```python
# Define connectivity graph
parent = [0,
          1]

child = [[1],
         []]

support = [[1],
           [1,2]]

ee_parent = 2
```

Connectivity graph of the robot. The parameters are generated to represent a serial robot by default. Modify parameters for tree-like structures. For serial robots these parameters are optional.

- **parent**: list of parent links for any joint. Use 0 for World.
- **child**: list of lists with child links for any link. Use empty list if no child is present.
- **support**: list of lists with all support links beginning with first link including current link for any link.
- **ee_parent**: parent joint index for end-effector link. Defaults to last link. For robots with more than one end-effector you can use a list of indices instead e.g.: `ee_parent = [1,2]`. Note, that this requires a list of end_effector transforms in `ee`.

---

```python
# gravity vector
gravity = sympy.Matrix([0,0,g])
```

Gravity vector as `sympy.Matrix`. Note that we can use symbolic variables here.

---

```python
# representation of joint screw coordinates and body reference configurations
representation = 'spatial' # alternative: 'body_fixed'
```

Define whether the representation of the joint screw coordinates and the body reference configuration is w.r.t. world frame (`representation = 'spatial'`) or in body fixed coordinates (`representation =  'body_fixed'`).

---

```python
# joint screw coordinates (6x1 sympy.Matrix per joint)
joint_screw_coord = []
joint_screw_coord.append(joint_screw(axis=[0,0,1], vec=[0,0,0], revolute=True))
joint_screw_coord.append(joint_screw(axis=[0,0,1], revolute=False))
```

The joint screw coordinates can be defined either using the syntax which is used above, where `axis` is the joint axis, `vec` is a vector from the origin to the joint axis and `revolute` has to be `True` for revolute joints and `False` for prismatic joints. Note that prismatic joints don't need the parameter `vec`.
Alternatively, you can directly use the 6D joint screw vectors instead:

```python
joint_screw_coord = []
joint_screw_coord.append(sympy.Matrix([0,0,1,0,0,0]))
joint_screw_coord.append(sympy.Matrix([0,0,0,0,0,1]))
```

---

```python
# body reference configurations (4x4 SE3 Pose (sympy.Matrix) per link)
body_ref_config = []
body_ref_config.append(transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0]))
body_ref_config.append(transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0]))
```

The body reference configuration is a list of SE(3) transformation matrices. To define them you have several options:

1. Write down the whole matrix e.g.:

    ```python
    body_ref_config.append(
        sympy.Matrix([[sympy.cos(pi/2),-sympy.sin(pi/2),0, 0],
                      [sympy.sin(pi/2), sympy.cos(pi/2),0, 0],
                      [              0,               0,1,L1],
                      [              0,               0,0, 1]])
    )
    ```

    Note: this example assumes you have defined the symbolic variable `L1` before.

2. Write rotation and translation separately:

    ```python
    body_ref_config.append(
        transformation_matrix(
            r=sympy.Matrix([[1,0,0],
                            [0,1,0],
                            [0,0,1]]),
            t=sympy.Matrix([0,0,L1])
        )
    )
    ```

3. Use axis angle representation for rotation:
   -> See code above.

4. For zero rotations or translations it is possible to omit the option:

    ```python
    body_ref_config.append(transformation_matrix(t=[0,0,0]))
    ```

5. Use xyz_rpy coordinates to define Pose:

    ```python
    body_ref_config.append(xyz_rpy_to_matrix([0, 0, L1, 0, pi/2, 0]))
    ```

    Note that you have to import the function using `from skidy import xyz_rpy_to_matrix`.

6. Use roll pitch yaw (rpy) euler angles to define rotation:

    ```python
    body_ref_config.append(
        transformation_matrix(
            r=rpy_to_matrix([0, pi/2, 0]),
            t=sympy.Matrix([0,0,L1])
        )
    )
    ```

    Note that you have to import the function using `from skidy import rpy_to_matrix`.

7. Use quaternion [w,x,y,z] to define rotation:

    ```python
    body_ref_config.append(
        transformation_matrix(
            r=quaternion_to_matrix([1,0,0,0]),
            t=sympy.Matrix([0,0,L1])
        )
    )
    ```

    Note that you have to import the function using `from skidy import quaternion_to_matrix`.

---

```python
# end-effector configuration w.r.t. last link body fixed frame in the chain (4x4 SE3 Pose (sympy.Matrix))
ee = transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0])
```

End-effector representation w.r.t. last link body frame in the chain as SE(3) transformation matrix. Here you have the same syntax options as for the body reference configuration.
For robots with more than one end-effector you can use a list of transforms instead. This requires a list of parent joint indices for the parameter `ee_parent`.

---

```python
# mass_inertia parameters (6x6 sympy.Matrix per link)
Mb = []
Mb.append(mass_matrix_mixed_data(m1, inertia_matrix(Ixx1,Ixy1,Ixz1,Iyy1,Iyz1,Izz1), sympy.Matrix([0,0,0]), frame='body'))
Mb.append(mass_matrix_mixed_data(m2, inertia_matrix(Ixx2,Ixy2,Ixz2,Iyy2,Iyz2,Izz2), sympy.Matrix([0,0,0]), frame='body'))
```

Mass-inertia matrices of all links. The mass inertia matrix has to be w.r.t. the body-fixed frame. But with the argument `frame = 'com'` you can define the inertia matrix w.r.t. the center of mass frame instead. In this case, there  is another argument `R` in the function `mass_matrix_mixed_data` which defines the rotation matrix of the COM frame to the body-fixed frame.

For the definition you have the following syntax options:

1. Write down whole matrix:

    ```python
    Mb.append(
        sympy.Matrix([[   Ixx1,    Ixy1,    Ixz1,       0, -cz1*m1,  cy1*m1],
                      [   Ixy1,    Iyy1,    Iyz1,  cz1*m1,       0, -cx1*m1],
                      [   Ixz1,    Iyz1,    Izz1, -cy1*m1,  cx1*m1,       0],
                      [      0,  cz1*m1, -cy1*m1,      m1,       0,       0],
                      [-cz1*m1,       0,  cx1*m1,       0,      m1,       0],
                      [ cy1*m1, -cx1*m1,       0,       0,       0,      m1]])
    )
    ```

2. Define mass, inertia matrix and center of mass separately:

    ```python
    Mb.append(
        mass_matrix_mixed_data(
            m1,
            sympy.Matrix([[Ixx1,Ixy1,Ixz1],
                          [Ixy1,Iyy1,Iyz1],
                          [Ixz1,Iyz1,Izz1]]),
            sympy.Matrix([cx1,cy1,cz1])
        )
    )
    ```

3. Only define the 6 independent inertia parameters:

    ```python
    Mb.append(
        mass_matrix_mixed_data(
            m1,
            inertia_matrix(Ixx1,Ixy1,Ixz1,Iyy1,Iyz1,Izz1),
            sympy.Matrix([cx1,cy1,cz1])
        )
    )
    ```

4. Automatically generate symbols in inertia matrix:

    ```python
    Mb.append(
        mass_matrix_mixed_data(
            m1,
            symbolic_inertia_matrix(index=1, pointmass=False),
            sympy.Matrix([cx1,cy1,cz1])
        )
    )
    ```

    where `symbolic_inertia_matrix(index=1, pointmass=False)` auto generates the variables `Ixx1`, `Ixy1`, etc. and creates a `sympy.Matrix` from it.
    With the parameter `pointmass=True` the resulting inertia matrix looks like this instead:

    ```python
    sympy.Matrix([[I1, 0, 0],
                  [ 0,I1, 0],
                  [ 0, 0,I1]])
    ```

    Note that you have to import the function using `from skidy import symbolic_inertia_matrix`.

---

```python
q, qd, q2d = generalized_vectors(len(body_ref_config), startindex=1)
```

Generate the generalized vectors (joint positions `q`, joint velocities `qd` and joint accelerations `q2d`). The symbols are auto generated starting at index `startindex`. The degrees of freedom in this case are taken from the length of `body_ref_config`.

The equivalent code would be:

```python
q1, q2 = sympy.symbols("q1 d2", real=True, constant=False)
dq1, dq2 = sympy.symbols("dq1 dd2", real=True, constant=False)
ddq1, ddq2 = sympy.symbols("ddq1 ddq2", real=True, constant=False)
q = sympy.Matrix([q1,q2])
qd = sympy.Matrix([dq1,dq2])
q2d = sympy.Matrix([ddq1,ddq2])
```

To generate jerk and jounce as well, use:

```python
q, qd, q2d, q3d, q4d = generalized_vectors(len(body_ref_config), startindex=1, derivatives=True)
```

---

```python
WEE = sympy.zeros(6,1)
```

Define the time varying wrench on the end-effector link. Should be a (6,1) sympy.Matrix with symbolic variables, e.g.: `sympy.Matrix([Mx,My,Mz,Fx,Fy,Fz])`.

Optionally, you can define the time derivatives `WDEE` and `W2DEE` here too. E.g.:

```python
WEE = sympy.Matrix([Mx,My,Mz,Fx,Fy,Fz])
WDEE = sympy.Matrix([dMx,dMy,dMz,dFx,dFy,dFz])
W2DEE = sympy.Matrix([ddMx,ddMy,ddMz,ddFx,ddFy,ddFz])
```

If your robot has more than one end-effector, you have to use one external wrench per end-effector in a list, e.g.:

```python
WEE = [sympy.Matrix([Mx1,My1,Mz1,Fx1,Fy1,Fz1]),
       sympy.Matrix([Mx2,My2,Mz2,Fx2,Fy2,Fz2])]
```

---

```python
skd = SymbolicKinDyn(gravity_vector=gravity,
                     ee=ee,
                     body_ref_config=body_ref_config,
                     joint_screw_coord=joint_screw_coord,
                     config_representation=representation,
                     Mb=Mb,
                     parent=parent,
                     child=child,
                     support=support,
                     )
```

Initialize class with all defined parameters.

---

```python
# run Calculations
skd.closed_form_kinematics_body_fixed(q, qd, q2d, simplify=True)
skd.closed_form_inv_dyn_body_fixed(q, qd, q2d, WEE=WEE, simplify=True)
```

Generate forward kinematics and inverse dynamics equations. Both functions share the following arguments:

- simplify: generated expressions are simplified. Note that the simplification takes a lot of time for robots with more than 2 revolute joints in a chain.
- cse: Use common subexpression elimination to shorten equations. Note that the equations are not human-readable afterwards.
- parallel: use parallel computation.

`skd.closed_form_kinematics_body_fixed` generates the following equations and saves them as class parameters:

- body_acceleration
- body_acceleration_ee
- body_jacobian_matrix
- body_jacobian_matrix_dot
- body_jacobian_matrix_ee
- body_jacobian_matrix_ee_dot
- body_twist_ee
- forward_kinematics
- hybrid_acceleration
- hybrid_acceleration_ee
- hybrid_jacobian_matrix
- hybrid_jacobian_matrix_dot
- hybrid_jacobian_matrix_ee
- hybrid_jacobian_matrix_ee_dot
- hybrid_twist_ee

and `skd.closed_form_inv_dyn_body_fixed` generates the following equations and saves them as class parameters:

- coriolis_centrifugal_matrix
- generalized_mass_inertia_matrix
- gravity_vector
- inverse_dynamics

with provided generalized jerk and jounce vectors, `skd.closed_form_inv_dyn_body_fixed` additionally generates the following derivatives of the equation of motion:

- coriolis_centrifugal_matrix_dot
- generalized_mass_inertia_matrix_dot
- gravity_vector_dot
- inverse_dynamics_dot
- coriolis_centrifugal_matrix_ddot
- generalized_mass_inertia_matrix_ddot
- gravity_vector_ddot
- inverse_dynamics_ddot

`skd.closed_form_inv_dyn_body_fixed` takes the wrench `WEE` (6x1 sympy.Matrix) (tau,F) on the end-effector link and its time derivatives `WDEE` and `W2DEE` as optional additional arguments.

To calculate the 1st and 2nd order time derivative of the EOM use e.g.:

```python
skd.closed_form_inv_dyn_body_fixed(q, qd, q2d, q3d, q4d, WEE, WDEE, W2DEE, simplify=True)
```

---

```python
# Generate Code
skd.generate_code(python=True, cpp=False, C=False, Matlab=False, latex=False,
                 folder="./generated_code", use_global_vars=True,
                 name="plant", project="Project")
```

Generate Python, Matlab, C++, C (C99), Julia, Cython and/or LaTeX code from the generated equations.
Note that this can take some time, especially for non-simplified equations and complex robots.

### 2.3. URDF

There are two ways to use URDF files. 
First, you can convert your URDF file to YAML or JSON using:

```bash
skidy --convert my_robot.urdf my_robot.yaml
```

Here, you can add end-effector transforms, and modify whatever you want.

Second, you can load your URDF using python. There is a function to generate a template python file, which loads your URDF. In the python file it is necessary to define:

1. the URDF path
2. the gravity vector
3. end-effector configuration w.r.t. last link body fixed frame in the chain

To generate the python template file use:

```bash
python -m skidy --template --urdf my_urdf_template.py
```

or the python function `skidy.generate_template_python(path, urdf=True)`.

This generates the following output:

```python
from skidy import (SymbolicKinDyn,
                   transformation_matrix,
                   SO3Exp,
                   generalized_vectors)
from skidy.symbols import g, pi
import sympy

# Define symbols:
lee = sympy.symbols('lee', real=True, const=True)

urdfpath = '/path/to/robot.urdf' # TODO: change me!

# gravity vector
gravity = sympy.Matrix([0,0,g])

# end-effector configuration w.r.t. last link body fixed frame in the chain
ee = transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0])

skd = SymbolicKinDyn(gravity_vector=gravity,
                     ee=ee,
                     )

skd.load_from_urdf(path = urdfpath,
                   symbolic=False, 
                   cse=False, 
                   simplify_numbers=True,  
                   tolerance=0.0001, 
                   max_denominator=8, 
                   )

q, qd, q2d = generalized_vectors(len(skd.body_ref_config), startindex=1)
WEE = sympy.zeros(6,1)

# run Calculations
skd.closed_form_kinematics_body_fixed(q, qd, q2d, simplify=True)
skd.closed_form_inv_dyn_body_fixed(q, qd, q2d, WEE=WEE, simplify=True)

# Generate Code
skd.generate_code(python=True, cpp=False, C=False, Matlab=False, latex=False,
                 folder="./generated_code", use_global_vars=True,
                 name="plant", project="Project")
```

The code explained:

```python
from skidy import (SymbolicKinDyn,
                   transformation_matrix,
                   SO3Exp,
                   generalized_vectors)
```

The class `SymbolicKinDyn` is the main object for calculating the kinematic and dynamic equations of your robot and generate the code.
Additionally, we import several helper functions for defining the matrices which are useful for the robot definition:

- `transformation_matrix`: Create SE(3) transformation matrix from SO(3) rotation and translation vector.
- `SO3Exp`: Exponential mapping of SO(3) to generate rotation matrix from rotation angle and rotation axis.
- `generalized_vectors`: generate symbolic generalized vectors q, qd and q2d of predefined length n.

```python
from skidy.symbols import g, pi
```

The package `skidy.symbols` includes the most common used symbolic variables, which can be used for defining your robot.

```python
import sympy
```

The whole library used sympy objects for all symbolic equations etc. Hence, we need `sympy` to create additional symbolic variables and matrices later.

---

```python
# Define symbols:
lee = sympy.symbols('lee', real=True, const=True)
```

Create symbolic variables which can be used in the equations for the robot definition later. The most common symbols are also already present in the `skidy.symbols` package and may be imported from there instead.

---

```python
urdfpath = '/path/to/robot.urdf' # TODO: change me!
```

Enter the path to your URDF file here.

---

```python
# gravity vector
gravity = sympy.Matrix([0,0,g])
```

Gravity vector as `sympy.Matrix`. Note that we can use symbolic variables here.

---

```python
# end-effector configuration w.r.t. last link body fixed frame in the chain
ee = transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0])
```

End-effector representation w.r.t. last link body frame in the chain as SE(3) transformation matrix. Look up the chapter [Python](#22-python) for all available syntax options.

---

```python
skd = SymbolicKinDyn(gravity_vector=gravity,
                     ee=ee,
                     )
```

Initialize class with the two defined parameters.

---

```python
skd.load_from_urdf(path = urdfpath,
                   symbolic=False, 
                   cse=False, 
                   simplify_numbers=True,  
                   tolerance=0.0001, 
                   max_denominator=8, 
                   )
```

Load the URDF file. Here you can specify the following options:

1. `symbolic`: symbolify values in urdf file (bool).
2. `cse`: use common subexpression elimination to shorten equations (bool).
3. `simplify_numbers`: round numbers if close to common fractions like 1/2 etc and replace eg 3.1416 by pi (bool).
4. `tolerance`: tolerance for simplify numbers.
5. `max_denominator`: define max denominator for simplify numbers to avoid simplification to something like 13/153. Use 0 to deactivate.

---

```python
q, qd, q2d = generalized_vectors(len(skd.body_ref_config), startindex=1)
```

Generate the generalized vectors (joint positions `q`, joint velocities `qd` and joint accelerations `q2d`). The symbols are auto generated starting at index `startindex`. The degrees of freedom in this case are taken from the length of the parameter `skd.body_ref_config`, which was generated by the function `load_from_urdf`.

---

```python
WEE = sympy.zeros(6,1)
```

Define the time varying wrench on the end-effector link. Should be a (6,1) sympy.Matrix with symbolic variables.

---

```python
# run Calculations
skd.closed_form_kinematics_body_fixed(q, qd, q2d, simplify=True)
skd.closed_form_inv_dyn_body_fixed(q, qd, q2d, WEE=WEE, simplify=True)
```

Generate forward kinematics and inverse dynamics equations. See chapter [Python](#22-python) for more information.

---

```python
# Generate Code
skd.generate_code(python=True, cpp=False, C=False, Matlab=False, latex=False,
                 folder="./generated_code", use_global_vars=True,
                 name="plant", project="Project")
```

Generate Python, Matlab, C++, C (C99), Julia, Cython and/or LaTeX code from the generated equations.
Note that this can take time, especially for non-simplified equations and complex robots.

## 3. Unit testing

To run the unit tests use:

```bash
python3 ./test/unit_testing.py
```

## 4. Benchmarking

For benchmarking the project the script `benchmarking/benchmarking.py` was used. This script loads 4 robots with increasing complexity (1 to 4 revolute joint in a chain with planar task space) and takes the execution time of the functions `closed_form_kinematics_body_fixed()`, `closed_form_inv_dyn_body_fixed()` and `generate_code()`. Additionally, the arguments `parallel`, `simplify` and `cse` have been altered.
The results are shown in the following table:

arguments | parallel | serial
:---------|:----------:|:-----------:
simplify|![1 dof: 0.65 s; 2 dof: 2.14 s; 3 dof: 8.77 s; 4 dof: 44.04 s](/benchmarking/parallel_with_simplification_without_cse.png) | ![1 dof: 0.16 s; 2 dof: 2.38 s; 3 dof: 13.06 s; 4 dof: 60.68 s](/benchmarking/serial_with_simplification_without_cse.png)
simplify + cse|![1 dof: 0.56 s; 2 dof: 2.22 s; 3 dof: 12.86 s; 4 dof: 84.28 s](/benchmarking/parallel_with_simplification_with_cse.png) |![1 dof: 0.22 s; 2 dof: 2.78 s; 3 dof: 18.49 s; 4 dof: 113.30 s](/benchmarking/serial_with_simplification_with_cse.png)
no simplify|![1 dof: 0.74 s; 2 dof: 2.65 s; 3 dof: 11.29 s; 4 dof: 47.14 s](/benchmarking/parallel_without_simplification_without_cse.png) | ![1 dof: 0.08 s; 2 dof: 1.57 s; 3 dof: 9.51 s; 4 dof: 44.57 s](/benchmarking/serial_without_simplification_without_cse.png)
no simplify + cse|![1 dof: 0.80 s; 2 dof: 4.50 s; 3 dof: 29.13 s; 4 dof: 161.01 s](/benchmarking/parallel_without_simplification_with_cse.png) |![1 dof: 0.08 s; 2 dof: 4.49 s; 3 dof: 37.74 s; 4 dof: 201.11 s](/benchmarking/serial_without_simplification_with_cse.png)

## 5. License

This work has been released under the BSD 3-Clause License. Details and terms of use are specified in the LICENSE file within this repository. Note that we do not publish third-party software, hence software packages from other developers are released under their very own terms and conditions. If you install third-party software packages along with this repo ensure that you follow each individual license agreement.

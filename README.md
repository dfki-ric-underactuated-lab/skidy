# python symbolic kinematics and dynamics

Symbolic kinematics and dynamics model generation using Equations of Motion in closed form.
This python file is almost a copy of the Matlab symbolic kinematics and dynamics generation tool.

## Install

The project requires the following packages:

- sympy (Version >= 1.8)

    ```bash
    python3 -m pip install --upgrade sympy
    ```

- numpy

    ```bash
    python3 -m pip install numpy
    ```

- urdfpy

    ```bash
    python3 -m pip install urdfpy
    ```

- regex

    ```bash
    python3 -m pip install regex
    ```

- PyYAML

    ```bash
    python3 -m pip install PyYAML
    ```

To install all dependencies use:

```bash
python3 setup.py install
```

<!-- ## Usage

The script kinematics generator contains one class `SymbolicKinDyn` which contains all needed functions.
This class has the attributes `gravity_vector`, `ee`, `A`, `B`, `X`, `Y` and `Mb`, which have to be set as described in <https://git.hb.dfki.de/underactuated-robotics/symbolic_kinematics_and_dynamics/-/blob/master/documentation/main.pdf>.

The in this documentation described example of a 2R kinematic chain robot is coded after the `if __name__ == "__main__:"` part in the kinematics_generator.py script.

If you have set all needed parameters (you have to either set `A` and `Y` or `B`, not all of them) the forward kinematics can be calculated with 
`closed_form_kinematics_body_fixed(q,dq,ddq)`.

The inverse dynamics is calculated using:
`closed_form_inv_dyn_body_fixed(q,dq,ddq)`. -->

## Usage

### Robot definition

There are four ways to load your robot into the library:

1. using a *YAML* file
2. using a *JSON* file
3. directly in *python*
4. using *URDF*

For defining a robot the following parameters are required:

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

#### YAML and JSON

**NOTE:** As *JSON* and *YAML* files represent the same data structure this documentation covers only *YAML* files. Use *JSON* accordingly and just replace `yaml` with `json` in all commands.

There is a function to generate a template YAML file in which it is easy to modify the parameters for your robot. As JSON and YAML files represent the same datastructure this documentation covers only YAML files. Use JSON accordingly.
To generate your robot template use

```bash
python3 analyze_my_robot.py --please [options] new_filename.yaml
```

or the python function `KinematicsGenerator.generate_template_yaml(path, structure)`.

For [options] the option `--structure` is highly recommended. There you can define which joint types to use in the template. E.g. use `--structure 'rrp'` for a robot which has two revolute joints followed by one prismatic joint.

The command `python3 analyze_my_robot.py --please --structure 'rp' my_robot.yaml` creates the following output file:

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

gravity: [0,0,g]

representation: spacial

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
  rotation:
    axis: [0,0,1]
    angle: 0
  translation: [0,0,0]

mass_inertia:
  - mass: m1
    inertia:
      Ixx: Ixx1
      Ixy: Ixy1
      Ixz: Ixz1
      Iyy: Iyy1
      Iyz: Iyz1
      Izz: Izz1
    com: [0,0,0]

  - mass: m2
    inertia:
      Ixx: Ixx2
      Ixy: Ixy2
      Ixz: Ixz2
      Iyy: Iyy2
      Iyz: Iyz2
      Izz: Izz2
    com: [0,0,0]
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
```

The parameters are generated to represent a serial robot by default. Modify parameters for tree-like structures. For serial robots these parameters are optional.

- **parent**: list of parent links for any joint. Use 0 for World.
- **child**: list of lists with child links for any link. Use empty list if no child is present.
- **support**: list of lists with all support links beginning with first link including current link for any link.

---

```yaml
gravity: [0,0,g]
```

Gravity vector.
**Note:** you can always use symbolic variables instead of numeric values.

---

```yaml
representation: spacial
```

Define whether the representation of the joint screw coordinates and the body reference configuration is w.r.t. word frame (`representation: spacial`) or in body fixed coordinates (`representation: body_fixed`)

---

```yaml
joint_screw_coord:
  - type: revolute
    axis: [0,0,1]
    vec: [0,0,0]

  - type: prismatic
    axis: [0,0,1]
```

The joint screw coordinates can be defined eighter using the syntax which is used above, where `type` is the joint type (`revolute` or `prismatic`), `axis` is the joint axis and `vec` is a vector from the origin to the joint axis.
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

---

```yaml
ee:
  rotation:
    axis: [0,0,1]
    angle: 0
  translation: [0,0,0]
```

End-effector representation wrt last link body frame in the chain as SE(3) transformation matrix. Here you have the same syntax options as for the body reference configuration. **Note** there is no trailing `-` as there is only one pose to be defined.

---

```yaml
mass_inertia:
  - mass: m1
    inertia:
      Ixx: Ixx1
      Ixy: Ixy1
      Ixz: Ixz1
      Iyy: Iyy1
      Iyz: Iyz1
      Izz: Izz1
    com: [0,0,0]

  - mass: m2
    inertia:
      Ixx: Ixx2
      Ixy: Ixy2
      Ixz: Ixz2
      Iyy: Iyy2
      Iyz: Iyz2
      Izz: Izz2
    com: [0,0,0]
```

Mass-inertia matrices of all links. For the definition you have the following syntax options:

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
        inertia: [Ixx1,Ixy1,Ixz1,Iyy1,Iyz1,Izz1]
        com: [cx1,cy1,cz1]

    ```

4. Define only necessary inertia parameters:

    ```yaml
    mass_inertia:
      - mass: m1
        inertia: 
          Ixx: Ixx1
          Iyy: Iyy1
          Izz: Izz1
        com: [cx1,cy1,cz1]

    ```

    Supports the parameters `Ixx`, `Ixy`, `Ixz`, `Iyy`, `Iyz` and `Izz`. All parameters default to 0.

5. Automatically generate symbols in inertia matrix:

    ```yaml
    mass_inertia:
      - mass: m1
        inertia:
          index: 1
          pointmass: False
        com: [0,0,0]

    ```

    Here the parameter index is appended to the names `Ixx` etc and generates an inertia matrix following the naming scheme used in the examples above.
    With the parameter `pointmass: True` the resulting inertia matrix looks like this:

    ```yaml
    [[I1, 0, 0],
     [ 0,I1, 0],
     [ 0, 0,I1]]
    ```

### Generate equations and code generation

### Code generation

There exists a code generation function, which can generate python, Matlab/Octave and C (C99) code.

This can be done with:
`generateCode()`
See docstring for more options.

## Unit testing

To run the unit tests use:

```bash
python3 ./unit_testing/unit_testing.py
```

## Benchmarking

## Shortcomings

Currently, the expression simplification takes ages for higher dimension system with 3 or 4 rotational degrees of freedom.

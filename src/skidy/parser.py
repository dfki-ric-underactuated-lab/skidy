from __future__ import annotations

import yaml
import json
from typing import Iterable, Optional
from skidy.kinematics_generator import SymbolicKinDyn
from skidy.matrices import (joint_screw, 
                            symbolic_inertia_matrix, mass_matrix_mixed_data, 
                            SO3Exp, transformation_matrix, inertia_matrix,
                            quaternion_to_matrix, xyz_rpy_to_matrix,
                            rpy_to_matrix)
from sympy import Matrix, Identity, parse_expr, Expr, zeros
import regex

def robot_from_yaml(path: str) -> SymbolicKinDyn:
    """Parse yaml robot description and return SymbolicKinDyn Object.

    Args:
        path (str): Path to yaml file.

    Returns:
        skidy.SymbolicKinDyn object.
    """
    with open(path, "r") as stream:
        y = yaml.safe_load(stream)
    return dict_parser(y)

def robot_from_json(path: str) -> SymbolicKinDyn:
    """Parse json robot description and return SymbolicKinDyn Object.

    Args:
        path (str): Path to json file.

    Returns:
        skidy.SymbolicKinDyn object.
    """
    with open(path, "r") as stream:
        y = json.load(stream)
    return dict_parser(y)

def parse_hierarchical_expr(x: list | dict | str, 
                            include_keys:Iterable={}, 
                            exclude_keys:Iterable={}) -> list | dict | Expr:
    """Convert strings in data structure (list or dict) to sympy 
    expressions.

    Args:
        x (list | dict | str): 
            Hierarchical structure which might contain symbolic 
            expression as string. 
        include_keys (Iterable): 
            Only convert strings with listed dict_keys. 
            Defaults to {}.
        exclude_keys (Iterable): 
            Don't convert strings with listed dict_keys. 
            Defaults to {}.
            
    Returns:
        list | dict | sympy.Expr: 
            Same data structure with strings converted to sympy.Expr. 
    """
    if type(x) in {dict, list}:
        for i in x if type(x) is dict else range(len(x)):
            if type(x[i]) in {list,dict}:
                x[i] = parse_hierarchical_expr(x[i],include_keys,exclude_keys)
            elif type(x[i]) is str:
                if (type(x) is list 
                    or (include_keys and i in include_keys) 
                    or (not include_keys and i not in exclude_keys)):
                    x[i] = parse_expr(x[i])
    elif type(x) is str:
        x = parse_expr(x)
    return x

def dict_parser(d: dict) -> SymbolicKinDyn:
    """Parse dict to SymbolicKinDyn object.

    Args:
        d (dict): Dictionary containing robot description.

    Raises:
        KeyError: Entry not found.
        ValueError: Unexpected entry.

    Returns:
        skidy.SymbolicKinDyn object.
    """
    d = parse_hierarchical_expr(d, include_keys={"mass", "index", 
                                                 "Ixx", "Ixy", "Ixz", 
                                                 "Iyy", "Ixz", "Izz", 
                                                 "q", "qd", "q2d", "q3d", "q4d",
                                                 "WEE", "WDEE", "W2DEE",
                                                 "inertia"})
    
    config_representation = d["representation"] if "representation" in d else "spatial"
    
    parent = d["parent"] if "parent" in d else []
    child = d["child"] if "child" in d else []
    support = d["support"] if "support" in d else []
    ee_parent = d["ee_parent"] if "ee_parent" in d else None
    
    try:
        gravity = d["gravity"] if "gravity" in d else d["gravity_vector"]
    except KeyError:
        gravity = None
    
    assert("joint_screw_coord" in d)
    joint_screw_coord = []
    for js in d["joint_screw_coord"]:
        if type(js) is list and len(js) == 6:
            joint_screw_coord.append(Matrix(js))
        elif type(js) is dict:
            if js["type"] == "prismatic":
                joint_screw_coord.append(
                    joint_screw(js["axis"],revolute=0))
            elif js["type"] == "revolute":
                joint_screw_coord.append(
                    joint_screw(
                        js["axis"],
                        js["vec"] if "vec" in js else [0,0,0],
                        revolute=1))
            else:
                raise ValueError(f"joint type {js['type']} not supported.")
        else:
            raise ValueError("joint screw coordinates corrupted.") 

    assert("body_ref_config" in d)
    
    body_ref_config = []
    for br in d["body_ref_config"]:
        if type(br) is list:
            body_ref_config.append(Matrix(br))
        elif type(br) is dict:
            if "xyzrpy" in br:
                body_ref_config.append(xyz_rpy_to_matrix(br["xyzrpy"]))                
            else:
                t = br["translation"] if "translation" in br else [0,0,0]
                if "rotation" in br:
                    if type(br["rotation"]) is dict:
                        if "Q" in br["rotation"]:
                            r = quaternion_to_matrix(br["rotation"]["Q"])
                        elif "rpy" in br["rotation"]:
                            r = rpy_to_matrix(br["rotation"]["rpy"])
                        elif "axis" in br["rotation"]:
                            axis = br["rotation"]["axis"] if "axis" in br["rotation"] else [0,0,1]
                            angle = br["rotation"]["angle"] if "angle" in br["rotation"] else 0
                            r = SO3Exp(Matrix(axis),angle)
                    else:
                        r = Matrix(br["rotation"])
                else:
                    r = Matrix(Identity(3))
                body_ref_config.append(transformation_matrix(r,t))
    
    assert("ee" in d)
    
    # transformation matrix directly in ee
    if type(d["ee"]) is list and type(d["ee"][0]) is list and type(d["ee"][0][0]) is not list:
        ee = Matrix(d["ee"])
    # dict in directly in ee
    elif type(d["ee"]) is dict:
        if "xyzrpy" in d["ee"]:
            ee = xyz_rpy_to_matrix(d["ee"]["xyzrpy"])
        else:
            t = d["ee"]["translation"] if "translation" in d["ee"] else [0,0,0]
            if "rotation" in d["ee"]:
                if type(d["ee"]["rotation"]) is dict:
                    if "Q" in d["ee"]["rotation"]:
                        r = quaternion_to_matrix(d["ee"]["rotation"]["Q"])
                    elif "rpy" in d["ee"]["rotation"]:
                        r = rpy_to_matrix(d["ee"]["rotation"]["rpy"])
                    elif "axis" in d["ee"]["rotation"]:
                        axis = d["ee"]["rotation"]["axis"] if "axis" in d["ee"]["rotation"] else [0,0,1]
                        angle = d["ee"]["rotation"]["angle"] if "angle" in d["ee"]["rotation"] else 0
                        r = SO3Exp(Matrix(axis),angle)
                else:
                    r = Matrix(d["ee"]["rotation"])
            else:
                r = Matrix(Identity(3))
            ee = transformation_matrix(r,t)
    # list of ee
    elif type(d["ee"]) is list:
        ee = []
        for eei in d["ee"]:
            if type(eei) is list:
                ee.append(Matrix(eei))
            elif type(eei) is dict:
                if "xyzrpy" in eei:
                    ee.append(xyz_rpy_to_matrix(eei["xyzrpy"]))                
                else:
                    t = eei["translation"] if "translation" in eei else [0,0,0]
                    if "rotation" in eei:
                        if type(eei["rotation"]) is dict:
                            if "Q" in eei["rotation"]:
                                r = quaternion_to_matrix(eei["rotation"]["Q"])
                            elif "rpy" in eei["rotation"]:
                                r = rpy_to_matrix(eei["rotation"]["rpy"])
                            elif "axis" in eei["rotation"]:
                                axis = eei["rotation"]["axis"] if "axis" in eei["rotation"] else [0,0,1]
                                angle = eei["rotation"]["angle"] if "angle" in eei["rotation"] else 0
                                r = SO3Exp(Matrix(axis),angle)
                        else:
                            r = Matrix(eei["rotation"])
                    else:
                        r = Matrix(Identity(3))
                    ee.append(transformation_matrix(r,t))        
    else:
        raise ValueError(f"ee {d['ee']} cannot be processed.")
    
    
    Mb = []
    if "mass_inertia" in d:
        for mb in d["mass_inertia"]:
            if type(mb) is list:
                Mb.append(Matrix(mb))
            elif type(mb) is dict:
                if "mass" in mb and "inertia" in mb and "com" in mb:
                    if type(mb["inertia"]) is list:
                        if len(mb["inertia"]) == 6:
                            Mb.append(
                                mass_matrix_mixed_data(
                                    mb["mass"],
                                    inertia_matrix(*mb["inertia"]),
                                    Matrix(mb["com"])
                                )
                            )
                        else:    
                            Mb.append(
                                mass_matrix_mixed_data(
                                    mb["mass"],
                                    Matrix(mb["inertia"]),
                                    Matrix(mb["com"])
                                )
                            )
                    elif type(mb["inertia"]) is dict and "index" in mb["inertia"]:
                        Mb.append(
                            mass_matrix_mixed_data(
                                mb["mass"], 
                                symbolic_inertia_matrix(mb["inertia"]["index"],
                                                    mb["inertia"]["pointmass"]),
                                Matrix(mb["com"])
                                )
                            )
                    elif type(mb["inertia"]) is dict:
                        Mb.append(
                            mass_matrix_mixed_data(
                                mb["mass"], 
                                inertia_matrix(**mb["inertia"]),
                                Matrix(mb["com"])
                                )
                            )
                    else:
                        Mb.append(
                            mass_matrix_mixed_data(
                                mb["mass"], 
                                mb["inertia"]*Identity(3),
                                Matrix(mb["com"])
                                )
                            )
                        # raise ValueError(f"Inertia {mb['inertia']} not supported.")
            else:
                raise ValueError("Unable to process mass_inertia.")
    
    q = Matrix(d["q"]) if "q" in d else None        
    qd = Matrix(d["qd"]) if "qd" in d else None        
    q2d = Matrix(d["q2d"]) if "q2d" in d else None        
    q3d = Matrix(d["q3d"]) if "q3d" in d else None        
    q4d = Matrix(d["q4d"]) if "q4d" in d else None
    if "WEE" in d:
        assert(type(d["WEE"] is list))
        if type(d["WEE"][0]) is list:
            WEE = [Matrix(w) for w in d["WEE"]]
        else:
            WEE = Matrix(d["WEE"])
    else:
        WEE = zeros(6,1)        
    if "WDEE" in d:
        assert(type(d["WDEE"] is list))
        if type(d["WDEE"][0]) is list:
            WDEE = [Matrix(w) for w in d["WDEE"]]
        else:
            WDEE = Matrix(d["WDEE"])
    else:
        WDEE = zeros(6,1)        
    if "W2DEE" in d:
        assert(type(d["W2DEE"] is list))
        if type(d["W2DEE"][0]) is list:
            W2DEE = [Matrix(w) for w in d["W2DEE"]]
        else:
            W2DEE = Matrix(d["W2DEE"])
    else:
        W2DEE = zeros(6,1)        
    
    skd = SymbolicKinDyn(gravity_vector=gravity,ee=ee,
                         body_ref_config=body_ref_config,
                         joint_screw_coord=joint_screw_coord,
                         config_representation=config_representation, 
                         Mb=Mb, parent=parent, support=support, 
                         child=child, ee_parent=ee_parent,
                         q=q,qd=qd,q2d=q2d,q3d=q3d,q4d=q4d,
                         WEE=WEE, WDEE=WDEE, W2DEE=W2DEE)
    return skd

def generate_template_yaml(path: str="edit_me.yaml", structure: Optional[str]=None, 
                        dof: int=0, tree: bool=True, **kwargs) -> None:
    """Generate template yaml file to modify for own robot.

    Args:
        path (str, optional): Path where to save generated yaml file.
            Defaults to 'edit_me.yaml' 
        structure (str, optional): string containing only 'r' and 'p' of 
            joint order. Use "prr" for a robot which has 1 prismatic joint
            followed by 2 revolute joints. Defaults to None.
        dof (int, optional): Degrees of freedom. Is usually calculated 
            by length of 'structure'. Defaults to 0.
        tree (bool, optional): Generate parent, child and support array 
            in yaml file. Defaults to True.

    Raises:
        ValueError: Unexpected letter in 'structure'.
    """
    if type(structure) is str:
        if not dof:
           dof = len(structure)
        else:
            # fill structure with r until length matches dof
            while len(structure) < dof:
                structure += "r"
            
    elif dof:
        structure = "r"*dof
    y = ["---"]
    if tree:
        y.append("parent:")
        for i in range(dof):
            y.append(f"  - {i}")
        y.append("")
     
        y.append("child:")
        for i in range(dof):
            y.append(f"  - [{i+1 if i < dof-1 else ''}]")
        y.append("")
     
        y.append("support:")
        for i in range(dof):
            y.append(f"  - [{','.join([str(j) for j in range(1,i+2)])}]")
        y.append("")
        
        y.append(f"ee_parent: {dof}")
        y.append("")
        
    y.append("gravity: [0,0,-g]")
    y.append("")
    
    y.append("representation: spatial")
    y.append("")
    
    y.append("joint_screw_coord:")
    for i in range(dof):
        if structure:
            if structure[i] in "rR":
                y.append(f"  - type: revolute")
                y.append(f"    axis: [0,0,1]")
                y.append(f"    vec: [0,0,0]")
                y.append("")
            elif structure[i] in "pP":
                y.append(f"  - type: prismatic")
                y.append(f"    axis: [0,0,1]")
                y.append("")
            else:
                raise ValueError(f"structure {structure[i]} not supported.")
        else:
            y.append("  - []")
            y.append("")
    
    y.append("body_ref_config:")
    for i in range(dof):
        y.append("  - rotation:")
        y.append("      axis: [0,0,1]")
        y.append("      angle: 0")
        y.append("    translation: [0,0,0]")
        y.append("")
    
    y.append("ee:")
    y.append("  - rotation:")
    y.append("      axis: [0,0,1]")
    y.append("      angle: 0")
    y.append("    translation: [0,0,0]")
    y.append("")
    
    y.append("mass_inertia:")
    for i in range(dof):
        y.append(f"  - mass: m{i+1}")
        y.append( "    inertia:")
        y.append(f"      Ixx: Ixx{i+1}")
        y.append(f"      Ixy: Ixy{i+1}")
        y.append(f"      Ixz: Ixz{i+1}")
        y.append(f"      Iyy: Iyy{i+1}")
        y.append(f"      Iyz: Iyz{i+1}")
        y.append(f"      Izz: Izz{i+1}")
        y.append( "    com: [0,0,0]")
        y.append("")
    
    y.append(f"q: [{','.join(['q'+str(i+1) for i in range(dof)])}]")
    y.append(f"qd: [{','.join(['dq'+str(i+1) for i in range(dof)])}]")
    y.append(f"q2d: [{','.join(['ddq'+str(i+1) for i in range(dof)])}]")
    y.append("WEE: [0,0,0,0,0,0]")
    y.append("")
    
    if "return_dict" in kwargs and kwargs["return_dict"]:
        return yaml.load("\n".join(y))
        
    
    # check if path ends with .yaml
    if not bool(regex.search("(.yaml|.YAML|.yml|.YML)\Z", path)):
        path += ".yaml"
        
    with open(path, "w+") as f:
        f.write("\n".join(y))
    
    
def generate_template_json(path: str="edit_me.json", structure: Optional[str]=None, 
                        dof: int=0, tree: bool=True) -> None:
    """Generate template json file to modify for own robot.

    Args:
        path (str, optional): Path where to save generated json file.
            Defaults to 'edit_me.json' 
        structure (str, optional): string containing only 'r' and 'p' of 
            joint order. Use "prr" for a robot which has 1 prismatic joint
            followed by 2 revolute joints. Defaults to None.
        dof (int, optional): Degrees of freedom. Is usually calculated 
            by length of 'structure'. Defaults to 0.
        tree (bool, optional): Generate parent, child and support array 
            in yaml file. Defaults to True.

    Raises:
        ValueError: Unexpected letter in 'structure'.
    """
    d = generate_template_yaml("",structure,dof,tree,return_dict=True)
    s= json.dumps(d,indent=2)
    # check if path ends with .json
    if not bool(regex.search("(.JSON|.json)\Z", path)):
        path += ".json"
    
    with open(path, "w+") as f:
        f.write(s)


def generate_template_python(path: str="edit_me.py", 
                             structure: Optional[str]=None, dof: int=0, 
                             tree: bool=True, urdf: bool=False) -> None:
    """Generate template python file to modify for own robot.

    Args:
        path (str, optional): Path where to save generated python file.
            Defaults to 'edit_me.py' 
        structure (str, optional): string containing only 'r' and 'p' of 
            joint order. Use "prr" for a robot which has 1 prismatic joint
            followed by 2 revolute joints. Defaults to None.
        dof (int, optional): Degrees of freedom. Is usually calculated 
            by length of 'structure'. Defaults to 0.
        tree (bool, optional): Generate parent, child and support array 
            in yaml file. Defaults to True.

    Raises:
        ValueError: Unexpected letter in 'structure'.
    """
    if type(structure) is str:
        if not dof:
           dof = len(structure)
        else:
            # fill structure with r until length matches dof
            while len(structure) < dof:
                structure += "r"
    elif dof:
        structure = "r"*dof
    
    p = ["from skidy import (SymbolicKinDyn,"]
    p.append("                   transformation_matrix,")
    if not urdf:
        p.append("                   mass_matrix_mixed_data,")
        p.append("                   joint_screw,")
        p.append("                   inertia_matrix,")
    p.append("                   SO3Exp,")
    p.append("                   generalized_vectors)")
    p.append("from skidy.symbols import g, pi")
    p.append("import sympy")
    p.append("")
    p.append("# Define symbols: (Hint: you can import the most common use symbols from skidy.symbols instead)")
    if not urdf:
        p.append(f"{'m'+ ', m'.join(str(i) for i in range(1, dof+1))} = sympy.symbols('{'m'+ ' m'.join(str(i) for i in range(1, dof+1))}', real=True, const=True)")
        p.append(f"{'Ixx'+ ', Ixx'.join(str(i) for i in range(1, dof+1))} = sympy.symbols('{'Ixx'+ ' Ixx'.join(str(i) for i in range(1, dof+1))}', real=True, const=True)")
        p.append(f"{'Ixy'+ ', Ixy'.join(str(i) for i in range(1, dof+1))} = sympy.symbols('{'Ixy'+ ' Ixy'.join(str(i) for i in range(1, dof+1))}', real=True, const=True)")
        p.append(f"{'Ixz'+ ', Ixz'.join(str(i) for i in range(1, dof+1))} = sympy.symbols('{'Ixz'+ ' Ixz'.join(str(i) for i in range(1, dof+1))}', real=True, const=True)")
        p.append(f"{'Iyy'+ ', Iyy'.join(str(i) for i in range(1, dof+1))} = sympy.symbols('{'Iyy'+ ' Iyy'.join(str(i) for i in range(1, dof+1))}', real=True, const=True)")
        p.append(f"{'Iyz'+ ', Iyz'.join(str(i) for i in range(1, dof+1))} = sympy.symbols('{'Iyz'+ ' Iyz'.join(str(i) for i in range(1, dof+1))}', real=True, const=True)")
        p.append(f"{'Izz'+ ', Izz'.join(str(i) for i in range(1, dof+1))} = sympy.symbols('{'Izz'+ ' Izz'.join(str(i) for i in range(1, dof+1))}', real=True, const=True)")
    else:
        p.append("lee = sympy.symbols('lee', real=True, const=True)")
    p.append("") # m1, m2,
    if tree and not urdf:
        p.append("# Define connectivity graph")
        p.append(f"parent = [0{',' if dof > 1 else ']'}")
        for i in range(1, dof):
            p.append(f"          {i}{',' if dof > i+1 else ']'}")
        p.append("")
     
        p.append(f"child = [[{1 if dof > 1 else ''}]{',' if dof > 1 else ']'}")
        for i in range(1, dof):
            p.append(f"         [{i+1 if dof > i+1 else ''}]{',' if dof > i+1 else ']'}")
        p.append("")
     
        p.append(f"support = [[1]{',' if dof > 1 else ']'}")
        for i in range(1,dof):
            p.append(f"           [{','.join([str(j) for j in range(1,i+2)])}]{',' if dof > i+1 else ']'}")
        p.append("")
        
        p.append(f"ee_parent = {dof}")
        p.append("")
        
    if urdf:
        p.append("urdfpath = '/path/to/urdf' # TODO: change me!")
        p.append("")
    
    p.append("# gravity vector")
    p.append("gravity = sympy.Matrix([0,0,-g])")
    p.append("")
    
    if not urdf:
        p.append("# representation of joint screw coordinates and body reference configurations")
        p.append("representation = 'spatial' # alternative: 'body_fixed'")
        p.append("")
        
        p.append("# joint screw coordinates (6x1 sympy.Matrix per joint)")
        p.append("joint_screw_coord = []")
        for i in range(dof):
            if structure:
                if structure[i] in "rR":
                    p.append('joint_screw_coord.append(joint_screw(axis=[0,0,1], vec=[0,0,0], revolute=True))')
                elif structure[i] in "pP":
                    p.append('joint_screw_coord.append(joint_screw(axis=[0,0,1], revolute=False))')
                else:
                    raise ValueError(f"structure {structure[i]} not supported.")
            else:
                p.append("  - []")
                p.append("")
        p.append("")
        
        p.append("# body reference configurations (4x4 SE3 Pose (sympy.Matrix) per link)")
        p.append("body_ref_config = []")
        for i in range(dof):
            p.append("body_ref_config.append(transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0]))")
        p.append("")
    
    p.append("# end-effector configuration w.r.t. last link body fixed frame in the chain (4x4 SE3 Pose (sympy.Matrix))")
    p.append("ee = transformation_matrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0])")
    p.append("")
    
    if not urdf:
        p.append("# mass_inertia parameters (6x6 sympy.Matrix per link)")
        p.append("Mb = []")
        for i in range(dof):
            p.append(f"Mb.append(mass_matrix_mixed_data(m{i+1}, inertia_matrix(Ixx{i+1},Ixy{i+1},Ixz{i+1},Iyy{i+1},Iyz{i+1},Izz{i+1}), sympy.Matrix([0,0,0])))")
        p.append("")
        
        p.append(f"q, qd, q2d = generalized_vectors(len(body_ref_config), startindex=1)")
        p.append("WEE = sympy.zeros(6,1)")
        p.append("")
    
    p.append("skd = SymbolicKinDyn(gravity_vector=gravity,")
    p.append("                     ee=ee,")
    if not urdf: 
        p.append("                     body_ref_config=body_ref_config,")
        p.append("                     joint_screw_coord=joint_screw_coord,")
        p.append("                     config_representation=representation,")
        p.append("                     Mb=Mb,")
        if tree:
            p.append("                     parent=parent,")
            p.append("                     child=child,")
            p.append("                     support=support,")
            p.append("                     ee_parent=ee_parent,")
    p.append("                     )")
    p.append("")
    if urdf:
        p.append("skd.load_from_urdf(path = urdfpath,")
        p.append("                   symbolic=False, # symbolify equations? (eg. use Ixx instead of numeric value)")
        p.append("                   cse=False, # use common subexpression elimination? ")
        p.append("                   simplify_numbers=True, # round numbers if close to common fractions like 1/2 etc and replace eg 3.1416 by pi?")
        p.append("                   tolerance=0.0001, # tolerance for simplify numbers")
        p.append("                   max_denominator=8, # define max denominator for simplify numbers to avoid simplification to something like 13/153")
        p.append("                   )")
        p.append("")
    
        p.append(f"q, qd, q2d = generalized_vectors(len(skd.body_ref_config), startindex=1)")
        p.append("WEE = sympy.zeros(6,1)")
        p.append("")
    
    p.append("# run Calculations")
    p.append("skd.closed_form_kinematics_body_fixed(q, qd, q2d, simplify=True)")
    p.append("skd.closed_form_inv_dyn_body_fixed(q, qd, q2d, WEE=WEE, simplify=True)")
    p.append("")
    
    p.append("# Generate Code")
    p.append('skd.generate_code(python=True, C=False, Matlab=False, latex=False,')
    p.append('                 folder="./generated_code", use_global_vars=True,')
    p.append('                 name="plant", project="Project")')
    p.append("")
    
    
    # check if path ends with .yaml
    if not bool(regex.search("(.py)\Z", path)):
        path += ".py"
        
    with open(path, "w+") as f:
        f.write("\n".join(p))


def robot_from_urdf(path: str, symbolic: bool=False, cse: bool=False, 
                    simplify_numbers: bool=True, tolerance: float=0.0001, 
                    max_denominator: int=9) -> SymbolicKinDyn:
    """Load robot from urdf.

        Args:
            path (str): path to URDF.
            symbolic (bool, optional): 
                generate symbols for numeric values. 
                Defaults to False.
            cse (bool, optional): 
                use common subexpression elimination. Defaults to False.
            simplify_numbers (bool, optional): 
                Use eg. pi/2 instead of 1.5708. Defaults to True.
            tolerance (float, optional): 
                tolerance for simplify_numbers. Defaults to 0.0001.
            max_denominator (int, optional): 
                Maximum denominator to use for simplify numbers to avoid
                values like 13/153. Use 0 to deactivate. Defaults to 9.

        Raises:
            NotImplementedError: supports only the joint types 
                "revolute", "continuous" and "prismatic".
        
        Returns:
            skidy.SymbolicKinDyn object.
    """            
    skd = SymbolicKinDyn()
    skd.load_from_urdf(path, symbolic, simplify_numbers, 
                       cse, tolerance, max_denominator)
    return skd

def skd_to_yaml(skd: SymbolicKinDyn, path: str="robot.yaml", **kwargs) -> None | dict:
    """Generate YAML file from SymbolicKinDyn object.

    Args:
        skd (SymbolicKinDyn): SymbolicKinDyn with set parameters.
        path (str, optional): path where to save .yaml file. 
            Defaults to "robot.yaml".
    """
    y = ["---"]
    dof = skd.n or len(skd.body_ref_config)
    if skd.parent is not None:
        y.append("parent:")
        for i in skd.parent:
            y.append(f"  - {i}")
        y.append("")

    if skd.child is not None:
        y.append("child:")
        for i in skd.child:
            y.append(f"  - {i}")
        y.append("")
    
    if skd.support is not None:
        y.append("support:")
        for i in skd.support:
            y.append(f"  - {i}")
        y.append("")
        
    if skd.ee_parent is not None:
        y.append(f"ee_parent: {skd.ee_parent}")
        y.append("")
        
    if skd.gravity_vector is not None:
        y.append(f"gravity: {[str(i) for i in skd.gravity_vector]}")
        y.append("")
    
    y.append(f"representation: {skd.config_representation}")
    y.append("")
    
    y.append("joint_screw_coord:")
    for i in skd.joint_screw_coord:
        y.append(f"  - {[str(j) for j in i]}")
    y.append("")
    
    y.append("body_ref_config:")
    for i in skd.body_ref_config:
        y.append("  - rotation:")
        y.append(f"      [{[str(j) for j in i[0,:3]]},")
        y.append(f"       {[str(j) for j in i[1,:3]]},")
        y.append(f"       {[str(j) for j in i[2,:3]]}]")
        y.append(f"    translation: {[str(j) for j in i[:3,3]]}")
        y.append("")
    
    y.append("ee:")
    if skd.ee is not None: 
        for i in skd.ee if type(skd.ee) is list else [skd.ee]:
            y.append("  - rotation:")
            y.append(f"      [{[str(j) for j in i[0,:3]]},")
            y.append(f"       {[str(j) for j in i[1,:3]]},")
            y.append(f"       {[str(j) for j in i[2,:3]]}]")
            y.append(f"    translation: {[str(j) for j in i[:3,3]]}")
            y.append("")
    else:
        y.append("  - rotation:")
        y.append("      axis: [0,0,1]")
        y.append("      angle: 0")
        y.append("    translation: [0,0,0]")
        y.append("")
    
    if skd.Mb is not None:
        y.append("mass_inertia:")
        for i in skd.Mb:
            if False:
                y.append(f"  - [{[str(j) for j in i[0,:]]},")
                y.append(f"     {[str(j) for j in i[1,:]]},")
                y.append(f"     {[str(j) for j in i[2,:]]},")
                y.append(f"     {[str(j) for j in i[3,:]]},")
                y.append(f"     {[str(j) for j in i[4,:]]},")
                y.append(f"     {[str(j) for j in i[5,:]]}]")
                y.append("")
            else:
                y.append(f"  - mass: {str(i[3,3])}")
                y.append( "    inertia:")
                y.append(f"      Ixx: {str(i[0,0])}")
                y.append(f"      Ixy: {str(i[0,1])}")
                y.append(f"      Ixz: {str(i[0,2])}")
                y.append(f"      Iyy: {str(i[1,1])}")
                y.append(f"      Iyz: {str(i[1,2])}")
                y.append(f"      Izz: {str(i[2,2])}")
                y.append(f"    com: [{str(skd.simplify(i[2,4]/i[3,3]))}, {str(skd.simplify(i[0,5]/i[3,3]))}, {str(skd.simplify(i[1,3]/i[3,3]))}]")
                y.append("")
    
    if skd.q is not None:
        y.append(f"q: {[str(i) for i in skd.q]}")    
    else:
        y.append(f"q: [{','.join(['q'+str(i+1) for i in range(dof)])}]")
    
    if skd.qd is not None:
        y.append(f"qd: {[str(i) for i in skd.qd]}")    
    else:
        y.append(f"qd: [{','.join(['dq'+str(i+1) for i in range(dof)])}]")
    
    if skd.q2d is not None:
        y.append(f"q2d: {[str(i) for i in skd.q2d]}")    
    else:
        y.append(f"q2d: [{','.join(['ddq'+str(i+1) for i in range(dof)])}]")
    
    if skd.q3d is not None:
        y.append(f"q3d: {[str(i) for i in skd.q3d]}")    
    
    if skd.q4d is not None:
        y.append(f"q4d: {[str(i) for i in skd.q4d]}")
    y.append("")
    
    
    if type(skd.WEE) is list:
        y.append("WEE:")
        for i in skd.WEE:
            y.append(f"  - {[str(j) for j in i]}")
    else: 
        y.append(f"WEE: {[str(i) for i in skd.WEE]}")
    
    if type(skd.WDEE) is list:
        y.append("WDEE:")
        for i in skd.WDEE:
            y.append(f"  - {[str(j) for j in i]}")
    elif skd.WDEE != zeros(6,1): 
        y.append(f"WDEE: {[str(i) for i in skd.WDEE]}")
    
    if type(skd.W2DEE) is list:
        y.append("W2DEE:")
        for i in skd.W2DEE:
            y.append(f"  - {[str(j) for j in i]}")
    elif skd.W2DEE != zeros(6,1): 
        y.append(f"W2DEE: {[str(i) for i in skd.W2DEE]}")

    y.append("")
    
    for i in range(len(y)):
        y[i] = y[i].replace("'","")
        # remove trailing zeros
        y[i] = regex.sub("(?<=\W\d+\.\d+)0+(?=(\D|$))|(?<=\W\d+)\.0+(?=(\D|$))", "",y[i])
    if "return_dict" in kwargs and kwargs["return_dict"]:
        return yaml.load("\n".join(y))
        
    
    # check if path ends with .yaml
    if not bool(regex.search("(.yaml|.YAML|.yml|.YML)\Z", path)):
        path += ".yaml"
        
    with open(path, "w+") as f:
        f.write("\n".join(y))
        


def skd_to_json(skd: SymbolicKinDyn, path: str="robot.json") -> None:
    """Generate JSON file from SymbolicKinDyn object.

    Args:
        skd (SymbolicKinDyn): SymbolicKinDyn with set parameters.
        path (str, optional): path where to save .json file. 
            Defaults to "robot.json".
    """
    d = skd_to_yaml(skd,return_dict=True)
    s= json.dumps(d,indent=2)
    # check if path ends with .json
    if not bool(regex.search("(.JSON|.json)\Z", path)):
        path += ".json"
    
    with open(path, "w+") as f:
        f.write(s)

def urdf_to_yaml(urdf_path: str, yaml_path: str="robot.yaml", 
                 symbolic: bool=False, cse: bool=False, 
                 simplify_numbers: bool=True, tolerance: float=0.0001, 
                 max_denominator: int=9) -> SymbolicKinDyn:
    """Load robot from urdf and save it to YAML file.

        Args:
            urdf_path (str): path to URDF.
            yaml_path (str, optional): path where to save .yaml file. 
                Defaults to "robot.yaml".
            symbolic (bool, optional): 
                generate symbols for numeric values. 
                Defaults to False.
            cse (bool, optional): 
                use common subexpression elimination. Defaults to False.
            simplify_numbers (bool, optional): 
                Use eg. pi/2 instead of 1.5708. Defaults to True.
            tolerance (float, optional): 
                tolerance for simplify_numbers. Defaults to 0.0001.
            max_denominator (int, optional): 
                Maximum denominator to use for simplify numbers to avoid
                values like 13/153. Use 0 to deactivate. Defaults to 9.

        Raises:
            NotImplementedError: supports only the joint types 
                "revolute", "continuous" and "prismatic".
    """            
    skd = SymbolicKinDyn()
    skd.load_from_urdf(urdf_path, symbolic, simplify_numbers, 
                       cse, tolerance, max_denominator)
    return skd_to_yaml(skd,yaml_path)

def urdf_to_json(urdf_path: str, json_path: str="robot.json", 
                 symbolic: bool=False, cse: bool=False, 
                 simplify_numbers: bool=True, tolerance: float=0.0001, 
                 max_denominator: int=9) -> SymbolicKinDyn:
    """Load robot from urdf and save it to JSON file.

        Args:
            urdf_path (str): path to URDF.
            json_path (str, optional): path where to save .json file. 
                Defaults to "robot.json".
            symbolic (bool, optional): 
                generate symbols for numeric values. 
                Defaults to False.
            cse (bool, optional): 
                use common subexpression elimination. Defaults to False.
            simplify_numbers (bool, optional): 
                Use eg. pi/2 instead of 1.5708. Defaults to True.
            tolerance (float, optional): 
                tolerance for simplify_numbers. Defaults to 0.0001.
            max_denominator (int, optional): 
                Maximum denominator to use for simplify numbers to avoid
                values like 13/153. Use 0 to deactivate. Defaults to 9.

        Raises:
            NotImplementedError: supports only the joint types 
                "revolute", "continuous" and "prismatic".
    """            
    skd = SymbolicKinDyn()
    skd.load_from_urdf(urdf_path, symbolic, simplify_numbers, 
                       cse, tolerance, max_denominator)
    return skd_to_json(skd,json_path)
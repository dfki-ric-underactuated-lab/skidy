import yaml
import json
from typing import Union, Iterable
from KinematicsGenerator import (SymbolicKinDyn, joint_screw, 
                                 SymbolicInertiaMatrix, MassMatrixMixedData, 
                                 SO3Exp, TransformationMatrix, InertiaMatrix,
                                 quaternion_to_matrix, xyz_rpy_to_matrix,
                                 rpy_to_matrix)
from sympy import Matrix, Identity, parse_expr, Expr
import regex

def robot_from_yaml(path: str) -> SymbolicKinDyn:
    """Parse yaml robot description and return SymbolicKinDyn Object.

    Args:
        path (str): Path to yaml file.

    Returns:
        KinematicsGenerator.SymbolicKinDyn object.
    """
    with open(path, "r") as stream:
        y = yaml.safe_load(stream)
    return dict_parser(y)

def robot_from_json(path: str) -> SymbolicKinDyn:
    """Parse json robot description and return SymbolicKinDyn Object.

    Args:
        path (str): Path to json file.

    Returns:
        KinematicsGenerator.SymbolicKinDyn object.
    """
    with open(path, "r") as stream:
        y = json.load(stream)
    return dict_parser(y)

def parse_hierarchical_expr(x:Union[list,dict,str], 
                            include_keys:Iterable={}, 
                            exclude_keys:Iterable={}) -> Union[list,dict,Expr]:
    """Convert strings in datastructure (list or dict) to sympy 
    expressions.

    Args:
        x (list|dict|str): Hierarchical structure which might contain 
            symbolic expression as string. 
        include_keys (Iterable): Only convert strings with listed dict_keys. 
            Defaults to {}.    
        exclude_keys (Iterable): Don't convert strings with listed dict_keys. 
            Defaults to {}.    
    Returns:
        list|dict|sympy.Expr: Same datastructure with strings converted 
            to sympy.Expr 
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
        KinematicsGenerator.SymbolicKinDyn object.
    """
    d = parse_hierarchical_expr(d, include_keys={"mass", "index", "Ixx", "Ixy", "Ixz", "Iyy", "Ixz", "Izz"})
    
    config_representation = d["representation"] if "representation" in d else "spacial"
    
    parent = d["parent"] if "parent" in d else []
    child = d["child"] if "child" in d else []
    support = d["support"] if "support" in d else []
    try:
        gravity = d["gravity"] if "gravity" in d else d["gravity_vector"]
    except KeyError:
        gravity = None
    
    joint_screw_coord = []
    if "joint_screw_coord" not in d:
        raise KeyError("joint_screw_coord not found.")
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
    
    body_ref_config = []
    if "body_ref_config" not in d:
        raise KeyError("body_ref_config not found")
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
                body_ref_config.append(TransformationMatrix(r,t))
    
    if "ee" not in d:
        raise KeyError("ee not found")
            
    if type(d["ee"]) is list:
        ee = Matrix(d["ee"])
    elif type(d["ee"]) is dict:
        if "xyzrpy" in br:
            ee = xyz_rpy_to_matrix(br["xyzrpy"])
        else:
            t = d["ee"]["translation"] if "translation" in d["ee"] else [0,0,0]
            if "rotation" in d["ee"]:
                if type(d["ee"]["rotation"]) is dict:
                    if "Q" in br["rotation"]:
                        r = quaternion_to_matrix(br["rotation"]["Q"])
                    elif "rpy" in br["rotation"]:
                        r = rpy_to_matrix(br["rotation"]["rpy"])
                    elif "axis" in br["rotation"]:
                        axis = d["ee"]["rotation"]["axis"] if "axis" in d["ee"]["rotation"] else [0,0,1]
                        angle = d["ee"]["rotation"]["angle"] if "angle" in d["ee"]["rotation"] else 0
                        r = SO3Exp(Matrix(axis),angle)
                else:
                    r = Matrix(d["ee"]["rotation"])
            else:
                r = Matrix(Identity(3))
            ee = TransformationMatrix(r,t)
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
                                MassMatrixMixedData(
                                    mb["mass"],
                                    InertiaMatrix(*mb["inertia"]),
                                    Matrix(mb["com"])
                                )
                            )
                        else:    
                            Mb.append(
                                MassMatrixMixedData(
                                    mb["mass"],
                                    Matrix(mb["inertia"]),
                                    Matrix(mb["com"])
                                )
                            )
                    elif type(mb["inertia"]) is dict and "index" in mb["inertia"]:
                        Mb.append(
                            MassMatrixMixedData(
                                mb["mass"], 
                                SymbolicInertiaMatrix(mb["inertia"]["index"],
                                                    mb["inertia"]["pointmass"]),
                                Matrix(mb["com"])
                                )
                            )
                    elif type(mb["inertia"]) is dict:
                        Mb.append(
                            MassMatrixMixedData(
                                mb["mass"], 
                                InertiaMatrix(**mb["inertia"]),
                                Matrix(mb["com"])
                                )
                            )
                    else:
                        raise ValueError(f"Inertia {mb['inertia']} not supported.")
            else:
                raise ValueError("Unable to process mass_inertia.")
            
    
    skd = SymbolicKinDyn(gravity,ee,body_ref_config,joint_screw_coord,config_representation, Mb, parent, support, child)
    return skd

def generate_template_yaml(path: str="edit_me.yaml", structure: str = None, 
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
            # fill structure with r until length mathes dof
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
        
    y.append("gravity: [0,0,g]")
    y.append("")
    
    y.append("representation: spacial")
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
    y.append("  rotation:")
    y.append("    axis: [0,0,1]")
    y.append("    angle: 0")
    y.append("  translation: [0,0,0]")
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
    
    if "return_dict" in kwargs and kwargs["return_dict"]:
        return yaml.load("\n".join(y))
        
    
    # check if path ends with .yaml
    if not bool(regex.search("(.yaml|.YAML|.yml|.YML)\Z", path)):
        path += ".yaml"
        
    with open(path, "w+") as f:
        f.write("\n".join(y))
    
    
def generate_template_json(path: str="edit_me.json", structure: str = None, 
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


def generate_template_python(path:str="edit_me.py", structure:str=None, dof:int=0, tree:bool=True, urdf=False):
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
    
    p = ["from KinematicsGenerator import (SymbolicKinDyn,"]
    p.append("                                 TransformationMatrix,")
    if not urdf:
        p.append("                                 MassMatrixMixedData,")
        p.append("                                 joint_screw,")
        p.append("                                 InertiaMatrix,")
    p.append("                                 SO3Exp,")
    p.append("                                 generalized_vectors)")
    p.append("from KinematicsGenerator.symbols import g, pi")
    p.append("import sympy")
    p.append("")
    p.append("# Define symbols: (Hint: you can import the most common use symbols from KinematicsGenerator.symbols instead)")
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
        
    if urdf:
        p.append("urdfpath = '/path/to/urdf' # TODO: change me!")
        p.append("")
    
    p.append("# gravity vector")
    p.append("gravity = sympy.Matrix([0,0,g])")
    p.append("")
    
    if not urdf:
        p.append("# representation of joint screw coordinates and body reference configurations")
        p.append("representation = 'spacial' # alternative: 'body_fixed'")
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
            p.append("body_ref_config.append(TransformationMatrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0]))")
        p.append("")
    
    p.append("# end-effector configuration w.r.t. last link body fixed frame in the chain (4x4 SE3 Pose (sympy.Matrix))")
    p.append("ee = TransformationMatrix(r=SO3Exp(axis=[0,0,1],angle=0),t=[0,0,0])")
    p.append("")
    
    if not urdf:
        p.append("# mass_inertia parameters (6x6 sympy.Matrix per link)")
        p.append("Mb = []")
        for i in range(dof):
            p.append(f"Mb.append(MassMatrixMixedData(m{i+1}, InertiaMatrix(Ixx{i+1},Ixy{i+1},Ixz{i+1},Iyy{i+1},Iyz{i+1},Izz{i+1}), sympy.Matrix([0,0,0])))")
        p.append("")
        
        p.append(f"q, qd, q2d = generalized_vectors(len(body_ref_config), startindex=1)")
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
    p.append("                     )")
    p.append("")
    if urdf:
        p.append("skd.load_from_urdf(path = urdfpath,")
        p.append("                   symbolic=True, # symbolify equations? (eg. use Ixx instead of numeric value)")
        p.append("                   cse_ex=False, # use common subexpression elimination? ")
        p.append("                   simplify_numbers=True, # round numbers if close to common fractions like 1/2 etc and replace eg 3.1416 by pi?")
        p.append("                   tolerance=0.0001, # tolerance for simplify numbers")
        p.append("                   max_denominator=8, # define max denominator for simplify numbers to avoid simplification to something like 13/153")
        p.append("                   )")
        p.append("")
    
        p.append(f"q, qd, q2d = generalized_vectors(len(skd.body_ref_config), startindex=1)")
        p.append("")
    
    p.append("# run Calculations")
    p.append("skd.closed_form_kinematics_body_fixed(q, qd, q2d, simplify_expressions=True)")
    p.append("skd.closed_form_inv_dyn_body_fixed(q, qd, q2d, simplify_expressions=True)")
    p.append("")
    
    p.append("# Generate Code")
    p.append('skd.generateCode(python=True, C=False, Matlab=False,')
    p.append('                 folder="./generated_code", use_global_vars=True,')
    p.append('                 name="plant", project="Project")')
    p.append("")
    
    
    # check if path ends with .yaml
    if not bool(regex.search("(.py)\Z", path)):
        path += ".py"
        
    with open(path, "w+") as f:
        f.write("\n".join(p))


def robot_from_urdf(path: str, symbolic: bool=True, simplify_numbers: bool=True, 
                    cse_ex: bool=False, tolerance: float=0.0001, 
                    max_denominator: int=9) -> SymbolicKinDyn:
    skd = SymbolicKinDyn()
    skd.load_from_urdf(path, symbolic, simplify_numbers, 
                       cse_ex, tolerance, max_denominator)
    return skd



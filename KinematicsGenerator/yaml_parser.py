import yaml
import json
from KinematicsGenerator import (SymbolicKinDyn, joint_screw, 
                                 SymbolicInertiaMatrix, MassMatrixMixedData, 
                                 SO3Exp, TransformationMatrix)
from sympy import Matrix, Identity
import regex

def robot_from_yaml(path: str):
    """Parse yaml robot description and return SymbolicKinDyn Object.

    Args:
        path (str): Path to yaml file.

    Returns:
        KinematicsGenerator.SymbolicKinDyn object.
    """
    with open(path, "r") as stream:
        y = yaml.safe_load(stream)
    return dict_parser(y)

def robot_from_json(path: str):
    """Parse json robot description and return SymbolicKinDyn Object.

    Args:
        path (str): Path to json file.

    Returns:
        KinematicsGenerator.SymbolicKinDyn object.
    """
    with open(path, "r") as stream:
        y = json.load(stream)
    return dict_parser(y)
    
def dict_parser(d: dict):
    """Parse dict to SymbolicKinDyn object.

    Args:
        d (dict): Dictionary conatianing robot description.

    Raises:
        KeyError: Entry not found.
        ValueError: Unexpected entry.

    Returns:
        KinematicsGenerator.SymbolicKinDyn object.
    """
    config_representation = d["representation"] if "representation" in d else "spacial"
    
    parent = d["parent"] if "parent" in d else []
    child = d["child"] if "child" in d else []
    support = d["support"] if "support" in d else []
    try:
        gravity = d["gravity"] if "gravity" in d else d["gravity_vector"]
    except KeyError:
        raise KeyError("gravity not found")
    
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
            t = br["translation"] if "translation" in br else [0,0,0]
            if "rotation" in br:
                if type(br["rotation"]) is dict:
                    axis = br["rotation"]["axis"] if "axis" in br["rotation"] else [0,0,1]
                    angle = br["rotation"]["angle"] if "angle" in br["rotation"] else 0
                    r = SO3Exp(axis,angle)
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
        t = d["ee"]["translation"] if "translation" in d["ee"] else [0,0,0]
        if "rotation" in d["ee"]:
            if type(d["ee"]["rotation"]) is dict:
                axis = d["ee"]["rotation"]["axis"] if "axis" in d["ee"]["rotation"] else [0,0,1]
                angle = d["ee"]["rotation"]["angle"] if "angle" in d["ee"]["rotation"] else 0
                r = SO3Exp(axis,angle)
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
                        Mb.append(
                            MassMatrixMixedData(
                                mb["mass"],
                                Matrix(mb["inertia"]),
                                Matrix(mb["com"])
                            )
                        )
                    elif type(mb["inertia"]) is dict:
                        Mb.append(
                            MassMatrixMixedData(
                                mb["mass"], 
                                SymbolicInertiaMatrix(mb["inertia"]["index"],
                                                    mb["inertia"]["pointmass"]),
                                Matrix(mb["com"])
                                )
                            )
                    else:
                        raise ValueError(f"Inertia {mb['inertia']} not supported.")
            else:
                raise ValueError("Unable to process mass_inertia.")
            
    
    skd = SymbolicKinDyn(gravity,ee,body_ref_config,joint_screw_coord,config_representation, Mb, parent, support, child)
    return skd

def generate_empty_yaml(path: str="edit_me.yaml", structure: str = None, dof: int=0, tree: bool=True, **kwargs):
    """Generate empty yaml file to modify for own robot.

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
        y.append(f"      index: {i+1}")
        y.append( "      pointmass: False")
        y.append( "    com: [0,0,0]")
        y.append("")
    
    if "return_dict" in kwargs and kwargs["return_dict"]:
        return yaml.load("\n".join(y))
        
    
    # check if path ends with .yaml
    if not bool(regex.search("(.yaml|.YAML|.yml|.YML)\Z", path)):
        path += ".yaml"
        
    with open(path, "w+") as f:
        f.write("\n".join(y))
    
    
def generate_empty_json(path: str="edit_me.json", structure: str = None, dof: int=0, tree: bool=True):
    """Generate empty json file to modify for own robot.

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
    d = generate_empty_yaml("",structure,dof,tree,return_dict=True)
    s= json.dumps(d,indent=2)
    # check if path ends with .json
    if not bool(regex.search("(.JSON|.json)\Z", path)):
        path += ".json"
    
    with open(path, "w+") as f:
        f.write(s)




from KinematicsGenerator.parser import robot_from_yaml
import time
import os
import matplotlib.pyplot as plt
import numpy as np

def benchmark(parallel=True,simplify=True, cse=False, all = False):
    # run through all options
    if all:
        for parallel in [True, False]:
            for simplify in [True, False]:
                for cse in [True, False]:
                    benchmark(parallel,simplify,cse)
        return
    
    os.chdir(os.path.dirname(__file__))

    r1 = robot_from_yaml("./r.yaml")
    r2 = robot_from_yaml("./rr.yaml")
    r3 = robot_from_yaml("./rrr.yaml")
    r4 = robot_from_yaml("./rrrr.yaml")


    startr1 = time.time()
    r1.closed_form_kinematics_body_fixed(simplify_expressions=simplify, cse_ex=cse, parallel=parallel)
    kinr1 = time.time()
    r1.closed_form_inv_dyn_body_fixed(simplify_expressions=simplify, cse_ex=cse, parallel=parallel)
    dynr1 =  time.time()
    r1.generateCode(python=True,name="r1")
    genr1 = time.time()

    print("r:")
    print("kin:", kinr1-startr1)
    print("dyn:", dynr1-kinr1)
    print("sum:", dynr1-startr1)

    startr2 = time.time()
    r2.closed_form_kinematics_body_fixed(simplify_expressions=simplify, cse_ex=cse, parallel=parallel)
    kinr2 = time.time()
    r2.closed_form_inv_dyn_body_fixed(simplify_expressions=simplify, cse_ex=cse, parallel=parallel)
    dynr2 =  time.time()
    r2.generateCode(python=True,name="r2")
    genr2 = time.time()

    print("rr:")
    print("kin:", kinr2-startr2)
    print("dyn:", dynr2-kinr2)
    print("sum:", dynr2-startr2)


    startr3 = time.time()
    r3.closed_form_kinematics_body_fixed(simplify_expressions=simplify, cse_ex=cse, parallel=parallel)
    kinr3 = time.time()
    r3.closed_form_inv_dyn_body_fixed(simplify_expressions=simplify, cse_ex=cse, parallel=parallel)
    dynr3 =  time.time()
    r3.generateCode(python=True,name="r3")
    genr3 = time.time()

    print("rrr:")
    print("kin:", kinr3-startr3)
    print("dyn:", dynr3-kinr3)
    print("sum:", dynr3-startr3)

    startr4 = time.time()
    r4.closed_form_kinematics_body_fixed(simplify_expressions=simplify, cse_ex=cse, parallel=parallel)
    kinr4 = time.time()
    r4.closed_form_inv_dyn_body_fixed(simplify_expressions=simplify, cse_ex=cse, parallel=parallel)
    dynr4 =  time.time()
    r4.generateCode(python=True,name="r4")
    genr4 = time.time()

    print("rrrr:")
    print("kin:", kinr4-startr4)
    print("dyn:", dynr4-kinr4)
    print("sum:", dynr4-startr4)

    barkin = [kinr1-startr1, kinr2-startr2, kinr3-startr3,kinr4-startr4]
    bardyn = [dynr1-kinr1, dynr2-kinr2, dynr3-kinr3, dynr4-kinr4]
    bargen = [genr1-dynr1, genr2-dynr2, genr3-dynr3, genr4-dynr4]

    r = [1,2,3,4]
    plt.bar(r, barkin, color='#7f6d5f', edgecolor='white', width=.8, label="forward kinematics")
    plt.bar(r, bardyn, bottom=barkin, color='#557f2d', edgecolor='white', width=.8, label="inverse dynamics")
    plt.bar(r, bargen, bottom=np.add(barkin,bardyn).tolist(), color='#2d7f5e', edgecolor='white', width=.8, label="code generation")

    plt.xticks(r, r)
    plt.xlabel("degrees of freedom")
    plt.ylabel("time in s")

    plt.legend()

    title = ""
    if parallel:
        title += "parallel; "
    else:
        title += "serial; "
    if simplify:
        title += "with simplification; "
    else:
        title += "without simplification; "
    if cse:    
        title += "with cse"
    else:
        title += "without cse"

    plt.title(title)
    name = title.replace(";","").replace(" ","_") + ".png"
    plt.savefig(name)
    # plt.show()


if __name__ == "__main__":
    benchmark(all=True)    
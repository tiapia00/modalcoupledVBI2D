import textwrap
import subprocess


def write_input(params: dict, base_inp_path: str, out_inp_path: str):
    prepend_text = textwrap.dedent(f"""\
    *PARAMETER
    h={params['height']}
    b={params['base']}
    density={params['density']}/(h*b)
    max_freq = {params['max_freq']}
    n_modes = {params['n_modes']}
    n_nodes = {params['n_nodes']}
    """)

    nodes = []
    for i in range(params['num_nodes']):
        x = i * params['dx']
        nodes.append((i+1, x, 0.0))

    num_elems = params['num_nodes'] - 1

    elements = []
    for i in range(num_elems):
        n1 = i + 1
        n2 = i + 2
        elements.append((i+1, n1, n2))

    with open(base_inp_path + '.inp', 'r') as file:
        original_content = file.read()

    with open(out_inp_path + '.inp', 'w') as file:
        file.write(prepend_text)
        file.write("*Node, nset=ALLNODES\n")
        for nid, x, y in nodes:
            file.write(f"{nid}, {x:.6f}, {y:.6f}\n")

        file.write("*Element, type=B23, elset=ALLELS\n")
        for eid, n1, n2 in elements:
            file.write(f"{eid}, {n1}, {n2}\n")

        file.write(original_content)


def run_job(out_inp_path: str, abaqus_cmd: str):
    subprocess.run([abaqus_cmd, f"job={out_inp_path}", "ask_delete=off", "interactive"])
    subprocess.run([abaqus_cmd, "cae", "noGUI=extractodb.py", "--", f"{out_inp_path}"])

from odbAccess import openOdb
from abaqusConstants import *
import numpy as np
import sys
import csv
import shutil

odb_path = sys.argv[-1]
# --- USER INPUT ---
step_name = 'Freqs'
output_csv = 'modes.csv'
# -------------------

# Open ODB
odb = openOdb(path=odb_path + '.odb')
step = odb.steps[step_name]
instance_name = list(odb.rootAssembly.instances.keys())[0]
instance = odb.rootAssembly.instances[instance_name]

# Get all node labels in the instance
all_nodes = sorted(instance.nodes, key=lambda n: n.label)
node_labels = [node.label for node in all_nodes]
node_coords = [node.coordinates for node in all_nodes]

elem_type = instance.elements[0].type

# Infer dimensionality
dim = 2
if elem_type.startswith('B2'):
    dim = 2
elif elem_type.startswith('B3'):
    dim = 3

# Sort by node label
num_nodes = len(all_nodes)
num_modes = len(step.frames) - 1

# Initialize displacement array: shape (nodes, 3 components, modes)
modes = np.zeros((num_nodes, dim, num_modes))

# Loop over modes (skip frame 0, which is initial state)
freqs = []
for i in range(1, len(step.frames)):
    frame = step.frames[i]
    freqs.append(frame.frequency)
    disp_field = frame.fieldOutputs['U']
    disp_dict = {val.nodeLabel: val.data for val in disp_field.values}
    for j, node in enumerate(all_nodes):
        u = disp_dict.get(node.label, (0.0, 0.0, 0.0))  # fallback if node missing
        modes[j, :, i-1] = u

destination = r'C:\Users\mattiaan\Documents\Coding\modaluncoupled'

with open('coords.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['NodeLabel', 'X', 'Y', 'Z'])
    for label, coord in zip(node_labels, node_coords):
        writer.writerow([label] + list(coord))

shutil.copy('coords.csv', destination)

# === Write U1, U2, U3 to separate CSV files ===
if dim == 2:
    comps = ['U1', 'U2']
else:
    comps = ['U1', 'U2', 'U3']

with open('freqs.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow('Frequencies')
    for freq in freqs:
        writer.writerow([freq])

shutil.copy('freqs.csv', destination)

for comp_idx, comp_name in enumerate(comps):
    filename = '{}.csv'.format(comp_name)
    with open(filename, 'w') as f:  # 'wb' instead of 'w', for Python 2.7 csv
        writer = csv.writer(f)
        # Write header
        header = ['NodeLabel'] + ['Mode_{}'.format(i + 1) for i in range(num_modes)]
        writer.writerow(header)
        # Write data
        for j in range(num_nodes):
            row = [node_labels[j]] + list(modes[j, comp_idx, :])
            writer.writerow(row)
    shutil.copy(filename, destination)

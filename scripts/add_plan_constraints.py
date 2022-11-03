import argparse
import pathlib
import json

import sumolib.net

from lib.graph_converter import GraphConverter


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--net', type=str, default='grid.net.xml')
parser.add_argument('--file', type=str, default='plan_constraints.json')
parser.add_argument('--phase_min', type=int, default=10)
parser.add_argument('--phase_max', type=int, default=100)
args = parser.parse_args()


if __name__ == "__main__":
    path = pathlib.Path(args.path)
    net_path = path / args.net
    file_path = path / args.file

    net = sumolib.net.readNet(str(net_path), withPrograms=True)
    graph_converter = GraphConverter(net)

    plan_constraints = []
    for ctrl_id, ctrl in graph_converter.controllers.items():
        plan_constraint = {
            "controller_id": ctrl_id,
            "mode": "free",
            "shift": 0,
            "phases": [{
                "phase_id": phase["phase_id"],
                "phase_min": args.phase_min,
                "phase_max": args.phase_max
            } for phase in ctrl['phases']]
        }
        plan_constraints.append(plan_constraint)

    with open(file_path, 'w') as file:
        json.dump(plan_constraints, file, indent=4)

    print(f"Output file path: {file_path}")
    print(f"Add default plan constraints")

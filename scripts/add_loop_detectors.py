import argparse
import pathlib
import xml.etree.ElementTree as ET

import sumolib.net

from lib.graph_converter import GraphConverter


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--net', type=str, default="grid.net.xml")
parser.add_argument('--file', type=str, default="grid.loop_detectors.xml")
parser.add_argument('--out', type=str, default="output/loop_detectors.xml")
parser.add_argument('--num_detectors', '-n', type=int, default=2)
parser.add_argument('--freq', type=int, default=60)
parser.add_argument('--min_length', type=float, default=50.)
args = parser.parse_args()


if __name__ == "__main__":
    path = pathlib.Path(args.path)
    net_path = path / args.net
    file_path = path / args.file

    net = sumolib.net.readNet(str(net_path), withPrograms=True)
    graph = GraphConverter(net)

    detectors_counter = 0
    root = ET.Element("additional")
    for edge in net.getEdges():
        edge_id = edge.getID()
        comment = ET.Comment(f"Edge: {edge_id}")
        root.append(comment)

        for lane in edge.getLanes():
            lane_id = lane.getID()
            lane_length = lane.getLength()

            if lane_id not in graph.chains:
                continue

            positions = []
            if lane_id in graph.input_chains:
                detector_offset = min(0.1 * lane_length, 10.)
                positions.append(("start", detector_offset))

            if lane_length > args.min_length:
                num_detectors = args.num_detectors
                segment_length = lane_length / (num_detectors + 1)
                for idx in range(num_detectors):
                    position = (idx + 1) * segment_length
                    positions.append((f"{idx + 1}", position))

            for pos_type, pos in positions:
                induction_loop = ET.SubElement(root, 'inductionLoop', {
                    "id": f"{lane_id}_{pos_type}_loop",
                    "lane": lane_id,
                    "pos": str(pos),
                    "freq": str(args.freq),
                    "file": str(args.out),
                })
                detectors_counter += 1

    tree = ET.ElementTree(root)
    tree.write(file_path)

    print(f"Output file path: {file_path}")
    print(f"Graph size: {len(graph.chains)}")
    print(f"Added loop detectors: {detectors_counter}")

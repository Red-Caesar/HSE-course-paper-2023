import xml.etree.ElementTree as ET
from typing import Tuple, List

def parsing(filename: str, fill_same: float) -> List[List[float]]:
    tree = ET.parse(f'data/{filename}')
    root = tree.getroot()
    graph = root.find('graph')
    matrix = []
    n = len(graph.findall('vertex'))
    for i, ver in enumerate(graph.findall('vertex')):
        matrix.append([fill_same for _ in range(n)])
        for edge in ver.findall('edge'):
            matrix[i][int(edge.text)] = float(edge.get('cost'))
    return matrix
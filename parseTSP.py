import xml.etree.ElementTree as ET
from typing import Tuple, Union
import numpy as np

def parsing(filename: str, fill_same: float, city_numbers: int=None) -> Tuple[np.array, Union[float, None]]:
    tree = ET.parse(f'data/{filename}.xml')
    root = tree.getroot()
    graph = root.find('graph')
    matrix = []
    if not city_numbers:
        n = len(graph.findall('vertex'))
    else:
        n = city_numbers
    for i, ver in enumerate(graph.findall('vertex')):
        if i >= n:
            continue
        matrix.append([fill_same for _ in range(n)])
        for edge in ver.findall('edge'):
            if int(edge.text) >= n:
                continue
            matrix[i][int(edge.text)] = float(edge.get('cost'))
    answer = None
    if not city_numbers:        
        with open(f'data/{filename}_answer.txt', 'r') as f:
            answer = float(f.readline())
    return np.array(matrix), answer
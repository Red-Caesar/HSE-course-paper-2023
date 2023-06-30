import xml.etree.ElementTree as ET
from typing import Tuple, Union
import numpy as np

def parsing(filename: str, fill_same: float, city_numbers: int) -> Tuple[np.array, Union[float, None]]:
    tree = ET.parse(f'data/{filename}.xml')
    root = tree.getroot()
    graph = root.find('graph')
    matrix = []
    for i, ver in enumerate(graph.findall('vertex')):
        if i >= city_numbers:
            continue
        matrix.append([fill_same for _ in range(city_numbers)])
        for edge in ver.findall('edge'):
            if int(edge.text) >= city_numbers:
                continue
            matrix[i][int(edge.text)] = float(edge.get('cost'))
    answer = None
    if city_numbers == len(graph.findall('vertex')):        
        with open(f'data/{filename}_answer.txt', 'r') as f:
            answer = float(f.readline())
    return np.array(matrix), answer
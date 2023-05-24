import numpy as np
from typing import List, Tuple
from collections import deque
import time 


class BnBMethod:
    def __init__(self) -> None:
        self.times = []
        self.records = []
    
    class Node:
        def __init__(self, upper_bound: float, lower_bound: int, level: int, branches: List[int]) -> None:
            self.upper_bound = upper_bound
            self.lower_bound = lower_bound
            self.level = level
            self.prev_branches = branches
            self.parent = None
            self.children = []
    
    class Record:
        def __init__(self, cost: float, path: List[int]) -> None:
            self.cost = cost
            self.path = path
    
    def remove_row_col(self, matrix: List[List], remove_list: List) -> List[List]:
        row = []
        col = []
        for i in range(len(remove_list)-1):
            row.append(remove_list[i])
            col.append(remove_list[i+1])

        matrix = np.delete(matrix, row, 0)
        matrix = np.delete(matrix, col, 1)
        return matrix

    def find_lower_bound(self, matrix: List[List], remove_list: List) -> int: 
        if len(remove_list) > 1:
            matrix = self.remove_row_col(matrix, remove_list)
        row_min = []
        col_min = []
        for i in range(len(matrix)):
            row_min.append(min(matrix[i])) 
            col_min.append(min(matrix[:, i]))
        return max(sum(row_min), sum(col_min))

    def find_upper_bound(self, matrix: List[List[float]], visited: List[int]) -> Tuple[float, List[int]]:
        upper_bound = 0
        if len(visited) > 1:
            for i in range(len(visited)-1):
                upper_bound += matrix[visited[i]][visited[i+1]]
            
        start = visited[-1]
        queue = []
        queue.append(start)
        while queue:
            i = queue.pop()
            if len(visited) == len(matrix):
                upper_bound += matrix[i][visited[0]]
                visited.append(visited[0])
                break

            row = dict(zip(range(len(matrix)), matrix[i]))
            row = sorted(row.items(), key=lambda x: x[1])
            
            for key, val in row:
                if key not in visited:
                    upper_bound += val
                    visited.append(key)
                    queue.append(key)
                    break
        return (upper_bound, visited)

    def branches_and_boundaries(self, matrix: List[List[float]], record: Record) -> Record:
        root = self.Node(0, 0, -1, [0])
        queue = deque()
        queue.append(root)
        while queue:
            node = queue.pop()
            level = node.level + 1
            if level >= len(matrix):
                continue

            node.lower_bound = self.find_lower_bound(matrix, node.prev_branches[:])
            node.upper_bound, path = self.find_upper_bound(matrix, node.prev_branches[:])

            if node.lower_bound > record.cost:
                continue
            if node.upper_bound < record.cost:
                record.cost, record.path = node.upper_bound, path
            
            for next_branch in range(len(matrix)):
                if next_branch not in node.prev_branches:
                    children = self.Node(0, 0, level, node.prev_branches + [next_branch])
                    children.parent = node
                    node.children.append(children)
                    queue.append(children)
    
    def start(self, example: List[List[float]], repeat: int):
        record = self.Record(float('inf'), None)
        # Можно распаралеллить потом
        for _ in range(repeat):
            start = time.time()
            self.branches_and_boundaries(example, record)
            end = time.time() - start
            self.times.append(end)
        return record.cost, record.path
    
    def getMeanTime(self) -> float:
        return np.mean(self.times)

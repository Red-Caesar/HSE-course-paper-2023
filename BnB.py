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
            self.prev_cost = 0
    
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

    def find_upper_bound(self, matrix: List[List[float]], prev_branches: List[int], cur_cost: float) -> Tuple[float, List[int]]:
        upper_bound = cur_cost

        start = prev_branches[-1]
        queue = [start]

        while queue:
            i = queue.pop()
            if len(prev_branches) == len(matrix):
                upper_bound += matrix[i][prev_branches[0]]
                prev_branches.append(prev_branches[0])
                break

            min_element = float('inf')
            min_index = -1
            for j in range(len(matrix)):
                if j not in prev_branches and matrix[i][j] < min_element:
                    min_element = matrix[i][j]
                    min_index = j
            queue.append(min_index)
            upper_bound += min_element
            prev_branches.append(min_index)

        return (upper_bound, prev_branches)

    def branches_and_boundaries(self, matrix: List[List[float]], record: Record) -> Record:
        root = self.Node(0, 0, -1, [0])
        queue = deque()
        queue.append(root)
        while queue:
            node = queue.pop()
            level = node.level + 1
            if level >= len(matrix):
                continue

            node.lower_bound = node.prev_cost + self.find_lower_bound(matrix, node.prev_branches[:])
            node.upper_bound, path = self.find_upper_bound(matrix, node.prev_branches[:], node.prev_cost)

            if node.lower_bound > record.cost:
                continue
            if node.upper_bound < record.cost:
                record.cost, record.path = node.upper_bound, path
            
            for next_branch in range(len(matrix)):
                if next_branch not in node.prev_branches:
                    children = self.Node(0, 0, level, node.prev_branches + [next_branch])
                    children.prev_cost = node.prev_cost + matrix[node.prev_branches[-1]][next_branch]
                    queue.append(children)
    
    def start(self, example: List[List[float]], repeat: int):
        record = self.Record(float('inf'), None)
        for _ in range(repeat):
            start = time.time()
            self.branches_and_boundaries(example, record)
            end = time.time() - start
            self.times.append(end)
        return record.cost, record.path
    
    def getMeanTime(self) -> float:
        return np.mean(self.times)
import numpy as np
from typing import List, Tuple
import time 


class BnBMethod:
    def __init__(self) -> None:
        self.times = []
        self.records = []
    
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

    def find_upper_bound(self, matrix: List[List], visited: List) -> Tuple[int, List]:
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
                # upper_bound += matrix[i][visited[0]]
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

    # Надо продумать еще раз, что мне возвращается из рекурсии и не перебивается ли это другой рекурсией
    def branches_and_boundaries(self, matrix: List[List], path: List, record: Tuple[float, List]) -> Tuple[float, List]:
        lower_bound = self.find_lower_bound(matrix, path[:])
        upper_bound, upper_path = self.find_upper_bound(matrix, path[:])
        if upper_bound < record[0]:
            record = (upper_bound, upper_path)
        if lower_bound >= record[0]:
            self.records.append(record)
            return record
        for i in range(len(matrix)):
            if i not in path:
                record = self.branches_and_boundaries(matrix, path + [i], record)
        self.records.append(record)
        return record
    
    def start(self, example: List[List[float]]):
        record = (float('inf'), None)
        # for i in range(len(example)):
        start = time.time()
        self.branches_and_boundaries(example, [0], record)
        end = time.time() - start
        self.times.append(end)
    
    def getMeanTime(self) -> float:
        return np.mean(self.times)
    
    def getBestRecord(self) -> float:
        best = (float('inf'), None)
        for record, path in self.records:
            if record < best[0]:
                best = (record, path)
        return best
    
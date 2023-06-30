import numpy as np
from typing import List, Dict, Tuple
from scipy.special import expit
from collections import OrderedDict
import time 
import pandas as pd

class HopfieldModel:
    '''
    Как работает модель.

    Ей подается матрица путей, высчитываются веса и смещение по формулам. Потом ей подается матрица маршрута, причем разные и несколько раз,
    т.к. модель находит локальный минимум, а не глобальный.

    '''
    
    def __init__(self, distance_matrix: List[List[float]], fun_activation: str, param: List[int]=[1,1,1,1], repetition: int=5, iterations_limit: int=1000000, without_changes: int=1000) -> None:
        if len(param) != 4:
            raise ValueError('You should put 4 hyperparameters to the network') 
        if fun_activation not in ['threshold', 'sigmoid']:
            raise ValueError('Function name should be one of it: "threshold", "sigmoid"') 
        if not distance_matrix.any():
            raise ValueError('You should put the distance matrix to the network')
        
        self.distance_matrix = distance_matrix
        self.fun_activation = fun_activation
        self.A, self.B, self.C, self.D = param
        self.repetition = repetition
        self.iterations_limit = iterations_limit
        self.without_changes = without_changes
        self.city = len(distance_matrix)


        self.index_list, self.bias, self.weights = self.calculate_params()
        self.layer = OrderedDict()

        self.start_counter = 0
        self.isFinished = False

        self.best_energy_fun = None
        self.all_energy = []  
        self.times = []
        self.all_total = []

        self.final_path = None
        self.total = None
        self.string_path = None

        
    
    def delta(self, x: int, y: int) -> int:
        return int(x == y)

    def calculate_params(self) -> Tuple[List[str], int, Dict[str, Dict[str, float]]]:
        # calculate list of future layer indexes
        index_list = [f'{i}_{j}' for i in range(self.city) for j in range(self.city)]
        # calculate bias
        bias = self.C*self.city
        # calculate weights
        weights = dict()
        for index_i in index_list:
            for index_j in index_list:
                if index_i not in weights:
                    weights[index_i] = dict()
                if index_i == index_j:
                    weights[index_i][index_j] = 0
                else:
                    city_1, order_1 = list(map(int, index_i.split('_')))
                    city_2, order_2 = list(map(int, index_j.split('_')))
                    d = self.delta
                    weights[index_i][index_j] = -self.A*d(city_1, city_2)*(1 - d(order_1, order_2)) 
                    weights[index_i][index_j] -= self.B*d(order_1, order_2)*(1 - d(city_1, city_2)) 
                    weights[index_i][index_j] -= self.C 
                    weights[index_i][index_j] -= self.D*self.distance_matrix[city_1][city_2]*(d(order_2, order_1+1) + d(order_2, order_1-1))
        return index_list, bias, weights
    
    def fun(self, value: float) -> float:
        if self.fun_activation == 'threshold':
            if value > 0:
                return 1
            else:
                return 0
        elif self.fun_activation == 'sigmoid':
            return expit(value)
    
    def calculate_neuron(self, current_neuron, previous_layer) -> None:
        self.layer[current_neuron] = np.sum([previous_layer[j]*self.weights[current_neuron][index_j] for j, index_j in enumerate(self.index_list)]) + self.bias
        self.layer[current_neuron] = self.fun(self.layer[current_neuron])

    def calculate_energy_fun(self) -> float:
        e_fun = 0
        for index_i in self.index_list:
            for index_j in self.index_list:
                e_fun -= self.weights[index_i][index_j]*self.layer[index_i]*self.layer[index_j]
            e_fun -= 2*self.bias*self.layer[index_i]
        return e_fun 
    
    def isTSP(self, final_path: dict) -> bool:
        order_set = set()
        city_set = set()
        visited = 0
        for index in self.index_list:
            if int(final_path[index]) == 1:
                visited += 1
                city, order = list(map(int, index.split('_')))
                if order not in order_set and city not in city_set:
                    order_set.add(order)
                    city_set.add(city)
                else: 
                    # "It's not a TSP path. Different city was visited at the same time"
                    return False
        if visited != self.city:
            # "It's not a TSP path. Not enought city was visited"
            return False
        return True

    def start_network(self) -> bool:
        if self.start_counter > 100:
            return
        else:
            self.start_counter += 1
        for _ in range(self.repetition):
            path = np.random.randint(2, size=self.city**2)
            for index_i in self.index_list:
                self.calculate_neuron(index_i, path)
            if not self.best_energy_fun:
                self.best_energy_fun = self.calculate_energy_fun()
            k = 0
            start = time.time()
            new_e_fun = float('inf')
            without_changes_counter = 0
            while k < self.iterations_limit:
                neuron = np.random.choice(self.index_list, 1)[0]
                self.calculate_neuron(neuron, list(self.layer.values()))
                old_e_fun, new_e_fun = new_e_fun, self.calculate_energy_fun()
                if new_e_fun < self.best_energy_fun:
                    self.best_energy_fun = new_e_fun
                    self.all_energy.append(new_e_fun)
                    self.final_path = self.layer
                if old_e_fun == new_e_fun:
                    without_changes_counter += 1
                else:
                    without_changes_counter = 0

                if without_changes_counter > self.without_changes:
                    break

            # if self.isTSP(self.final_path):
            end = time.time() - start
            self.times.append(end)
        self.total = self.calculate_distance()
        # self.all_total.append(self.total)
        self.isFinished = True
        
        if not self.isFinished:
            self.start_network()

    def calculate_distance(self) -> float:
        order_dict = dict()
        for index in self.index_list:
            if int(self.final_path[index]) == 1:
                city, order = list(map(int, index.split('_')))
                if order not in order_dict:
                    order_dict[order] = [city]
                else:
                    order_dict[order].append(city)

        order_turple = sorted(order_dict.items(), key=lambda x: x[0])
        # total = self.distance_matrix[len(order_turple) - 1][0] 
        total = 0
        path = []
        for i, order_city in enumerate(order_turple):
            path.append(', '.join(list(map(str, order_city[1]))))
            if i == 0:
                continue
            cur_cities = order_city[1]
            prev_cities = order_turple[i-1][1]
            for cur_city in cur_cities:
                total += self.distance_matrix[prev_cities[0]][cur_city]
        for cur_city in order_turple[0][1]:
            total += self.distance_matrix[order_turple[-1][1][0]][cur_city]
        path.append(', '.join(list(map(str, order_turple[0][1]))))
        self.string_path = ' -> '.join(path)
        return total

    
    def print_result(self) -> None:
        print('Total distance: ', self.total)
        print(self.string_path)
    
    def get_mean_time(self) -> float:
        return np.mean(self.times)
    
    def get_mean_path(self) -> float:
        return np.mean(self.all_total)
    
    def get_isFinished(self) -> bool:
        return self.isFinished
    
    def get_total(self) -> float:
        return self.total
    
    def get_path(self) -> str:
        return self.string_path


def tune_model(task_name: str, matrix: np.array, fun: str, A: List[int], B: List[int], C: List[int], D: List[int], rep: int) -> None:
    table = []
    for i in range(len(A)):
        print(i)
        model = HopfieldModel(matrix, fun, [A[i], B[i], C[i], D[i]], rep)
        model.start_network()
        if model.get_isFinished():
            total = model.get_total()
            mean_time = model.get_mean_time()
            mean_path = model.get_mean_path()
            path = model.get_path()
            table.append([A[i], B[i], C[i], D[i], total, mean_time, mean_path, path])
            df = pd.DataFrame(table, columns=['A', 'B', 'C', 'D', 'result', 'mean_time', 'mean_path', 'path'])
            df.to_excel(f'results/{task_name}_{i}.xlsx')  

    df = pd.DataFrame(table, columns=['A', 'B', 'C', 'D', 'result', 'mean_time', 'mean_path', 'path'])
    df.to_excel(f'results/{task_name}.xlsx')  
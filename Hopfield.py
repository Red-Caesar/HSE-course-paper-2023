import numpy as np
from typing import List, Dict, Tuple
from scipy.special import expit
from collections import OrderedDict
import time 

class HopfieldModel:
    '''
    Как работает модель.

    Ей подается матрица путей, высчитываются веса и смещение по формулам. Потом ей подается матрица маршрута, причем разные и несколько раз,
    т.к. модель находит локальный минимум, а не глобальный.

    Храним:
    1. Гиперпараметры: A, B, C, D
    2. Веса
    3. Смещение
    4. Слой (название нейронов слоя)
    5. Прошлое значение функции энергии
    6. Название функции активации, которую мы хотим использовать
    7. Сколько раз будем запускать сеть
    8. Финальный путь
    9. Возможно результаты, которые мы захотим сохранить в файл(время, финальный путь)
    10. Стоп параметр
    
    Функции, которые нам нужны:
    1. Функция, которая высчитывает веса и записывает их в переменную
    2. Функция, которая создает рандомный маршрут и возвращает его
    3. Функция активации, которую мы будем использовать
    4. Функция, которая высчитывает значение одного нейрона и возвращает его
    5. Расчет функции энергии
    6. Функция, которая работает с сетью: высчитывает слой в первый раз и итерирует его потом.
    7. Вывол финального пути
    8. Функция для сохранения результатов в файл?
    9. Функция, которая выводит стоимость пути
    

    (неплохо было бы сделать принты, которые бы выводили прогресс)
    Нужно будет еще потом замерять время.
    '''
    
    def __init__(self, distance_matrix: List[List], fun_activation: str, param: List=[1,1,1,1], iterations: int=5, iterations_limit = 100000) -> None:
        if len(param) != 4:
            raise ValueError('You should put 4 hyperparameters to the network') 
        if fun_activation not in ['threshold', 'sigmoid']:
            raise ValueError('Function name should be one of it: "threshold", "sigmoid"') 
        if not distance_matrix.any():
            raise ValueError('You should put the distance matrix to the network')
        
        self.distance_matrix = distance_matrix
        self.fun_activation = fun_activation
        self.A, self.B, self.C, self.D = param
        self.iterations = iterations
        self.iterations_limit = iterations_limit
        self.city = len(distance_matrix)


        self.index_list, self.bias, self.weights = self.calculate_params()
        self.layer = OrderedDict()

        self.energy_fun = None
        self.all_energy = []  
        self.times = []

        self.final_path = None
        self.total = None

        
    
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
    
    def check_if_TSP(self, final_path: dict) -> bool:
        order_dict = dict()
        visited = 0
        for index in self.index_list:
            if int(final_path[index]) == 1:
                visited += 1
                city, order = list(map(int, index.split('_')))
                if order not in order_dict:
                    order_dict[order] = city
                else: 
                    # "It's not a TSP path. Different city was visited at the same time"
                    return False
        if visited != self.city:
            # "It's not a TSP path. Not enought city was visited"
            return False
        return True

    def clear_previous_results(self):
        self.energy_fun = None
        self.all_energy = []  
        self.times = []
        self.final_path = None
# Сейчас у меня сохраняется лучший путь последней итерации)) А так быть не должно
    def get_path(self) -> bool:
        self.clear_previous_results()
        for _ in range(self.iterations):
            path = np.random.randint(2, size=self.city**2)
            for index_i in self.index_list:
                self.calculate_neuron(index_i, path)
            self.energy_fun = self.calculate_energy_fun()
            k = 0
            start = time.time()
            # print(self.final_path, self.city)
            # while not self.final_path:
            while True:
                if k >= self.iterations_limit:
                    break
                else:
                    k += 1 
                neuron = np.random.choice(self.index_list, 1)[0]
                self.calculate_neuron(neuron, list(self.layer.values()))
                new_e_fun = self.calculate_energy_fun()
                # Возможно это условие излишне и просто надо поставить контин, когда new_e_fun > self.energy_fun, а почему вообще > ?
                if new_e_fun > self.energy_fun:
                    break
                else:
                    self.energy_fun = new_e_fun
                    self.all_energy.append(self.energy_fun)
                    # if self.check_if_TSP(self.layer):
                    #     self.final_path = self.layer
                    self.final_path = self.layer
            end = time.time() - start
            self.times.append(end)
        if not self.final_path:
            return False
        else:
            self.total = self.calculate_distance()
            return True

    # Пока что я не могу считать это верно, потому что, чтобы это посчитать мы еще должны знать, какой был прошлый город 
    def calculate_distance(self) -> float:
        total = 0
        order_dict = dict()
        visited = 0
        for index in self.index_list:
            if int(self.final_path[index]) == 1:
                visited += 1
                city, order = list(map(int, index.split('_')))
                order_dict[order] = city

        order_turple = sorted(order_dict.items(), key=lambda x: x[0])
        for i, order_city in enumerate(order_turple):
            if i == 0:
                continue
            cur_city = int(order_city[1])
            prev_city = int(order_turple[i-1][1])
            total += self.distance_matrix[prev_city][cur_city]
        return total

    
    def print_result(self) -> None:
        print(self.total)
        print(self.all_energy)
        print(self.times)
        for index in self.index_list:
            if int(self.final_path[index]) == 1:
                city, order = list(map(int, index.split('_')))
                print(f'City #{city} order in path {order}')
    
    def mean_time(self) -> float:
        return np.mean(self.times)

from typing import List

import numpy as np

class KnapsackProblem:

    def __init__(self, file_path) -> None:
        instance = open(file_path, 'r')

        first_line : str = instance.readline()

        elements, max_weight = first_line.strip().split(' ')

        self.num_elements : int = int(elements)
        self.max_weight : int = int(max_weight)
        self.weights : List = []
        self.values : List = []

        for i in range(self.num_elements):
            line = instance.readline()
            value, weight = line.strip().split(' ')
            self.weights.append(int(weight))
            self.values.append(int(value))

    def weight(self, solution : List):
        cont = 0
        for i in range(self.num_elements):
            if solution[i] == 1:
                cont = cont + self.weights[i]
        return cont

    def fitness(self,solution : List) -> int:
        cont = 0
        for i in range(self.num_elements):
            if solution[i] == 1:
                cont = cont + self.values[i]
        return -1 * cont

    def is_valid_solution(self, solution : List):
        assert len(solution) == self.num_elements
        if self.weight(solution) <= self.max_weight:
            #print(f"Total Weight : {self.weight(solution)} Max Weight : {self.max_weight}" )
            #print(solution)
            return True
        return False


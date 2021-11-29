import numpy as np

def q_shape_one(vector):
    vector_max = np.max(np.absolute(vector))
    result = np.zeros(vector.shape)
    for i in range(len(vector)):
        if vector[i] < (0.5 * vector_max):
            result[i] = abs(vector[i]/(0.5 * vector_max))
        else:
            result[i] = 1
    return result

def q_shape_two(vector):
    vector_max = np.max(np.absolute(vector))
    result = np.zeros(vector.shape)
    for i in range(len(vector)):
        if vector[i] < (0.5 * vector_max):
            result[i] = np.power(vector[i]/(0.5 * vector_max),2)
        else:
            result[i] = 1
    return result

def q_shape_three(vector):
    vector_max = np.max(np.absolute(vector))
    result = np.zeros(vector.shape)
    for i in range(len(vector)):
        if vector[i] < (0.5 * vector_max):
            result[i] = np.power(vector[i]/(0.5 * vector_max),3)
        else:
            result[i] = 1
    return result

def q_shape_four(vector):
    vector_max = np.max(np.absolute(vector))
    result = np.zeros(vector.shape)
    for i in range(len(vector)):
        if vector[i] < (0.5 * vector_max):
            result[i] = np.sqrt(vector[i]/(0.5 * vector_max))
        else:
            result[i] = 1
    return result

q_functions = {
    "q1" : q_shape_one,
    "q2" : q_shape_two,
    "q3" : q_shape_three,
    "q4" : q_shape_four
}

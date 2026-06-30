import numpy as np

def get_dataset(program: str, data_size: int):
    if program == 'if':
        data = generate_if_dataset(data_size)
    else:
        raise ValueError(f"Unknown program type: {program}")     
    return data



def generate_if_dataset(data_size):
    data = []
    for _ in range(data_size):
        a = np.random.normal(1, 2)
        if a < 0:
            b = a * 3 + np.random.normal(0, 1)
        else:
            b = np.random.normal(8, 1)
        data.append([a, b])
    return data
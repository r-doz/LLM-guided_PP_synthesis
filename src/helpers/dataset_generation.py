import numpy as np

def get_dataset(program: str, data_size: int):
    if program == 'if':
        data = generate_if_dataset(data_size)
    elif program == 'mog1':
        data = generate_mog1_dataset(data_size)
    elif program == 'burglary':
        data = generate_burglary_dataset(data_size)
    elif program == 'csi':
        data = generate_csi_dataset(data_size)
    elif program == 'easytugwar':
        data = generate_easytugwar_dataset(data_size)
    else:
        raise ValueError(f"Unknown program type: {program}")     
    return data

def get_var_names(program: str):
    if program == 'if':
        return ['a', 'b']
    elif program == 'mog1':
        return ['mu', 'sigma', 'x']
    elif program == 'burglary':
        return ['burglary', 'earthquake', 'alarm', 'johncalls']
    elif program == 'csi':
        return ['u', 'v', 'w', 'x']
    elif program == 'easytugwar':
        return ['skill1', 'skill2', 'p1wins']
    else:
        raise ValueError(f"Unknown program type: {program}")



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

def generate_mog1_dataset(data_size):
    data = []
    for _ in range(data_size):
        mu = np.random.normal(20, 3)
        sigma = np.random.normal(2, 1)
        x = mu + sigma * np.random.normal(1, 1)
        data.append([mu, sigma, x])
    return data

def generate_burglary_dataset(data_size):
    data = []
    for _ in range(data_size):
        burglary = np.random.binomial(1, 0.001)  # 0.001
        earthquake = np.random.binomial(1, 0.002) # 0.002
        if burglary:
            if earthquake:
                alarm = np.random.binomial(1, 0.95)
            else:
                alarm = np.random.binomial(1, 0.94)
        else:
            if earthquake:
                alarm = np.random.binomial(1, 0.29)
            else:
                alarm = np.random.binomial(1, 0.001) # 0.001
        if alarm:
            johncalls = np.random.binomial(1, 0.9)
        else:
            johncalls = np.random.binomial(1, 0.05)
        data.append([burglary, earthquake, alarm, johncalls])
    return data

def generate_csi_dataset(data_size):
    data = []
    for _ in range(data_size):
        u = np.random.binomial(1, 0.3)
        v = np.random.binomial(1, 0.9)
        w = np.random.binomial(1, 0.1)
        if u:
            if w:
                x = np.random.binomial(1, 0.8)
            else:
                x = np.random.binomial(1, 0.2)
        else:
            if v:
                x = np.random.binomial(1, 0.8)
            else:
                x = np.random.binomial(1, 0.2)
        data.append([u, v, w, x])
    return data

def generate_easytugwar_dataset(data_size):
    data = []
    for _ in range(data_size):
        skill1 = np.random.normal(20, 4)
        skill2 = np.random.normal(20, 4)
        if skill1 > skill2:
            p1wins = 1.0
        else:
            p1wins = 0.0
        data.append([skill1, skill2, p1wins])
    return data
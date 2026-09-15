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
    elif program == 'biasedtugwar':
        data = generate_biasedtugwar_dataset(data_size)
    elif program == 'mixedcondition':
        data = generate_mixedcondition_dataset(data_size)
    elif program == 'multiplebranches':
        data = generate_multiplebranches_dataset(data_size)
    elif program == 'eyecolor':
        data = generate_eyecolor_dataset(data_size)
    elif program == 'hurricane':
        data = generate_hurricane_dataset(data_size)
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
    elif program == 'biasedtugwar':
        return ['skill1', 'skill2', 'p1wins']
    elif program == 'mixedcondition':
        return ['u', 'v', 'w']
    elif program == 'multiplebranches':
        return ['contentDifficulty', 'questionsAfterLectureLength']
    elif program == 'eyecolor':
        return ['eyecolor', 'haircolor', 'hairlenght']
    elif program == 'hurricane':
        return ['preplevel', 'damage']
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

def generate_biasedtugwar_dataset(data_size):
    data = []
    for _ in range(data_size):
        skill1 = np.random.normal(20, 4)
        skill2 = np.random.normal(20, 4)
        if 1.3 * skill2 - skill1 < 0:
            p1wins = 1.0
        else:
            p1wins = 0.0
        data.append([skill1, skill2, p1wins])
    return data

def generate_mixedcondition_dataset(data_size):
    data = []
    for _ in range(data_size):
        u = np.random.binomial(1, 0.3)
        v = np.random.normal(10, 2)
        if u == 1 and v > 12:
            w = np.random.normal(12, 2)
        else:
            w = np.random.normal(6, 2)
        data.append([u, v, w])
    return data

def generate_multiplebranches_dataset(data_size):
    data = []
    for _ in range(data_size):
        contentDifficulty = np.random.normal(30, 5)
        if contentDifficulty < 35:
            if contentDifficulty < 20:
                questionsAfterLectureLength = np.random.normal(2, 1)
            else:
                questionsAfterLectureLength = np.random.normal(10, 3)
        else:
            questionsAfterLectureLength = np.random.normal(25, 6)
        data.append([contentDifficulty, questionsAfterLectureLength])
    return data

def generate_eyecolor_dataset(data_size):
    # real_programs.py's eyecolor weights [0.8, 0.05, 0.04, 0.01] sum to 0.90, not 1
    # (already present in the ported program, inherited from the upstream .soga baseline
    # as-is) -- normalized here since np.random.choice requires a valid distribution.
    eyecolor_p = np.array([0.8, 0.05, 0.04, 0.01])
    eyecolor_p = eyecolor_p / eyecolor_p.sum()
    haircolor_p_by_eyecolor = {
        0: [0.8, 0.05, 0.04, 0.01, 0.1],
        1: [0.7, 0.15, 0.04, 0.01, 0.1],
        2: [0.4, 0.3, 0.18, 0.02, 0.1],
        3: [0.4, 0.29, 0.18, 0.03, 0.1],
    }
    data = []
    for _ in range(data_size):
        eyecolor = np.random.choice([0, 1, 2, 3], p=eyecolor_p)
        haircolor = np.random.choice([0, 1, 2, 3, 4], p=haircolor_p_by_eyecolor[eyecolor])
        hairlenght = np.random.choice([0, 1, 2], p=[0.6, 0.15, 0.25])
        data.append([eyecolor, haircolor, hairlenght])
    return data

def generate_hurricane_dataset(data_size):
    damage_p_by_preplevel = {0: [0.2, 0.8], 1: [0.2, 0.8], 2: [0.8, 0.2]}
    data = []
    for _ in range(data_size):
        preplevel = np.random.choice([0, 1, 2], p=[0.5, 0.2, 0.3])
        damage = np.random.choice([0, 1], p=damage_p_by_preplevel[preplevel])
        data.append([preplevel, damage])
    return data
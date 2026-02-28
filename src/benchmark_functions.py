import numpy as np


def get_benchmark_functions(default_dim=30):

    problems = []

    problems.append({
        "Name": "Sphere",
        "CostFunction": sphere_fcn,
        "nVar": default_dim,
        "VarMin": -100.0,
        "VarMax": 100.0,
    })

    problems.append({
        "Name": "Rastrigin",
        "CostFunction": rastrigin_fcn,
        "nVar": default_dim,
        "VarMin": -5.12,
        "VarMax": 5.12,
    })

    problems.append({
        "Name": "Rosenbrock",
        "CostFunction": rosenbrock_fcn,
        "nVar": default_dim,
        "VarMin": -5.0,
        "VarMax": 10.0,
    })

    problems.append({
        "Name": "Ackley",
        "CostFunction": ackley_fcn,
        "nVar": default_dim,
        "VarMin": -32.768,
        "VarMax": 32.768,
    })

    problems.append({
        "Name": "Griewank",
        "CostFunction": griewank_fcn,
        "nVar": default_dim,
        "VarMin": -600.0,
        "VarMax": 600.0,
    })

    problems.append({
        "Name": "Schwefel",
        "CostFunction": schwefel_fcn,
        "nVar": default_dim,
        "VarMin": -500.0,
        "VarMax": 500.0,
    })

    problems.append({
        "Name": "Zakharov",
        "CostFunction": zakharov_fcn,
        "nVar": default_dim,
        "VarMin": -10.0,
        "VarMax": 10.0,
    })

    return problems


# ================= COST FUNCTIONS =================

def sphere_fcn(x):
    x = np.asarray(x, dtype=float)
    return np.sum(x**2)


def rastrigin_fcn(x):
    x = np.asarray(x, dtype=float)
    d = x.size
    return 10*d + np.sum(x**2 - 10*np.cos(2*np.pi*x))


def rosenbrock_fcn(x):
    x = np.asarray(x, dtype=float)
    return np.sum(
        100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2
    )


def ackley_fcn(x):
    x = np.asarray(x, dtype=float)
    d = x.size
    a, b, c = 20.0, 0.2, 2*np.pi

    term1 = -a * np.exp(-b*np.sqrt(np.sum(x**2)/d))
    term2 = -np.exp(np.sum(np.cos(c*x))/d)

    return term1 + term2 + a + np.e


def griewank_fcn(x):
    x = np.asarray(x, dtype=float)
    d = x.size
    sum_term = np.sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d+1))))
    return sum_term - prod_term + 1


def schwefel_fcn(x):
    x = np.asarray(x, dtype=float)
    d = x.size
    return 418.9829*d - np.sum(x*np.sin(np.sqrt(np.abs(x))))


def zakharov_fcn(x):
    x = np.asarray(x, dtype=float)
    d = x.size
    i = np.arange(1, d+1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5*i*x)
    return sum1 + sum2**2 + sum2**4

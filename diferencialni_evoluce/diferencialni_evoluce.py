import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DEFINICE ÚČELOVÝCH FUNKCÍ (PROBLEM DEFINITIONS)
# ==========================================
# Všechny funkce jsou minimalizační a pracují vektorově nad celou populací:
# vstup X má tvar (počet_jedinců, dimenze), výstup tvar (počet_jedinců,).

def sphere(X: np.ndarray) -> np.ndarray:
    """
    Sphere (De Jong 1): f(x) = sum(x_i^2)
    Unimodální, separabilní. Globální minimum f(0, ..., 0) = 0.
    """
    return np.sum(X ** 2, axis=1)


def rosenbrock(X: np.ndarray) -> np.ndarray:
    """
    Rosenbrock (De Jong 2): f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Nonseparabilní, minimum leží v úzkém zakřiveném "údolí". Minimum f(1, ..., 1) = 0.
    """
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)


def schwefel(X: np.ndarray) -> np.ndarray:
    """
    Schwefel: f(x) = 418.9829 * D - sum(x_i * sin(sqrt(|x_i|)))
    Silně multimodální, separabilní, globální minimum je daleko od druhého nejlepšího lokálního.
    Minimum f(420.9687, ..., 420.9687) ≈ 0.
    """
    D = X.shape[1]
    return 418.9829 * D - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


@dataclass
class Problem:
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    lower: float
    upper: float


PROBLEMS = [
    Problem('Sphere', sphere, -5.12, 5.12),
    Problem('Rosenbrock', rosenbrock, -2.048, 2.048),
    Problem('Schwefel', schwefel, -500.0, 500.0),
]


# ==========================================
# 2. OPERÁTORY DE (MUTACE, KŘÍŽENÍ, HRANICE)
# ==========================================
# Všechny operátory pracují nad celou populací najednou (jedna generace = jedna maticová operace).

def random_indices(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """
    Pro každého aktivního jedince i vybere `count` vzájemně různých indexů r1, r2, ...,
    které jsou zároveň různé od i (podmínka i != r1 != r2 != r3).
    Vrací matici tvaru (n, count).
    """
    keys = rng.random((n, n))
    np.fill_diagonal(keys, np.inf)       # jedinec i nesmí vybrat sám sebe
    return np.argsort(keys, axis=1)[:, :count]


def mutation_rand1(P: np.ndarray, fitnesses: np.ndarray, F: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """DE/rand/1: v_i = x_r1 + F * (x_r2 - x_r3)"""
    r = random_indices(len(P), 3, rng)
    return P[r[:, 0]] + F[:, None] * (P[r[:, 1]] - P[r[:, 2]])


def mutation_best1(P: np.ndarray, fitnesses: np.ndarray, F: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """DE/best/1: v_i = x_best + F * (x_r1 - x_r2), x_best je nejlepší jedinec aktuální generace."""
    r = random_indices(len(P), 2, rng)
    x_best = P[np.argmin(fitnesses)]
    return x_best + F[:, None] * (P[r[:, 0]] - P[r[:, 1]])


def binomial_crossover(P: np.ndarray, V: np.ndarray, CR: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Binomiální křížení:
        u_ij = v_ij, pokud rand_j(0, 1) <= CR nebo j = j_rand
        u_ij = x_ij, jinak
    Pozice j_rand zaručuje, že alespoň jeden parametr zkušebního vektoru pochází z mutanta.
    """
    n, D = P.shape
    mask = rng.random((n, D)) <= CR[:, None]
    mask[np.arange(n), rng.integers(0, D, size=n)] = True
    return np.where(mask, V, P)


def repair_bounds(U: np.ndarray, problem: Problem, rng: np.random.Generator) -> np.ndarray:
    """Souřadnice mimo prohledávaný prostor se nahradí náhodnou hodnotou z rozsahu [lower, upper]."""
    outside = (U < problem.lower) | (U > problem.upper)
    return np.where(outside, rng.uniform(problem.lower, problem.upper, size=U.shape), U)


# ==========================================
# 3. HLAVNÍ TŘÍDA DIFERENCIÁLNÍ EVOLUCE
# ==========================================

class DifferentialEvolution:
    """
    Klasická DE (workflow MUTACE -> KŘÍŽENÍ -> SELEKCE) s volitelnou mutační strategií
    a volitelnou samoadaptací parametrů F a CR podle jDE (Brest et al., 2006).
    """

    def __init__(
        self,
        problem: Problem,
        mutation: Callable = mutation_rand1,
        dimension: int = 10,
        pop_size: int = 50,
        F: float = 0.5,
        CR: float = 0.8,
        max_evaluations: int = 10000,
        jde: bool = False,
        tau1: float = 0.1,
        tau2: float = 0.1,
        F_range: Tuple[float, float] = (0.1, 0.9),
        CR_range: Tuple[float, float] = (0.0, 1.0),
        seed: int = None,
    ):
        self.problem = problem
        self.mutation = mutation
        self.dimension = dimension
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_evaluations = max_evaluations
        self.jde = jde
        self.tau1, self.tau2 = tau1, tau2
        self.F_range, self.CR_range = F_range, CR_range
        self.rng = np.random.default_rng(seed)

    def run(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Spustí DE a sleduje nejlepší dosaženou hodnotu účelové funkce po každém ohodnocení.

        :return: (historie best-so-far délky max_evaluations, nejlepší hodnota, nejlepší řešení)
        """
        rng = self.rng
        NP = self.pop_size
        history = np.empty(self.max_evaluations)

        # Inicializace: pseudonáhodně s rovnoměrným rozložením v prohledávaném prostoru
        P = rng.uniform(self.problem.lower, self.problem.upper, size=(NP, self.dimension))
        fitnesses = self.problem.func(P)
        evals = NP
        history[:evals] = np.minimum.accumulate(fitnesses)

        # Každý jedinec má "zakódovanou" svou dvojici F a CR (u klasické DE jsou všechny stejné)
        F = np.full(NP, self.F)
        CR = np.full(NP, self.CR)

        while evals < self.max_evaluations:
            # jDE: před mutací/křížením se F a CR s pravděpodobností tau1/tau2 re-inicializují
            if self.jde:
                F_trial = np.where(rng.random(NP) < self.tau1, rng.uniform(*self.F_range, size=NP), F)
                CR_trial = np.where(rng.random(NP) < self.tau2, rng.uniform(*self.CR_range, size=NP), CR)
            else:
                F_trial, CR_trial = F, CR

            # 1. Mutace -> mutační (šumový) vektor v_i
            V = self.mutation(P, fitnesses, F_trial, rng)
            # 2. Křížení -> zkušební vektor u_i
            U = binomial_crossover(P, V, CR_trial, rng)
            U = repair_bounds(U, self.problem, rng)

            # Poslední generace se ohodnotí jen částečně, aby nebyl překročen rozpočet ohodnocení
            n = min(NP, self.max_evaluations - evals)
            U = U[:n]
            f_U = self.problem.func(U)
            history[evals:evals + n] = np.minimum(history[evals - 1], np.minimum.accumulate(f_U))
            evals += n

            # 3. Selekce: do nové generace postupuje lepší z dvojice (x_i, u_i), podmínka f(u_i) <= f(x_i)
            success = f_U <= fitnesses[:n]
            P[:n][success] = U[success]
            fitnesses[:n][success] = f_U[success]

            # jDE: úspěšné hodnoty F a CR přežívají s novým jedincem, jinak zůstávají původní
            if self.jde:
                F[:n][success] = F_trial[:n][success]
                CR[:n][success] = CR_trial[:n][success]

        best_idx = int(np.argmin(fitnesses))
        return history, float(fitnesses[best_idx]), P[best_idx].copy()


# ==========================================
# 4. KONFIGURACE VARIANT
# ==========================================

DIMENSION = 10
MAX_EVALS = 10000 * DIMENSION   # běžný rozpočet podle CEC benchmarků
RUNS = 30
POP_SIZE = 50                   # nízko-dimenzionální problém: NP >= 20

# Doporučené hodnoty ze zadání
VARIANTS = {
    'DE/rand/1/bin': dict(mutation=mutation_rand1, F=0.5, CR=0.8),
    'DE/best/1/bin': dict(mutation=mutation_best1, F=0.5, CR=0.8),
    'jDE (rand/1/bin)': dict(mutation=mutation_rand1, F=0.5, CR=0.9, jde=True, tau1=0.1, tau2=0.1,
                             F_range=(0.1, 0.9), CR_range=(0.0, 1.0)),
}


def run_many(problem: Problem, params: dict, runs: int = RUNS) -> Tuple[np.ndarray, np.ndarray]:
    """Provede `runs` nezávislých běhů. Vrací matici historií a pole finálních výsledků."""
    histories = np.empty((runs, MAX_EVALS))
    finals = np.empty(runs)
    for r in range(runs):
        de = DifferentialEvolution(problem, dimension=DIMENSION, pop_size=POP_SIZE,
                                   max_evaluations=MAX_EVALS, seed=r, **params)
        histories[r], finals[r], _ = de.run()
    return histories, finals


def stats(values: np.ndarray) -> Dict[str, float]:
    return {
        'best': np.min(values), 'worst': np.max(values), 'mean': np.mean(values),
        'median': np.median(values), 'std': np.std(values),
    }


# ==========================================
# 5. HLAVNÍ EXPERIMENT A VIZUALIZACE
# ==========================================

def run_experiment_suite(results_dir: str):
    """
    Pro každou funkci (Sphere, Rosenbrock, Schwefel) a každou variantu DE provede 30 běhů
    v 10D, vykreslí průměrné konvergenční křivky, boxploty a uloží statistiky
    (best / worst / mean / median / std).
    """
    print('=' * 78)
    print(f' DIFERENCIÁLNÍ EVOLUCE | D={DIMENSION}, {MAX_EVALS} ohodnocení, {RUNS} běhů, NP={POP_SIZE}')
    print('=' * 78)

    lines = []
    all_finals = {}

    for problem in PROBLEMS:
        header = f'{"Varianta":<20}{"Best":>12}{"Worst":>12}{"Mean":>12}{"Median":>12}{"Std":>12}'
        block = [f'\n--- {problem.name} [{problem.lower}, {problem.upper}]^{DIMENSION} ---', header, '-' * len(header)]

        plt.figure(figsize=(10, 6))
        finals_per_variant = {}
        for name, params in VARIANTS.items():
            histories, finals = run_many(problem, params)
            finals_per_variant[name] = finals
            s = stats(finals)
            block.append(f'{name:<20}{s["best"]:>12.4g}{s["worst"]:>12.4g}{s["mean"]:>12.4g}'
                         f'{s["median"]:>12.4g}{s["std"]:>12.4g}')
            plt.plot(np.arange(1, MAX_EVALS + 1), histories.mean(axis=0), label=name, linewidth=2)

        all_finals[problem.name] = finals_per_variant
        for line in block:
            print(line)
        lines.extend(block)

        plt.yscale('log')
        plt.title(f'Průměrná konvergence DE: {problem.name} (D={DIMENSION}, {RUNS} běhů)')
        plt.xlabel('Počet ohodnocení účelové funkce')
        plt.ylabel('Nejlepší nalezená hodnota f(x) (log)')
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend()
        filename = os.path.join(results_dir, f'{problem.name.lower()}_convergence.png')
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'  [+] Graf uložen do: {filename}')

    # Boxploty finálních výsledků všech variant
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(18, 6))
    for ax, problem in zip(axes, PROBLEMS):
        data = all_finals[problem.name]
        ax.boxplot(list(data.values()), tick_labels=list(data))
        ax.set_yscale('log')
        ax.set_title(f'{problem.name} – finální hodnoty ({RUNS} běhů)')
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.tick_params(axis='x', labelsize=9)
    filename = os.path.join(results_dir, 'boxplot_final.png')
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[+] Boxploty uloženy do: {filename}')

    path = os.path.join(results_dir, 'statistiky.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'D={DIMENSION}, {MAX_EVALS} ohodnocení, {RUNS} běhů, NP={POP_SIZE}\n')
        f.write('rand/1 a best/1: F=0.5, CR=0.8 | jDE: F0=0.5 (0.1-0.9), CR0=0.9 (0-1), tau1=tau2=0.1\n')
        f.write('\n'.join(lines) + '\n')
    print(f'[+] Statistiky uloženy do: {path}')


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    run_experiment_suite(results_dir)

# Při práci jsem využil Claude Opus 5.5.
# Pomáhal hlavně s převodem operátorů DE do vektorové podoby v NumPy (výběr indexů r1, r2, r3,
# binomiální křížení přes masku), s kontrolou implementace jDE podle přednášky a s úpravou
# grafů. Návrh experimentu, volbu testovacích funkcí a vyhodnocení výsledků jsem dělal sám.

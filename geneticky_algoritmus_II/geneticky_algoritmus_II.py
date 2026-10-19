import argparse
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
# 2. REPREZENTACE JEDINCŮ (ENCODINGS)
# ==========================================
# Každá reprezentace umí:
#   encode(X)  - převod reálných vektorů na genotyp (použito jen při inicializaci populace)
#   decode(G)  - převod genotypu na reálné vektory (pro ohodnocení účelovou funkcí)
#   crossover  - jednobodové křížení
#   mutate     - mutace

BITS_PER_DIM = 32


def _bits_to_bytes(G: np.ndarray, D: int) -> np.ndarray:
    """(n, D*32) bitů -> (n, D, 4) bytů (big-endian)."""
    return np.ascontiguousarray(np.packbits(G.reshape(G.shape[0], D, BITS_PER_DIM), axis=-1))


def _bytes_to_bits(B: np.ndarray) -> np.ndarray:
    """(n, D, 4) bytů -> (n, D*32) bitů."""
    return np.unpackbits(B, axis=-1).reshape(B.shape[0], -1)


class BinaryRepresentation:
    """
    Společný základ binárních reprezentací: 32 bitů na dimenzi, jednobodové křížení
    na libovolném bitu řetězce a bitová mutace (inverze bitu s pravděpodobností p_mutation).
    Dekódovaná hodnota mimo prohledávaný prostor se ořízne na jeho hranici.
    """
    name = 'binary'

    def __init__(self, problem: Problem, dimension: int, p_mutation: float):
        self.problem = problem
        self.D = dimension
        self.p_mutation = p_mutation

    def crossover(self, P1: np.ndarray, P2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        length = P1.shape[1]
        points = rng.integers(1, length, size=P1.shape[0])
        # mask = True na pozicích před bodem křížení -> první části rodičů se zamění
        mask = np.arange(length) < points[:, None]
        return np.where(mask, P2, P1), np.where(mask, P1, P2)

    def mutate(self, G: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        flips = rng.random(G.shape) < self.p_mutation
        return G ^ flips.astype(np.uint8)

    def decode(self, G: np.ndarray) -> np.ndarray:
        X = self._decode_raw(G)
        return np.clip(X, self.problem.lower, self.problem.upper)

    def _decode_raw(self, G: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def encode(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IEEE754Representation(BinaryRepresentation):
    """
    IEEE 754 single precision (float32): 1 bit znaménko, 8 bitů exponent, 23 bitů mantisa.
    Mutace bitu v exponentu mění hodnotu řádově (např. 3.2 -> 1e+38), NaN/Inf se nahradí hranicí.
    """
    name = 'Binární IEEE 754'

    def encode(self, X: np.ndarray) -> np.ndarray:
        B = X.astype('>f4').view(np.uint8).reshape(X.shape[0], self.D, 4)
        return _bytes_to_bits(B)

    def _decode_raw(self, G: np.ndarray) -> np.ndarray:
        with np.errstate(invalid='ignore', over='ignore'):  # NaN/Inf jsou platné bitové vzory float32
            X = _bits_to_bytes(G, self.D).view('>f4').reshape(G.shape[0], self.D).astype(np.float64)
        return np.nan_to_num(X, nan=self.problem.upper, posinf=self.problem.upper, neginf=self.problem.lower)


class FixedPointRepresentation(BinaryRepresentation):
    """
    Fixed-point (dvojkový doplněk): 1 bit znaménko + int_bits pro celou část + zbytek pro desetinnou část.
    Počet bitů celé části se volí tak, aby pokryl rozsah funkce (např. Schwefel ±500 -> 9 bitů).
    """
    name = 'Binární Fixed-point'

    def __init__(self, problem: Problem, dimension: int, p_mutation: float):
        super().__init__(problem, dimension, p_mutation)
        max_abs = max(abs(problem.lower), abs(problem.upper))
        self.int_bits = max(0, int(np.ceil(np.log2(max_abs))))
        self.frac_bits = BITS_PER_DIM - 1 - self.int_bits
        self.scale = 2.0 ** self.frac_bits

    def encode(self, X: np.ndarray) -> np.ndarray:
        Q = np.clip(np.round(X * self.scale), -2 ** 31, 2 ** 31 - 1).astype('>i4')
        return _bytes_to_bits(Q.view(np.uint8).reshape(X.shape[0], self.D, 4))

    def _decode_raw(self, G: np.ndarray) -> np.ndarray:
        Q = _bits_to_bytes(G, self.D).view('>i4').reshape(G.shape[0], self.D)
        return Q.astype(np.float64) / self.scale


class BCDRepresentation(BinaryRepresentation):
    """
    Binary Coded Decimal: 32 bitů = 8 nibblů (4 bity). První nibble je znaménko
    (jako u packed BCD: 0xC = +, 0xD = -, rozhoduje nejnižší bit), zbylých 7 nibblů jsou
    desítkové číslice s pevnou desetinnou čárkou (např. Schwefel: 3 číslice celá část, 4 desetinná).
    Neplatné nibbly 10-15 (vzniklé mutací/křížením) se interpretují jako 9.
    """
    name = 'Binární BCD'
    DIGITS = BITS_PER_DIM // 4 - 1  # 7 číslic + 1 znaménkový nibble

    def __init__(self, problem: Problem, dimension: int, p_mutation: float):
        super().__init__(problem, dimension, p_mutation)
        max_abs = max(abs(problem.lower), abs(problem.upper))
        self.int_digits = len(str(int(max_abs)))
        self.frac_digits = self.DIGITS - self.int_digits
        self.weights = 10 ** np.arange(self.DIGITS - 1, -1, -1, dtype=np.int64)

    def encode(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        magnitude = np.clip(np.round(np.abs(X) * 10 ** self.frac_digits), 0, 10 ** self.DIGITS - 1).astype(np.int64)
        digits = (magnitude[..., None] // self.weights) % 10                    # (n, D, 7)
        sign = np.where(X < 0, 0xD, 0xC)[..., None]                             # (n, D, 1)
        nibbles = np.concatenate([sign, digits], axis=-1).astype(np.uint8)      # (n, D, 8)
        bits = (nibbles[..., None] >> np.array([3, 2, 1, 0], dtype=np.uint8)) & 1
        return bits.reshape(n, -1).astype(np.uint8)

    def _decode_raw(self, G: np.ndarray) -> np.ndarray:
        nibbles = G.reshape(G.shape[0], self.D, self.DIGITS + 1, 4) @ np.array([8, 4, 2, 1])
        sign = np.where(nibbles[..., 0] & 1, -1.0, 1.0)
        digits = np.minimum(nibbles[..., 1:], 9)
        return sign * (digits @ self.weights) / 10 ** self.frac_digits


class RealRepresentation:
    """
    Real-valued GA: jedinec je přímo vektor reálných čísel.
    Křížení: jednobodové mezi dimenzemi (bod křížení leží na hranici mezi dvěma souřadnicemi).
    Mutace: každá souřadnice je s pravděpodobností p_mutation změněna
      - 'gauss'   : x + N(0, sigma^2), sigma je zadána jako podíl šířky rozsahu (sigma_rel * (ub - lb))
      - 'uniform' : nahrazena náhodnou hodnotou z celého rozsahu U(lb, ub)
    """

    def __init__(self, problem: Problem, dimension: int, p_mutation: float,
                 mutation: str = 'gauss', sigma_rel: float = 0.1):
        self.problem = problem
        self.D = dimension
        self.p_mutation = p_mutation
        self.mutation = mutation
        self.sigma = sigma_rel * (problem.upper - problem.lower)
        self.name = 'Real + Gauss mutace' if mutation == 'gauss' else 'Real + uniformní mutace'

    def encode(self, X: np.ndarray) -> np.ndarray:
        return X.copy()

    def decode(self, G: np.ndarray) -> np.ndarray:
        return G

    def crossover(self, P1: np.ndarray, P2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        points = rng.integers(1, self.D, size=P1.shape[0])
        mask = np.arange(self.D) < points[:, None]
        return np.where(mask, P2, P1), np.where(mask, P1, P2)

    def mutate(self, G: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mask = rng.random(G.shape) < self.p_mutation
        if self.mutation == 'gauss':
            mutated = G + rng.normal(0.0, self.sigma, size=G.shape)
            mutated = np.clip(mutated, self.problem.lower, self.problem.upper)
        else:
            mutated = rng.uniform(self.problem.lower, self.problem.upper, size=G.shape)
        return np.where(mask, mutated, G)


# ==========================================
# 3. OPERÁTORY SELEKCE (SELECTION OPERATORS) - minimalizace
# ==========================================

def rank_selection(fitnesses: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """
    Pořadová selekce: nejhorší jedinec má váhu 1, nejlepší váhu N.
    Vrací indexy vybraných jedinců.
    """
    n = len(fitnesses)
    order = np.argsort(fitnesses)[::-1]          # od nejhoršího (největší f) po nejlepšího
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    return rng.choice(n, size=count, p=ranks / ranks.sum())


def roulette_selection(fitnesses: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """
    Ruletová selekce pro minimalizaci: váha jedince = (f_max - f) + epsilon.
    """
    weights = fitnesses.max() - fitnesses + 1e-12
    return rng.choice(len(fitnesses), size=count, p=weights / weights.sum())


# ==========================================
# 4. HLAVNÍ TŘÍDA GENETICKÉHO ALGORITMU
# ==========================================

class GeneticAlgorithm:
    def __init__(
        self,
        problem: Problem,
        representation,
        dimension: int = 10,
        pop_size: int = 100,
        elitism_ratio: float = 0.1,
        selection_type: str = 'rank',  # 'roulette' nebo 'rank'
        max_evaluations: int = 10000,
        seed: int = None,
    ):
        self.problem = problem
        self.rep = representation
        self.dimension = dimension
        self.pop_size = pop_size
        self.elite_count = max(1, int(pop_size * elitism_ratio))
        self.select = roulette_selection if selection_type == 'roulette' else rank_selection
        self.max_evaluations = max_evaluations
        self.rng = np.random.default_rng(seed)

    def run(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Spustí GA a sleduje nejlepší dosaženou hodnotu účelové funkce po každém ohodnocení.

        :return: (historie best-so-far délky max_evaluations, nejlepší hodnota, nejlepší řešení)
        """
        rng = self.rng
        history = np.empty(self.max_evaluations)

        # Inicializace: rovnoměrně náhodné body v prohledávaném prostoru, zakódované do genotypu.
        # Všechny varianty tak startují ze stejně rozložené populace.
        X0 = rng.uniform(self.problem.lower, self.problem.upper, size=(self.pop_size, self.dimension))
        population = self.rep.encode(X0)
        phenotypes = self.rep.decode(population)
        fitnesses = self.problem.func(phenotypes)

        evals = len(fitnesses)
        history[:evals] = np.minimum.accumulate(fitnesses)
        best_idx = int(np.argmin(fitnesses))
        best_value, best_x = fitnesses[best_idx], phenotypes[best_idx].copy()

        while evals < self.max_evaluations:
            # 1. Elitismus: nejlepší jedinci přecházejí do nové populace beze změny
            elite_idx = np.argsort(fitnesses)[:self.elite_count]

            # 2. Výběr rodičů, křížení a mutace pro zbytek populace
            n_children = min(self.pop_size - self.elite_count, self.max_evaluations - evals)
            n_pairs = (n_children + 1) // 2
            parents = self.select(fitnesses, 2 * n_pairs, rng)
            c1, c2 = self.rep.crossover(population[parents[0::2]], population[parents[1::2]], rng)

            children = np.empty((2 * n_pairs,) + population.shape[1:], dtype=population.dtype)
            children[0::2], children[1::2] = c1, c2
            children = self.rep.mutate(children[:n_children], rng)

            # 3. Ohodnocení potomků
            child_x = self.rep.decode(children)
            child_f = self.problem.func(child_x)
            history[evals:evals + n_children] = np.minimum(best_value, np.minimum.accumulate(child_f))
            evals += n_children

            i = int(np.argmin(child_f))
            if child_f[i] < best_value:
                best_value, best_x = child_f[i], child_x[i].copy()

            population = np.concatenate([population[elite_idx], children])
            fitnesses = np.concatenate([fitnesses[elite_idx], child_f])

        return history, float(best_value), best_x


# ==========================================
# 5. KONFIGURACE VARIANT
# ==========================================

DIMENSION = 10
MAX_EVALS = 10000
RUNS = 10
POP_SIZE = 100
ELITISM = 0.1
SELECTION = 'rank'

# Nastavení vybrané na základě ladění (viz --tune a results/ladeni_parametru.txt)
BINARY_P_MUTATION = 1.0 / (DIMENSION * BITS_PER_DIM)   # ~0.31 % na bit = v průměru 1 bit na jedince
GAUSS_P_MUTATION = 0.1
GAUSS_SIGMA_REL = 0.01                                  # sigma = 1 % šířky rozsahu
UNIFORM_P_MUTATION = 0.05


def build_variants(problem: Problem) -> Dict[str, object]:
    D = DIMENSION
    return {
        'Binární IEEE 754': IEEE754Representation(problem, D, BINARY_P_MUTATION),
        'Binární Fixed-point': FixedPointRepresentation(problem, D, BINARY_P_MUTATION),
        'Binární BCD': BCDRepresentation(problem, D, BINARY_P_MUTATION),
        'Real + Gauss mutace': RealRepresentation(problem, D, GAUSS_P_MUTATION, 'gauss', GAUSS_SIGMA_REL),
        'Real + uniformní mutace': RealRepresentation(problem, D, UNIFORM_P_MUTATION, 'uniform'),
    }


def run_many(problem: Problem, representation, runs: int = RUNS, seed_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Provede `runs` nezávislých běhů. Vrací matici historií a pole finálních výsledků."""
    histories = np.empty((runs, MAX_EVALS))
    finals = np.empty(runs)
    for r in range(runs):
        ga = GeneticAlgorithm(problem, representation, DIMENSION, POP_SIZE, ELITISM, SELECTION,
                              MAX_EVALS, seed=seed_offset + r)
        histories[r], finals[r], _ = ga.run()
    return histories, finals


def stats(values: np.ndarray) -> Dict[str, float]:
    return {
        'best': np.min(values), 'worst': np.max(values), 'mean': np.mean(values),
        'median': np.median(values), 'std': np.std(values),
    }


# ==========================================
# 6. LADĚNÍ PARAMETRŮ (PARAMETER TUNING)
# ==========================================

def run_parameter_tuning(results_dir: str):
    """
    Projde mřížku parametrů mutace pro každou variantu a vypíše medián finální hodnoty (10 běhů).
    Používá jiná semínka než hlavní experiment, aby výběr parametrů neovlivnil vyhodnocení.
    """
    L = DIMENSION * BITS_PER_DIM
    grids = {
        'Binární (p_mut na bit)': (
            [0.5 / L, 1.0 / L, 2.0 / L, 0.01],
            lambda pr, p: [IEEE754Representation(pr, DIMENSION, p),
                           FixedPointRepresentation(pr, DIMENSION, p),
                           BCDRepresentation(pr, DIMENSION, p)]),
        'Real + uniformní (p_mut na dimenzi)': (
            [0.02, 0.05, 0.1, 0.2],
            lambda pr, p: [RealRepresentation(pr, DIMENSION, p, 'uniform')]),
    }
    sigmas = [0.001, 0.01, 0.05, 0.1, 0.2]
    gauss_p = [0.05, 0.1, 0.2, 0.5]

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    out('LADĚNÍ PARAMETRŮ (medián finální hodnoty z 10 běhů, D=10, 10000 ohodnocení)')
    for problem in PROBLEMS:
        out(f'\n=== {problem.name} ===')
        for title, (values, make) in grids.items():
            out(f'\n{title}')
            for p in values:
                row = []
                for rep in make(problem, p):
                    _, finals = run_many(problem, rep, seed_offset=1000)
                    row.append(f'{rep.name}: {np.median(finals):.4g}')
                out(f'  p_mut={p:.4f} | ' + ' | '.join(row))

        out('\nReal + Gauss (řádky: sigma jako podíl rozsahu, sloupce: p_mut na dimenzi)')
        out('  sigma\\p  ' + ''.join(f'{p:>12}' for p in gauss_p))
        for s in sigmas:
            row = []
            for p in gauss_p:
                _, finals = run_many(problem, RealRepresentation(problem, DIMENSION, p, 'gauss', s), seed_offset=1000)
                row.append(f'{np.median(finals):>12.4g}')
            out(f'  {s:<8} ' + ''.join(row))

    path = os.path.join(results_dir, 'ladeni_parametru.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n[+] Ladění uloženo do: {path}')


# ==========================================
# 7. HLAVNÍ EXPERIMENT A VIZUALIZACE
# ==========================================

def run_experiment_suite(results_dir: str):
    """
    Pro každou funkci (Sphere, Rosenbrock, Schwefel) a každou variantu GA provede 10 běhů
    po 10000 ohodnoceních v 10D, vykreslí průměrné konvergenční křivky, boxploty
    a uloží statistiky (best / worst / mean / median / std).
    """
    print('=' * 78)
    print(f' GA V REÁLNÉ DOMÉNĚ | D={DIMENSION}, {MAX_EVALS} ohodnocení, {RUNS} běhů, '
          f'pop={POP_SIZE}, elitismus={ELITISM:.0%}, selekce={SELECTION}')
    print('=' * 78)

    lines = []
    all_finals = {}

    for problem in PROBLEMS:
        variants = build_variants(problem)
        header = f'{"Varianta":<26}{"Best":>12}{"Worst":>12}{"Mean":>12}{"Median":>12}{"Std":>12}'
        block = [f'\n--- {problem.name} [{problem.lower}, {problem.upper}]^{DIMENSION} ---', header, '-' * len(header)]

        plt.figure(figsize=(10, 6))
        finals_per_variant = {}
        for name, rep in variants.items():
            histories, finals = run_many(problem, rep)
            finals_per_variant[name] = finals
            s = stats(finals)
            block.append(f'{name:<26}{s["best"]:>12.4g}{s["worst"]:>12.4g}{s["mean"]:>12.4g}'
                         f'{s["median"]:>12.4g}{s["std"]:>12.4g}')
            plt.plot(np.arange(1, MAX_EVALS + 1), histories.mean(axis=0), label=name, linewidth=2)

        all_finals[problem.name] = finals_per_variant
        for line in block:
            print(line)
        lines.extend(block)

        plt.yscale('log')
        plt.title(f'Průměrná konvergence: {problem.name} (D={DIMENSION}, {RUNS} běhů)')
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
        ax.boxplot(list(data.values()), tick_labels=[n.replace(' + ', '\n').replace('Binární ', 'Binární\n') for n in data])
        ax.set_yscale('log')
        ax.set_title(f'{problem.name} – finální hodnoty ({RUNS} běhů)')
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.tick_params(axis='x', labelsize=8)
    filename = os.path.join(results_dir, 'boxplot_final.png')
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[+] Boxploty uloženy do: {filename}')

    path = os.path.join(results_dir, 'statistiky.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'D={DIMENSION}, {MAX_EVALS} ohodnocení, {RUNS} běhů, pop={POP_SIZE}, '
                f'elitismus={ELITISM:.0%}, selekce={SELECTION}\n')
        f.write(f'Binární p_mut={BINARY_P_MUTATION:.5f}/bit, Gauss p_mut={GAUSS_P_MUTATION}, '
                f'sigma={GAUSS_SIGMA_REL} * rozsah, uniformní p_mut={UNIFORM_P_MUTATION}\n')
        f.write('\n'.join(lines) + '\n')
    print(f'[+] Statistiky uloženy do: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genetický algoritmus v reálné doméně')
    parser.add_argument('--tune', action='store_true', help='spustit i ladění parametrů mutace')
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    if args.tune:
        run_parameter_tuning(results_dir)
    run_experiment_suite(results_dir)

"""
Při vypracování úkolu jsem využil AI (Claude Opus 5.5).
Nejprve jsem převzal zdrojový kód z předchozího úkolu a nechal si ho od AI vysvětlit, 
aby mi pomohl pochopit jak funguje a poté jsem ho za pomoci AI rozšířil o nové zadání.

IEEE na Sphere: výsledek vyšel nečekaně dobře, ale je to spíš vlastnost formátu než síla algoritmu. Float32 umí kolem nuly vyjádřit hodnoty až kolem 1e-45, protože čísla jsou tam nejhustší. Minimum Sphere leží přesně v nule, takže GA jen „zmenšuje exponent“.
Fixed-point a BCD: chovají se velmi podobně. Mutace bitu má náhodný dopad: nejnižší bit změní hodnotu nepatrně, nejvyšší bit o polovinu rozsahu. Navíc se bod křížení může trefit doprostřed čísla. Algoritmus tak sice najde přibližnou oblast, ale doladit řešení nedokáže. U BCD se navíc ztrácí část kombinací bitů na neplatné číslice.
Real + Gauss: dělá malé, plynulé kroky v okolí rodiče, proto je nejlepší na Rosenbrocku, kde se postupuje úzkým údolím. Na Schwefelu má ale velký rozptyl výsledků (od 0.08 do 770): když populace uvízne v lokálním minimu, malá sigma ji z něj nedostane.
Real + uniformní: na Schwefelu jasně vyhrává. Funkce je separabilní, takže stačí v každé dimenzi zvlášť najít správnou hodnotu, a skok kamkoli v rozsahu spolu s křížením mezi dimenzemi přesně tohle umožňuje. Na Sphere a Rosenbrocku je ale nejhorší, protože náhodným skokem se k minimu nedá jemně přiblížit.
Rosenbrock: v 10 000 ohodnoceních nezkonvergovala žádná varianta. Je to nejtěžší funkce pro GA, protože proměnné spolu souvisí a jednobodové křížení tuhle závislost rozbíjí.
"""

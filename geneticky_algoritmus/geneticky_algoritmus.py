import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import os

# ==========================================
# 1. DEFINICE ÚČELOVÝCH FUNKCÍ (PROBLEM DEFINITIONS)
# ==========================================

def one_max(individual: np.ndarray) -> int:
    """
    Účelová funkce One-Max: Spočítá celkový počet jedniček v binárním řetězci.
    
    :param individual: Binární řetězec (1D numpy pole 0 a 1)
    :return: Počet jedniček
    """
    return int(np.sum(individual))


def leading_ones(individual: np.ndarray) -> int:
    """
    Účelová funkce Leading-Ones: Spočítá délku souvislého řetězce jedniček z levé strany.
    
    :param individual: Binární řetězec (1D numpy pole 0 a 1)
    :return: Počet po sobě jdoucích jedniček od prvního bitu
    """
    count = 0
    for bit in individual:
        if bit == 1:
            count += 1
        else:
            break
    return count


# ==========================================
# 2. OPERÁTORY SELEKCE (SELECTION OPERATORS)
# ==========================================

def roulette_selection(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """
    Ruletová selekce (Roulette Wheel Selection): Pravděpodobnost výběru je přímo
    úměrná hodnotě fitness daného jedince.
    """
    total_fitness = sum(fitnesses)
    
    # Ošetření případu, kdy je celková fitness 0
    if total_fitness == 0:
        return population[random.randint(0, len(population) - 1)].copy()
        
    probabilities = [f / total_fitness for f in fitnesses]
    # Výběr jednoho jedince na základě pravděpodobností
    selected_idx = np.random.choice(len(population), p=probabilities)
    return population[selected_idx].copy()


def rank_selection(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """
    Pořadová selekce (Rank Selection): Populace je seřazena podle fitness a pravděpodobnost
    výběru závisí na pořadí (ranku), čímž se snižuje riziko předčasné konvergence.
    """
    pop_size = len(population)
    
    # Získání indexů seřazených od nejhoršího po nejlepšího jedince
    sorted_indices = np.argsort(fitnesses)
    
    # Přiřazení vah podle pořadí (1 pro nejhoršího, pop_size pro nejlepšího)
    ranks = np.arange(1, pop_size + 1)
    total_rank_sum = np.sum(ranks)
    probabilities = ranks / total_rank_sum
    
    # Výběr indexu podle pořadí
    selected_rank_idx = np.random.choice(pop_size, p=probabilities)
    actual_idx = sorted_indices[selected_rank_idx]
    
    return population[actual_idx].copy()


# ==========================================
# 3. GENETICKÉ OPERÁTORY (CROSSOVER & MUTATION)
# ==========================================

def single_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jednobodové křížení (Single-point crossover):
    Náhodně vygeneruje místo křížení, rozdělené části rodičů zamění.
    """
    length = len(parent1)
    if length <= 1:
        return parent1.copy(), parent2.copy()
        
    # Náhodný bod křížení mezi indexem 1 a length-1
    crossover_point = random.randint(1, length - 1)
    
    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    
    return offspring1, offspring2


def bit_flip_mutation(individual: np.ndarray, p_mutation: float) -> np.ndarray:
    """
    Jednobodová / bitová mutace: Prochází jedince bit po bitu a s pravděpodobností p_mutation
    provede inverzi (0 -> 1, 1 -> 0).
    """
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < p_mutation:
            mutated[i] = 1 - mutated[i]  # Inverze bitu
    return mutated


# ==========================================
# 4. HLAVNÍ TŘÍDA GENETICKÉHO ALGORITMU
# ==========================================

class GeneticAlgorithm:
    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], int],
        dimension: int,
        pop_size: int = 50,
        elitism_ratio: float = 0.1,
        selection_type: str = 'rank',  # 'roulette' nebo 'rank'
        p_mutation: float = 0.01,
        max_evaluations: int = 1000
    ):
        self.fitness_func = fitness_func
        self.dimension = dimension
        self.pop_size = pop_size
        self.elitism_ratio = elitism_ratio
        self.selection_type = selection_type
        self.p_mutation = p_mutation
        self.max_evaluations = max_evaluations
        
        # Počet elitních jedinců
        self.elite_count = max(1, int(self.pop_size * self.elitism_ratio))

    def run(self) -> Tuple[List[float], float]:
        """
        Spustí genetický algoritmus a sleduje průběh nejlepší fitness vzhledem k počtu ohodnocení.
        
        :return: (historie nejlepší fitness po každém ohodnocení, konečná nejlepší fitness)
        """
        evaluations_used = 0
        
        # Inicilizace populace (náhodné binární řetězce)
        population = [np.random.randint(0, 2, size=self.dimension) for _ in range(self.pop_size)]
        
        # Ohodnocení počáteční populace
        fitnesses = []
        for ind in population:
            fitnesses.append(self.fitness_func(ind))
            evaluations_used += 1
            if evaluations_used >= self.max_evaluations:
                break

        # Historie vývoje nejlepší fitness
        best_fitness_history = []
        current_best = max(fitnesses)
        
        # Doplníme historii pro už provedená ohodnocení
        for _ in range(len(fitnesses)):
            best_fitness_history.append(current_best)

        # Hlavní evoluční cyklus
        while evaluations_used < self.max_evaluations:
            # 1. Seřazení populace podle fitness (sestupně)
            sorted_indices = np.argsort(fitnesses)[::-1]
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitnesses = [fitnesses[i] for i in sorted_indices]

            new_population = []
            new_fitnesses = []

            # 2. Elitismus: Přenesení nejlepších jedinců bez úpravy
            for i in range(self.elite_count):
                new_population.append(sorted_population[i].copy())
                new_fitnesses.append(sorted_fitnesses[i])

            # 3. Tvorba zbytku nové populace pomocí křížení a mutace
            while len(new_population) < self.pop_size and evaluations_used < self.max_evaluations:
                # Výběr rodičů podle zvolené selekce
                if self.selection_type == 'roulette':
                    parent1 = roulette_selection(sorted_population, sorted_fitnesses)
                    parent2 = roulette_selection(sorted_population, sorted_fitnesses)
                else:  # 'rank'
                    parent1 = rank_selection(sorted_population, sorted_fitnesses)
                    parent2 = rank_selection(sorted_population, sorted_fitnesses)

                # Křížení
                offspring1, offspring2 = single_point_crossover(parent1, parent2)

                # Mutace
                offspring1 = bit_flip_mutation(offspring1, self.p_mutation)
                offspring2 = bit_flip_mutation(offspring2, self.p_mutation)

                # Přidání prvního potomka a vyhodnocení
                new_population.append(offspring1)
                fit1 = self.fitness_func(offspring1)
                new_fitnesses.append(fit1)
                evaluations_used += 1
                
                if fit1 > current_best:
                    current_best = fit1
                best_fitness_history.append(current_best)

                # Přidání druhého potomka, pokud je ještě místo a nepřekročili jsme limit ohodnocení
                if len(new_population) < self.pop_size and evaluations_used < self.max_evaluations:
                    new_population.append(offspring2)
                    fit2 = self.fitness_func(offspring2)
                    new_fitnesses.append(fit2)
                    evaluations_used += 1
                    
                    if fit2 > current_best:
                        current_best = fit2
                    best_fitness_history.append(current_best)

            population = new_population
            fitnesses = new_fitnesses

        # Zajištění, že historie má přesně délku max_evaluations
        while len(best_fitness_history) < self.max_evaluations:
            best_fitness_history.append(current_best)

        return best_fitness_history[:self.max_evaluations], current_best


# ==========================================
# 5. EXECUTION & EXPERIMENTATION ENGINE
# ==========================================

def run_experiment_suite():
    """
    Spustí kompletní sadu experimentů podle zadání:
    - Úlohy: One-Max, Leading-Ones
    - Dimenze (D): 10, 30, 100
    - Limit ohodnocení: 100 * D
    - Opakování: 10 nezávislých Běhů
    - Generuje statistiky a průměrný konvergenční graf.
    """
    dimensions = [10, 30, 100]
    runs_per_exp = 10
    problems = [
        ('One-Max', one_max),
        ('Leading-Ones', leading_ones)
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print(" ZAHÁJENÍ EVOLUČNÍCH EXPERIMENTŮ ")
    print("=" * 70)

    for problem_name, fitness_func in problems:
        print(f"\n>>> TESTOVANÝ PROBLÉM: {problem_name.upper()} <<<")
        
        for D in dimensions:
            max_evals = 100 * D
            
            # Doporučené nastavení hyperparametrů na základě rozměru problému
            pop_size = 30 if D <= 30 else 50
            elitism_ratio = 0.15  # 15% elitismus
            selection_type = 'rank' # Pořadová selekce funguje stabilněji
            p_mutation = 1.0 / D    # Standardní mutační poměr 1/D (odpovídá cca 1-10% na řetězec)

            history_matrix = np.zeros((runs_per_exp, max_evals))
            final_best_scores = []

            for run in range(runs_per_exp):
                ga = GeneticAlgorithm(
                    fitness_func=fitness_func,
                    dimension=D,
                    pop_size=pop_size,
                    elitism_ratio=elitism_ratio,
                    selection_type=selection_type,
                    p_mutation=p_mutation,
                    max_evaluations=max_evals
                )
                history, final_best = ga.run()
                history_matrix[run, :] = history
                final_best_scores.append(final_best)

            # Výpočet průměrného průběhu konvergence
            mean_history = np.mean(history_matrix, axis=0)

            # Výpočet základních statistik finálních výsledků
            best_val = np.max(final_best_scores)
            worst_val = np.min(final_best_scores)
            mean_val = np.mean(final_best_scores)
            median_val = np.median(final_best_scores)
            std_val = np.std(final_best_scores)

            # Tisk statistik
            print(f"\n--- Statistika pro {problem_name} | D={D} (Max Evals: {max_evals}) ---")
            print(f"  Poptimum / Cílové maximum : {D}")
            print(f"  Nejlepší (Best)           : {best_val:.2f}")
            print(f"  Nejhorší (Worst)          : {worst_val:.2f}")
            print(f"  Průměr (Mean)             : {mean_val:.2f}")
            print(f"  Medián (Median)           : {median_val:.2f}")
            print(f"  Směrodatná odchylka (Std) : {std_val:.4f}")

            # Vykreslení a uložení konvergenčního grafu
            plt.figure(figsize=(9, 5))
            plt.plot(range(1, max_evals + 1), mean_history, label='Průměrná best-fitness', color='blue', linewidth=2)
            plt.axhline(y=D, color='red', linestyle='--', label=f'Teoretické optimum ({D})')
            plt.title(f'Průměrná konvergence: {problem_name} (D={D}, 10 Běhů)')
            plt.xlabel('Počet ohodnocení účelové funkce (Evaluations)')
            plt.ylabel('Fitness (Počet správných bitů)')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()

            filename = os.path.join(results_dir, f"{problem_name.lower().replace('-', '_')}_{D}D_convergence.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [+] Graf uložen do: {filename}")

    print("\n" + "=" * 70)
    print(" VŠECHNY EXPERIMENTY BYLY ÚSPĚŠNĚ DOKONČENY ")
    print(" Výsledné grafy naleznete ve složce 'results/'.")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment_suite()

"""
Při vypracování projektu jsem využil AI (Gemini 3.8 Flash).
Využil jsem ji při návrhu struktury kódu, generování komentářů a dokumentace.
Ladění parametrů i vyhodnocení výsledků jsem si procházel, testoval a dával dohromady sám.
"""    
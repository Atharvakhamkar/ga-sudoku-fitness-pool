"""
ga.py  (SUDOKU)
---------------
GA operators for a SINGLE island, using the ROW-PERMUTATION encoding that
makes GA Sudoku tractable:

  * Each ROW is always a permutation of 1..9 (givens fixed in place).
  * So ROWS are never broken -> the GA only needs to fix columns & boxes.
  * Crossover swaps whole rows (each row stays a valid permutation).
  * Mutation swaps two NON-GIVEN cells within a row (stays a permutation).
  * Givens are never moved or changed.

Fitness is delegated to the external pool (fitness_client.py). A solved
grid scores MAX_SCORE = 162.
"""

import random

GRID = 9


class SudokuGA:
    def __init__(self, puzzle, given_mask, pop_size, mutation_rate,
                 crossover_rate, tournament_k, elite_count,
                 hypermutation_factor=6, seed=None):
        self.puzzle = puzzle
        self.mask = given_mask              # True where cell is a given
        self.pop_size = pop_size
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_k = tournament_k
        self.elite_count = elite_count
        self.hypermutation_factor = hypermutation_factor
        self.rng = random.Random(seed)

        # per-row: which digits are missing, and which positions are free
        self.row_missing = []
        self.row_free_pos = []
        for r in range(GRID):
            row = puzzle[r * GRID:(r + 1) * GRID]
            present = {v for v in row if v != 0}
            missing = [d for d in range(1, 10) if d not in present]
            free = [c for c in range(GRID) if row[c] == 0]
            self.row_missing.append(missing)
            self.row_free_pos.append(free)

        self.population = [self._random_individual() for _ in range(pop_size)]
        self.fitnesses = [0] * pop_size

    # ---- represent an individual as 9 rows (each a list of 9 ints) ----
    def _random_individual(self):
        rows = []
        for r in range(GRID):
            row = self.puzzle[r * GRID:(r + 1) * GRID][:]
            missing = self.row_missing[r][:]
            self.rng.shuffle(missing)
            for pos, digit in zip(self.row_free_pos[r], missing):
                row[pos] = digit
            rows.append(row)
        return rows

    @staticmethod
    def flatten(individual):
        flat = []
        for row in individual:
            flat.extend(row)
        return flat

    # ---- selection ----
    def tournament_selection(self):
        best = self.rng.randrange(self.pop_size)
        for _ in range(self.tournament_k - 1):
            ch = self.rng.randrange(self.pop_size)
            if self.fitnesses[ch] > self.fitnesses[best]:
                best = ch
        return self.population[best]

    # ---- crossover: swap whole rows ----
    def crossover(self, p1, p2):
        if self.rng.random() > self.crossover_rate:
            return [r[:] for r in p1], [r[:] for r in p2]
        point = self.rng.randrange(1, GRID)
        c1 = [r[:] for r in p1[:point]] + [r[:] for r in p2[point:]]
        c2 = [r[:] for r in p2[:point]] + [r[:] for r in p1[point:]]
        return c1, c2

    # ---- mutation: swap two free cells within a row ----
    def mutate(self, individual):
        for r in range(GRID):
            free = self.row_free_pos[r]
            if len(free) < 2:
                continue
            if self.rng.random() < self.mutation_rate:
                a, b = self.rng.sample(free, 2)
                individual[r][a], individual[r][b] = (
                    individual[r][b], individual[r][a]
                )
        return individual

    # ---- build next generation ----
    def next_generation(self, stagnating=False):
        # Adaptive hypermutation: when stuck, raise mutation to escape the
        # local optimum. (Separate from the SCALER, which changes throughput.)
        self.mutation_rate = (
            self.base_mutation_rate * self.hypermutation_factor
            if stagnating else self.base_mutation_rate
        )

        ranked = sorted(range(self.pop_size),
                        key=lambda i: self.fitnesses[i], reverse=True)
        new_pop = [[r[:] for r in self.population[i]]
                   for i in ranked[:self.elite_count]]

        while len(new_pop) < self.pop_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            c1, c2 = self.crossover(p1, p2)
            new_pop.append(self.mutate(c1))
            if len(new_pop) < self.pop_size:
                new_pop.append(self.mutate(c2))

        self.population = new_pop


    def restart(self, keep=None):
        """
        Population RESTART (a.k.a. random immigrants / catastrophe): when the
        island is deeply stuck, keep the few best individuals and regenerate
        the rest from scratch. This injects massive diversity to escape a
        stubborn local optimum.
        """
        keep = self.elite_count if keep is None else keep
        ranked = sorted(range(self.pop_size),
                        key=lambda i: self.fitnesses[i], reverse=True)
        survivors = [[r[:] for r in self.population[i]] for i in ranked[:keep]]
        fresh = [self._random_individual()
                 for _ in range(self.pop_size - keep)]
        self.population = survivors + fresh

    def best_index(self):
        return max(range(self.pop_size), key=lambda i: self.fitnesses[i])

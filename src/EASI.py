import pandas as pd
import numpy as np
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.ensemble import *
from sklearn.neural_network import *
import math


class EASI():
    population_size = 200
    parents_size = 100
    children_size = 100
    tourn_size = 64
    max_generations = 250
    max_rounds = 1
    mut_rate = 0.5
    init_repeats = 100
    inner_repeats = 10

    def __init__(self, features, train_file, test_file, validate_file, model_key, N=50, thresh=1, stds=3):
        self.features = int(features)
        self.feature_names = pd.read_csv(train_file, index_col=0).columns
        self.train = pd.read_csv(train_file, index_col=0).values
        self.test = pd.read_csv(test_file, index_col=0).values
        self.validate = pd.read_csv(validate_file, index_col=0).values
        self.model = {'knnc': KNeighborsClassifier(),
                      'knnr': KNeighborsRegressor(),
                      'svmc': SVC(),
                      'svmr': SVR(),
                      'rfc': RandomForestClassifier(random_state=0),
                      'rfr': RandomForestRegressor(random_state=0),
                      'gbc': GradientBoostingClassifier(random_state=0),
                      'gbr': GradientBoostingRegressor(random_state=0),
                      'mlpc': MLPClassifier(random_state=0, max_iter=1000),
                      'mlpr': MLPRegressor(random_state=0, max_iter=1000)}[model_key]
        self.proxy_model = {'c': SVC(),
                            'r': SVR()}[model_key[-1]]
        self.thresh, self.N = float(thresh), int(N)
        self.stds = float(stds)
        self.test_reference = 0
        self.max_crowding = []
        self.index = np.arange(self.features)

    def calc_main_effects(self, n_repeats):
        main_effects = np.zeros(self.features)

        for i in range(self.features):
            main_effects[i] = self.not_permute(
                binary=np.array([1 if i == j else 0 for j in range(self.features)]),
                test=self.test.copy(),
                n_repeats=n_repeats,
                main_effects=np.zeros(self.features)
            )

        return main_effects

    def genetic_algorithm(self):
        self.fit_model(self.train)
        self.test_reference = self.not_permute(
            binary=np.zeros(self.features),
            test=self.test.copy(),
            n_repeats=self.init_repeats,
            main_effects=np.zeros(self.features)
        )
        self.main_effects = self.calc_main_effects(n_repeats=self.init_repeats)

        population = self.initialize()
        for generation in range(self.max_generations):
            population = self.evaluate(population=population)
            parents, fronts = self.nsga_ii(
                population=population,
                selection_size=self.parents_size)
            children = self.recombine(parents=parents)
            children = self.mutate(children=children)
            children = self.evaluate(population=children)
            population, fronts = self.nsga_ii(
                population=np.concatenate((population, children)),
                selection_size=self.population_size
            )

            try:
                self.max_crowding.append(np.max([individual['meta']['distance'] for individual in population if
                                                 individual['meta']['distance'] < math.inf]))
            except:
                self.max_crowding.append(0)

            self.display_dynamic(
                population=population,
                fronts=fronts,
                max_crowding=np.array(self.max_crowding),
                generation=generation
            )

            if self.crowding_stagnation(self.max_crowding, generation):
                break

        self.pareto = fronts[0]
        self.population = population
        self.importance = self.calc_importance(population=self.pareto)
        self.dependency = self.calc_dependancy(population=self.pareto)
        self.interactions = self.get_interactions(population=self.pareto)

    def display_dynamic(self, population, fronts, max_crowding, generation):
        print('generation max_test min_test max_size min_size crowding_std pareto_size')
        print(generation, np.max([individual['meta']['test'] for individual in population]),
              np.min([individual['meta']['test'] for individual in population]),
              np.max([individual['meta']['size'] for individual in population]),
              np.min([individual['meta']['size'] for individual in population]), np.std(max_crowding[-self.N:]),
              len(fronts[0]))
        print('#################################################')

    def display_static(self, importance, outliers):
        print('importance_mean importance_std importance_max outliers')
        print(np.mean(importance), np.std(importance), np.max(importance), len(outliers))
        print('#################################################')

    def fit_model(self, set):
        self.model.fit(set[:, 1:], set[:, 0])

    def predict_model(self, set):
        return self.model.predict(set[:, 1:])

    def initialize(self):
        return [{'meta': {'test': None,
                          'tests': [],
                          'validate': None,
                          'size': None,
                          'dominates': None,
                          'dominated': None,
                          'distance': None},
                 'data': np.random.randint(2, size=self.features).tolist()} for _ in range(self.population_size)]

    def evaluate(self, population):

        for individual in population:
            individual['meta']['size'] = np.sum(individual['data'])
            if individual['meta']['size'] <= 1:
                individual['meta']['test'], individual['meta']['size'] = -math.inf, math.inf
            else:
                individual['meta']['test'] = self.not_permute(
                    binary=np.array(individual['data']),
                    test=self.test.copy(),
                    n_repeats=self.inner_repeats,
                    main_effects=self.main_effects)

        return population

    def not_permute(self, binary, test, n_repeats, main_effects):
        X_test, y_test = test[:, 1:], test[:, 0]

        scores = []
        for _ in range(n_repeats):
            indices = np.where(binary == 1)[0]
            X_test[:, indices] = np.random.permutation(X_test[:, indices])
            scores.append(self.model.score(X_test, y_test))

        return np.mean(scores) - self.test_reference - np.sum(main_effects[np.where(binary == 1)])

    def nsga_ii(self, population, selection_size):

        selection = []
        j, k = 0, 0

        fronts = self.fast_nondominated_sort(population, len(population))
        fronts = self.crowding_distance_assignment(fronts)

        while len(selection) + len(fronts[j]) < selection_size:
            for individual in fronts[j]:
                selection.append(individual)
            j += 1

        fronts[j] = sorted(fronts[j], key=lambda individual: individual['meta']['distance'], reverse=True)

        while len(selection) < selection_size:
            selection.append(fronts[j][k])
            k += 1

        return selection, fronts

    def fast_nondominated_sort(self, population, num_fronts):
        fronts, k = [[] for _ in range(num_fronts)], 0

        for i in range(len(population)):
            population[i]['meta']['dominates'], population[i]['meta']['dominated'], population[i]['meta'][
                'distance'] = set(), 0, 0
            for j in range(len(population)):
                if population[i]['meta']['test'] > population[j]['meta']['test'] and population[i]['meta']['size'] < \
                        population[j]['meta']['size']:
                    population[i]['meta']['dominates'].add(j)
                if population[j]['meta']['test'] > population[i]['meta']['test'] and population[j]['meta']['size'] < \
                        population[i]['meta']['size']:
                    population[i]['meta']['dominated'] += 1
            if population[i]['meta']['dominated'] == 0:
                fronts[0].append(population[i])

        while len(fronts[k]) > 0:
            for i in range(len(fronts[k])):
                for j in fronts[k][i]['meta']['dominates']:
                    population[j]['meta']['dominated'] -= 1
                    if population[j]['meta']['dominated'] == 0:
                        fronts[k + 1].append(population[j])
            k += 1

        return fronts

    def crowding_distance_assignment(self, fronts):
        objectives = ['test', 'size']

        for front in fronts:
            if len(front) > 0:
                for objective in objectives:
                    front = sorted(front, key=lambda individual: individual['meta'][objective])
                    front[0]['meta']['distance'], front[-1]['meta']['distance'] = math.inf, math.inf
                    for i in range(2, len(front) - 1):
                        front[i]['meta']['distance'] += front[i + 1]['meta'][objective] - front[i - 1]['meta'][
                            objective]

        return fronts

    def recombine(self, parents):
        children = []

        np.random.shuffle(parents)

        for i in range(len(parents) - 1):
            children.append(self.one_point(parents[i], parents[i + 1]))

        children.append(self.one_point(parents[0], parents[-1]))

        return children

    def one_point(self, mother, father):
        child = {'meta': {'test': None,
                          'tests': [],
                          'validate': None,
                          'size': None,
                          'dominates': None,
                          'dominated': None,
                          'distance': None},
                 'index': self.index,
                 'data': None}

        i = np.random.randint(len(self.index))
        child['data'] = mother['data'][:i] + father['data'][i:]

        if np.sum(child['data']) <= 1:
            idx = np.random.choice(a=self.index, size=2, replace=False)
            child['data'][idx[0]], child['data'][idx[1]] = 1, 1
            return child
        else:
            return child

    def mutate(self, children):

        for i in range(self.children_size):

            if np.random.uniform() < self.mut_rate:
                j = np.random.randint(len(self.index))

                if children[i]['data'][j] == 1:
                    if np.sum(children[i]['data']) > 1:
                        children[i]['data'][j] = 0
                else:
                    children[i]['data'][j] = 1

        return children

    def crowding_stagnation(self, max_crowding, generation):

        std = np.array(max_crowding[-self.N:]).std()
        return std < self.thresh and generation > self.N

    def calc_feature_statistics(self, population):

        return self.calc_importance(population), self.calc_synergy(population), self.calc_dependancy(population)

    def calc_importance(self, population):
        importance = np.zeros(self.features, dtype=float)

        for individual in population:
            for j in range(len(individual['data'])):
                if individual['data'][j] == 1:
                    importance[individual['index'][j]] += individual['meta']['test'] / (
                        individual['meta']['size']) / len(population)

        return importance

    def get_interactions(self, population):
        interaction = {}

        for individual in population:
            try:
                interaction[tuple(self.feature_names[1:][np.where(np.array(individual['data']) == 1)])] += \
                    individual['meta']['test'] / (individual['meta']['size']) / len(population)
            except:
                interaction[tuple(self.feature_names[1:][np.where(np.array(individual['data']) == 1)])] = \
                    individual['meta']['test'] / (individual['meta']['size']) / len(population)

        return interaction

    def calc_dependancy(self, population):
        intersection = np.zeros((self.features, self.features), dtype=float)
        total = np.zeros((self.features, self.features), dtype=float)

        for individual in population:
            for i in range(len(individual['data'])):
                for j in range(len(individual['data'])):
                    if i == j:
                        continue

                    if individual['data'][i] == 1 and individual['data'][j] == 1:
                        intersection[individual['index'][i]][individual['index'][j]] += (individual['meta']['test']) / (
                                2 ** individual['meta']['size'] - individual['meta']['size'] - 1) / len(population)
                    if individual['data'][i] == 1:
                        total[individual['index'][i]][individual['index'][j]] += (individual['meta']['test']) / (
                                2 ** individual['meta']['size'] - individual['meta']['size'] - 1) / len(population)

        return intersection / total

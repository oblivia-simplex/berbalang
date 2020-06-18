import json
import joypy
import os
import seaborn as sns
import matplotlib.pyplot as plot
import pandas as pd
import gzip
import functools as ft

def load_population(path):
    print(f"Decompressing and deserializing {path}")
    try:
        with gzip.open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read population from {path}: {e}")

def fitness_of_creature(creature):
    return creature['fitness']['cached_scalar'] if 'fitness' in creature and \
            'cached_scalar' in creature['fitness'] else None

def generation_of_creature(creature):
    return creature['generation']

def mem_write_ratio_of_creature(creature):
    return creature['fitness']['scores']['mem_write_ratio']

def mem_ratio_written_of_creature(creature):
    return creature['fitness']['scores']['mem_ratio_written']

def code_coverage_of_creature(creature):
    return creature['fitness']['scores']['code_coverage']

def fitness_scores_of_population(population):
    """Assumes Weighted fitness scores, which have already been evaluated and which
    have cached their scalar values."""
    if type(population) is str:
        population = load_population(population)
    fitnesses = [p['fitness']['cached_scalar'] for p in population if 'fitness' in p and 'cached_scalar' in p['fitness']]
    return fitnesses

def generations_of_population(population):
    if type(population) is str:
        population = load_population(population)
    return [p['generation'] for p in population]

def files(directory):
    fs = [os.path.join(directory, f) for f in next(os.walk(directory))[2]]
    print(fs)
    return fs

def build_dataframe(path, c1_name, c2_name, f1, f2):
    population = load_population(path)
    col1 = [f1(c) for c in population]
    col2 = [f2(c) for c in population]
    d = {c1_name: col1, c2_name: col2}
    return pd.DataFrame(data=d)

def dataframe_for_dir(directory, c1_name, c2_name, f1, f2):
    return ft.reduce(pd.DataFrame.append, (build_dataframe(path, c1_name, c2_name, f1, f2) \
            for path in files(directory)))


def plot_hexbin(data, col_1, col_2):
    sns.jointplot(x=col_1, y=col_2, data=data, kind="hex", color="#ee1105")#color="#2cb391")
    return data

def plot_pleasures(data, col_1, col_2):
    fig, axes = joypy.joyplot(data, 
            by=col_1, 
            column=col_2, 
            range_style='own', 
            grid=False, 
            xlabels=False,
            linewidth=1, 
            legend=False, 
            title=f"{col_2} by {col_1}", 
            bins=40, 
            ylabels=False, 
            overlap=0.9, 
            fill=True, 
            kind="counts", 
            # The Unknown Pleasures colour scheme is set here:
            background='k', 
            linecolor='w', 
            color='k')
    #plot.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return data, fig, axes

def population_dirs(population_name, island=None):
    if island is None:
        return glob.glob(f"{population_name}/island_*/population")
    else:
        return glob.glob(f"{population_name}/island_{island}/population")
### 

def plot_fitness_by_generation(pop_name, island=None):
    pop_dirs = population_dirs(pop_name, island)
    data = ft.reduce(pd.DataFrame.append, [dataframe_for_dir(pop_dir, "generation", "fitness", generation_of_creature, fitness_of_creature) for pop_dir in pop_dirs])
    plot_hexbin(data, "generation", "fitness")
    return data


def unknown_pleasures_fitness_by_generation(pop_name, island=None):
    pop_dirs = population_dirs(pop_name, island)
    data = ft.reduce(pd.DataFrame.append, [dataframe_for_dir(pop_dir, "generation", "fitness", generation_of_creature, fitness_of_creature) for pop_dir in pop_dirs])
    min_fit = min(data['fitness'])
    max_fit = max(data['fitness'])
    left = max_fit * 1.1
    right = min_fit * 0.9
    fig, axes = joypy.joyplot(data, 
            by="generation", 
            column="fitness", 
            range_style='own', 
            grid=False, 
            xlabels=False,
            linewidth=1, 
            legend=False, 
            title="Fitness density by generation, for entry-slang population", 
            bins=40, 
            ylabels=False, 
            overlap=0.9, 
            fill=True, 
            kind="counts", 
            # The Unknown Pleasures colour scheme is set here:
            background='k', 
            linecolor='w', 
            color='k')
    plot.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for axis in axes[:-1]:
        axis.set_xlim(left, right)
    return data, fig, axes

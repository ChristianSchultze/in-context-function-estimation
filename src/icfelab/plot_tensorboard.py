import json
from csv import reader
from typing import Any, Tuple, List

import matplot2tikz
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

import requests

TAGS = ['train_loss_epoch',
        'val_loss']


# RUNS = [['bender_train_B/0/version_0']]
# RUNS = [['bender_train_A/0/version_0']]
RUNS = [['bender_train_A\\0\\version_0'], ['bender_train_A-small\\0\\version_0'], ['bender_train_A-large\\0\\version_1']]

plt.rcParams["figure.figsize"] = (30, 20)
plt.rcParams["font.size"] = 35
plt.rcParams['savefig.pad_inches'] = 0
plt.rcParams.update({'font.size': 40})


def get_data(tag: str, run: str) -> Tuple[ndarray, ndarray]:
    """
    Load tensorboard data from localhost
    :param tag: which data to load from a run
    :param run: which runs to load
    :return: Tuple with steps and data 1d ndarrays
    """
    url = f'http://localhost:6006/experiment/defaultExperimentId/data/plugin/scalars/scalars?tag={tag}&run={run}&format=csv'
    r = requests.get(url, allow_redirects=True)
    data_csv = reader(r.text.splitlines())
    data = np.array(list(data_csv)[1:], dtype=float)
    return data[:, 1], data[:, 2]


def get_timeseries(tag: str, runs: List[List[str]] = RUNS) -> Tuple[List[ndarray], List[ndarray]]:
    """
    Build up lists for each run containing all versions of that run.
    :param tag: tag of data that should be loaded
    :param runs: which runs to use
    :return: steps and data lists
    """
    data = []
    step_lists = []
    for run in runs:
        value_list = []
        for version in run:
            steps, values = get_data(tag, version)
            value_list.append(values)
        data.append(np.array(value_list))
        step_lists.append(np.array(steps, dtype=int))

    # custom step correction
    # step_lists[-1] = step_lists[-1]//2
    # step_lists[-2] = step_lists[-2]//2

    return step_lists, data


def average(data):
    avg_data = []
    for i in range(0, len(data), 3):
        avg_data.append(np.mean(data[i:i + 3], axis=0))
    return avg_data


STEPS, EPOCHS = get_timeseries('epoch')

def set_xticks(steps, epochs=EPOCHS[0][0].astype(int)):
    """Arange x ticks so that the units is epochs and not steps. Calculates step per value based on last epoch and
    last step. This only works if this does not change throughout training and versions."""
    number_epochs = epochs[-1]
    number_steps = steps[0][-1]
    step_per_epoch = number_steps // number_epochs
    epoch_tiks = 50
    plt.xticks(np.append(np.arange(0, number_steps, step_per_epoch * epoch_tiks)[:-1], steps[0][-1] + 1),
               np.arange(0, number_epochs + 1, epoch_tiks))

def set_xticks_multiple(steps, epochs=EPOCHS[0][0].astype(int)):
    """Arange x ticks so that the units is epochs and not steps. Calculates step per value based on last epoch and
    last step. This only works if this does not change throughout training and versions."""
    number_epochs = epochs[-1]
    number_steps = steps[-1]
    step_per_epoch = number_steps // number_epochs
    epoch_tiks = 50
    plt.xticks(np.append(np.arange(0, number_steps, step_per_epoch * epoch_tiks)[:-1], steps[-1] + 1),
               np.arange(0, number_epochs + 1, epoch_tiks))

def plot(steps, data, main_color, background_color, title, labels, tiks_name, ylabel, legend):
    """Plots timeseries with error bands"""

    for index, timeseries in enumerate(data):
        plt.plot(steps[index][0], timeseries[0].squeeze(), color=main_color[index], label=labels[index])
    plt.title(title)

    # set_xticks(steps, epochs)
    set_xticks(steps[0])

    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(loc=legend)

    plt.savefig(f"{tiks_name}.png")
    fig = plt.gcf()
    matplot2tikz.clean_figure()
    matplot2tikz.save(f"{tiks_name}.tex")
    plt.clf()
    # plt.show()

def plot_multiple(steps, data, main_color, background_color, title, labels, tiks_name, ylabel, legend):
    """Plots timeseries with error bands"""

    fig, ax = plt.subplots()

    for index, timeseries in enumerate(data):
        timeseries[timeseries>2] = 2
        ax.plot(steps[index], timeseries.squeeze(), color=main_color[index], label=labels[index])
    plt.title(title)

    # set_xticks(steps, epochs)
    set_xticks_multiple(steps[0])

    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    ax.grid()
    ax.legend(loc=legend)
    ax.set_ylim(0, 1.25)

    plt.savefig(f"{tiks_name}.png")
    fig = plt.gcf()
    matplot2tikz.clean_figure()
    matplot2tikz.save(f"{tiks_name}.tex")
    plt.clf()
    # plt.show()



def train_val_loss():
    step_list, data_list = [], []
    steps, data = get_timeseries('train_loss_step')
    step_list.append(steps)
    data_list.append(data)
    steps, data = get_timeseries('val_loss')
    step_list.append(steps)
    data_list.append(data)
    title = "Model B (2,5)"
    tiks_name = "final_loss_B"
    ylabel = "Loss"
    legend = "upper right"

    return step_list, data_list, title, tiks_name, ylabel, legend

def val_loss_only():
    steps, data = get_timeseries('val_loss')
    title = "Learnrate comparison"
    tiks_name = "final_dim"
    ylabel = "RMSE Loss"
    legend = "upper right"

    return steps, data, title, tiks_name, ylabel, legend


def graph():
    main_labels = ['256', '128', '512']
    # main_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    # background_color = ['lightsteelblue', 'peachpuff', 'palegreen', 'tab:red', 'tab:purple']
    main_color = ['tab:green', 'tab:blue', 'tab:orange']
    background_color = ['palegreen', 'lightsteelblue', 'peachpuff']

    steps, data, title, tiks_name, ylabel, legend = val_loss_only()

    plot_multiple(steps, data, main_color, background_color, title, main_labels, tiks_name, ylabel, legend)


if __name__ == "__main__":
    graph()

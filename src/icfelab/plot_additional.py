import json
import lzma

import matplot2tikz
from matplotlib import pyplot as plt
from scipy.stats import beta
import numpy as np


def plot_beta():
    sampled_values = []
    x = np.linspace(0, 1, 1000)
    pdf = beta.pdf(x, 2, 10)
    pdf_2 = beta.pdf(x, 2, 5)

    plt.plot(x, pdf, label="A(2,10)", color='blue')
    plt.plot(x, pdf_2, label="B(2,5)", color='orange')

    plt.title("")
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()
    matplot2tikz.clean_figure()
    matplot2tikz.save("beta.tex")

    plt.savefig("beta.png")
    plt.close()


def plot_sample_size_histogramm():
    with lzma.open("data/sample_sizes.lzma", "rb") as file:
        data = json.loads(file.read().decode("utf-8"))

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=range(5, 52), edgecolor='black', align='left')

    plt.title('Sample size histogram')
    plt.xlabel('Sample size')
    plt.ylabel('')
    plt.xticks(range(5, 51))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    fig = plt.gcf()
    matplot2tikz.clean_figure()
    matplot2tikz.save("sample_hist.tex")

    plt.savefig("sample_hist.png")
    plt.close()


plot_sample_size_histogramm()
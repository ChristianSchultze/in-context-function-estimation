import matplot2tikz
from matplotlib import pyplot as plt
from scipy.stats import beta
import numpy as np

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
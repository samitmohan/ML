# includes computations that are out of scope of numpy (add on top of numpy)
# matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
# plt.plot(x, np.sin(x))
plt.plot(x, np.sin(x), label=("sin(x)"), linestyle="", marker="o")
plt.xlim([2, 8])
plt.ylim([0, 0.75])
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Sine Wave")
plt.show()
# scatter plots
rng = np.random.RandomState(123)
x = rng.normal(size=500)
y = rng.normal(size=500)
plt.scatter(x, y)
plt.show()

# bar
means = [5, 8, 10]
stddevs = [0.2, 0.4, 0.5]
bar_labels = ["bar 1", "bar 2", "bar 3"]


# plot bars
x_pos = list(range(len(bar_labels)))
print(x_pos)
plt.bar(x_pos, means, yerr=stddevs)

plt.show()

# histograms
rng = np.random.RandomState(123)
x = rng.normal(0, 20, 1000)

# fixed bin size
bins = np.arange(-100, 100, 5)  # fixed bin size

plt.hist(x, bins=bins)
plt.show()

# subplots
x = range(11)
y = range(11)

fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

for row in ax:
    for col in row:
        col.plot(x, y)

plt.show()

# colors
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), color="blue", marker="^", linestyle="")
plt.show()
# saving
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))

plt.savefig("myplot.png", dpi=300)
plt.savefig("myplot.pdf")

plt.show()

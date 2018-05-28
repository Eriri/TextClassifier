import matplotlib.pyplot as plt


class drawer:
    def draw(self, title, xlabel, x, ylabel, y):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y)
        plt.show()

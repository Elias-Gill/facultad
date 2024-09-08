import numpy as np
from matplotlib import pyplot


def fA(x):
    if x < 1:
        return 0
    elif 1 <= x and x < 7:
        return (x - 1) / 6
    elif 1 <= x and x < 10:
        return (10 - x) / 3
    elif x >= 10:
        return 0


def fB(x):
    if x < 2:
        return 0
    elif 1 <= x and x < 2:
        return (x - 1) / 6
    elif 2 <= x and x < 3:
        return x - 2
    elif 3 <= x and x < 4:
        return 1
    elif 4 <= x and x < 6:
        return (6 - x) / 2
    elif x > 6:
        return 0


def fC(x):
    if x < 1:
        return 0
    elif 1 <= x and x < 2.2:
        return (x - 1) / 6
    elif 2.2 <= x and x < 3:
        return x - 2
    elif 3 <= x and x < 4:
        return 1
    elif 4 <= x and x < 4.75:
        return (6 - x) / 2
    elif 4.75 <= x and x < 7:
        return (x - 1) / 6
    elif 7 <= x and x < 10:
        return (10 - x) / 3
    elif x >= 10:
        return 0


x = np.linspace(1, 10, 100)
pyplot.plot(x, [fA(i) for i in x], "--")
pyplot.plot(x, [fB(i) for i in x], "--")
pyplot.plot(x, [fC(i) for i in x], "-")


def fA(x):
    if x < 1:
        return 0
    elif 1 <= x and x < 7:
        return (x - 1) / 6
    elif 1 <= x and x < 10:
        return (10 - x) / 3
    elif x >= 10:
        return 0


def fB(x):
    if x < 2:
        return 0
    elif 1 <= x and x < 2:
        return (x - 1) / 6
    elif 2 <= x and x < 3:
        return x - 2
    elif 3 <= x and x < 4:
        return 1
    elif 4 <= x and x < 6:
        return (6 - x) / 2
    elif x > 6:
        return 0


def fC(x):
    if x < 2:
        return 0
    elif 2 <= x and x < 2.2:
        return x - 2
    elif 2.2 <= x and x < 4.75:
        return (x - 1) / 6
    elif 4.75 <= x and x < 6:
        return (6 - x) / 2
    elif x >= 6:
        return 0


x = np.linspace(1, 10, 100)
pyplot.plot(x, [fA(i) for i in x], "--")
pyplot.plot(x, [fB(i) for i in x], "--")
pyplot.plot(x, [fC(i) for i in x], "-")


def fA(x):
    if x < 1:
        return 0
    elif 1 <= x and x < 7:
        return (x - 1) / 6
    elif 7 <= x and x < 10:
        return (10 - x) / 3
    elif x >= 10:
        return 0


def fAA(x):
    if x < 1:
        return 1
    elif 1 <= x and x < 7:
        return 1 - ((x - 1) / 6)
    elif 7 <= x and x < 10:
        return 1 - ((10 - x) / 3)
    elif x >= 10:
        return 1


x = np.linspace(1, 10, 100)
pyplot.plot(x, [fA(i) for i in x], "--")
pyplot.plot(x, [fAA(i) for i in x], "-")


def fB(x):
    if x < 2:
        return 0
    elif 1 <= x and x < 2:
        return (x - 1) / 6
    elif 2 <= x and x < 3:
        return x - 2
    elif 3 <= x and x < 4:
        return 1
    elif 4 <= x and x < 6:
        return (6 - x) / 2
    elif x > 6:
        return 0


def fBB(x):
    if x < 2:
        return 1
    elif 1 <= x and x < 2:
        return 1 - (x - 1) / 6
    elif 2 <= x and x < 3:
        return 1 - (x - 2)
    elif 3 <= x and x < 4:
        return 0
    elif 4 <= x and x < 6:
        return 1 - (6 - x) / 2
    elif x > 6:
        return 1


x = np.linspace(1, 10, 100)
pyplot.plot(x, [fB(i) for i in x], "--")
pyplot.plot(x, [fBB(i) for i in x], "-")


def fA(x):
    if x < 1:
        return 0
    elif 1 <= x and x < 7:
        return (x - 1) / 6
    elif 7 <= x and x < 10:
        return (10 - x) / 3
    elif x >= 10:
        return 0


def fAA(x):
    if x < 1:
        return 1
    elif 1 <= x and x < 7:
        return 1 - ((x - 1) / 6)
    elif 7 <= x and x < 10:
        return 1 - ((10 - x) / 3)
    elif x >= 10:
        return 1


def fAuAA(x):
    if x < 1:
        return 1
    elif 1 <= x and x < 4:
        return 1 - ((x - 1) / 6)
    elif 4 <= x and x < 7:
        return (x - 1) / 6
    elif 7 <= x and x < 8.5:
        return (10 - x) / 3
    elif 8.5 <= x and x < 10:
        return 1 - ((10 - x) / 3)
    elif x >= 10:
        return 1


x = np.linspace(1, 10, 100)
pyplot.plot(x, [fA(i) for i in x], "--")
pyplot.plot(x, [fAA(i) for i in x], "--")
pyplot.plot(x, [fAuAA(i) for i in x], "-")


def fA(x):
    if x < 1:
        return 0
    elif 1 <= x and x < 7:
        return (x - 1) / 6
    elif 7 <= x and x < 10:
        return (10 - x) / 3
    elif x >= 10:
        return 0


def fAA(x):
    if x < 1:
        return 1
    elif 1 <= x and x < 7:
        return 1 - ((x - 1) / 6)
    elif 7 <= x and x < 10:
        return 1 - ((10 - x) / 3)
    elif x >= 10:
        return 1


def fAuAA(x):
    if x < 1:
        return 0
    elif 1 <= x and x < 4:
        return (x - 1) / 6
    elif 4 <= x and x < 7:
        return 1 - ((x - 1) / 6)
    elif 7 <= x and x < 8.5:
        return 1 - ((10 - x) / 3)
    elif 8.5 <= x and x < 10:
        return (10 - x) / 3
    elif x >= 10:
        return 0


x = np.linspace(1, 10, 100)
pyplot.plot(x, [fA(i) for i in x], "--")
pyplot.plot(x, [fAA(i) for i in x], "--")
pyplot.plot(x, [fAuAA(i) for i in x], "-")


def fA(x):
    if x < 1:
        return 0
    elif 1 <= x and x < 2.2:
        return (x - 1) / 6
    elif 2.2 <= x and x < 3:
        return x - 2
    elif 3 <= x and x < 4:
        return 1
    elif 4 <= x and x < 4.75:
        return (6 - x) / 2
    elif 4.75 <= x and x < 6:
        return (x - 1) / 6
    elif x >= 6:
        return 0


def fB(x):
    if x < 1:
        return 1
    elif 1 <= x and x < 2.2:
        return 1 - (x - 1) / 6
    elif 2.2 <= x and x < 3:
        return 1 - (x - 2)
    elif 3 <= x and x < 4:
        return 0
    elif 4 <= x and x < 4.75:
        return 1 - (6 - x) / 2
    elif 4.75 <= x and x < 6:
        return 1 - (x - 1) / 6
    elif x >= 6:
        return 1


x = np.linspace(1, 10, 100)
pyplot.plot(x, [fA(i) for i in x], "--")
pyplot.plot(x, [fB(i) for i in x], "-")

def fA(x):
    if x < 2:
        return 0
    elif  2 <= x and x < 2.2:
        return (x -2)
    elif  2.2 <= x and x < 4.75:
        return (x - 1)/6
    elif  4.75 <= x and x < 6:
        return (6-x)/2
    elif x >= 6:
        return 0
def fB(x):
    if x < 2:
        return 1
    elif  2 <= x and x < 2.2:
        return 1-(x -2)
    elif  2.2 <= x and x < 4.75:
        return 1-(x - 1)/6
    elif  4.75 <= x and x < 6:
        return 1-(6-x)/2
    elif x >= 6:
        return 1

x = np.linspace(1, 10, 100)
pyplot.plot(x, [fA(i) for i in x],'--')
pyplot.plot(x, [fB(i) for i in x],'-')

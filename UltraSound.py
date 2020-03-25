import matplotlib.pyplot as plt
import numpy as np
from func import *
from math import e


def detect_defect(mass, L, c, lyambda, a0, alpha1):
    '''
        Функция обнаруживает дефект в обсласти от 0 до L-1
        Если волна болше дефекта, то волна проходит дальше
        Если волна меньше дефекта, то она отражается и идет назад
    :param mass: массив с дефектами
    :param L: Длина массива
    :param с: Скорость
    :return: амплитуду
    '''
    f = 10 ** 7
    r = 0
    for i in range(L):
        r += 1
        if lyambda < (mass[i] / f):
            r = i
            break

    a = a0 * e ** (-alpha1 * r * 2)
    t = 2 * r / c
    return a, t, r


def ultra_sound():
    L = 100 # Длина стержня
    c = 5 * (10 ** 3) # Скорость ультразвуковой волны
    f = 10 ** 7
    lyambda = c / f
    print(lyambda)
    a0 = 10 ** 10
    alpha1 = 0.1
    a = a0 * np.e ** (-alpha1 * L * 2)
    print(a)

    # Создаем массив скоростей
    mass_area = np.full(L, c)
    x_mass_area = np.arange(L)
    mass_spikes = spikes(mass_area, m=1)

    Ampl, t, r = detect_defect(mass_area, L, c, lyambda, a0, alpha1)
    print('Время отражения', t, 'Расстояние до дефекта или дна', r)

    plt.plot(x_mass_area, mass_area)
    plt.show()


if __name__ == '__main__':
    ultra_sound()




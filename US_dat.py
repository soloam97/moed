from matplotlib import pyplot as plt
import numpy as np
import struct


# Чтение массива из файла .dat
def open_reader_us(file):
    with open(file, "rb") as binary_file:
        figures = []

        data = binary_file.read()
        for i in range(0, len(data), 8):
            elem = struct.unpack('d', data[i:i + 8])
            figures.append(elem[0])
        return figures


# Добавление спайков
def spikes(mass, s=10, m=10):
    '''
    Функция добавляет спайки ко входному массиву значений
    :param mass:
    :param s:
    :param m:
    :return: массив со спайками
    '''
    N = len(mass)
    y = [0 for i in range(N)]
    spikes_mass = [0 for i in range(N)]
    for i in range(m):
        k = np.random.randint(N)
        y[k] = np.random.randint(s * (10 ** 1), 1.1 * s * (10 ** 1))
        y[k] *= np.random.choice([-1, 1])

    for i in range(N):
        spikes_mass[i] = mass[i] + y[i]
    return spikes_mass


# Функция antispikes устраняет помеху нехарактерных значений
def antispikes(mass, N, s):
    '''
    Функция устраняет нехарактерные значения, экстраполируя их
    :param mass: входной массив значений
    :param N: количество значений в массиве
    :param s: разброс значений
    :return: отфильтрованный массив
    '''
    for i in range(1, N - 1):
        if abs(mass[i]) > (s + 0.5 * s):
            mass[i] = (mass[i - 1] + mass[i + 1]) / 2
    return mass


# Сдвиг данных - функция shift
def shift(mass, x1, x2, c):
    '''
    :param mass: массив значений, часть которого нужно сдвинуть
    :param x1: первая координата сдвига
    :param x2: вторая координата сдвига
    :param c: константа сдвига
    :return: массив значений со сдвигом
    '''
    N = len(mass)
    shift_mass = [0 for i in range(N)]
    for i in range(0, N):
        if x1 <= i <= x2:
            shift_mass[i] = mass[i] + c
        else:
            shift_mass[i] = mass[i]
    return shift_mass


# Математическое ожидание
def expected_value(mass):
    '''
    Функция считает математическое ожидание массива значений
    :param mass: входной массив значений
    :return: математическое ожидание массива
    '''
    mo = 0
    for i in mass:
        mo += i
    return mo / (len(mass))


# Функция antishift иссключает помеху сдвига из данных
def anti_shift(mass, N, exp_val):
    '''
    Функция удаляет помеху сдвига
    :param mass: входной массив данных с сдвигом
    :param N: количество элементов входного массива
    :param exp_val: математическое ожидание значений массива
    :return: значения массива без сдвига
    '''
    mass1 = []
    for i in range(N):
        mass1.append(mass[i])
    mass1[0] = 0
    for i in range(1, N):
        mass1[i] -= exp_val
    return mass1


# Смоделировать аддитивный шум в виде тренда
def mass_trend(mass, trend):
    '''
    Функция суммирует массив с шумом ввиде тренда
    :param mass: входной массив значений
    :param trend: шумовой тренд
    :return: массив значений с трендом
    '''
    N = len(mass)
    mass_trend_y = []
    for i in range(N):
        mass_trend_y.append(mass[i] + trend[i])
    return mass_trend_y


# Функции для генерации линейных трендов
def lin_trends(N, k, b):
    '''
    Функция генерирует зависимость линейного тренда от времени
    :param N: количество значений в массиве
    :param k: коэффициент наклона линейной заивисимости
    :param b: параметр сдвига линейной зависимости
    :return: ось значений линейной зависимости
    '''
    x = np.arange(N)
    y = k * x + b
    return y


# Функция устраняет тренд методом скользящего среднего
def antitrend(mass, N, L):
    '''
    Функция устраняет шум ввиде тренда
    :param mass: входной массив с трендом
    :param N: количество элементов массива
    :param L: ширина окна
    :return: массив значений без тренда
    '''
    a = 0  # переменная для усреднения значений в окне
    mass_antitrend_y = []
    for i in range(N - L):
        for j in range(L):
            a += mass[i + j]
        a /= L
        mass_antitrend_y.append(a)
    mass_antitrend_x = np.arange(N - L)
    for i in range(N - L):
        mass_antitrend_y[i] -= mass[i]
    return mass_antitrend_x, mass_antitrend_y


def main():
    input_mass = open_reader_us("C:/Users/soloa/OneDrive/Документы/MATLAB/matrica4.dat")
    N = len(input_mass)
    x_input_mass = np.arange(N)
    # Применим спайки к входному массиву
    s = 10
    spikes_mass = spikes(input_mass, s, m=10)


# Спайк и антиспайк
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x_input_mass, input_mass)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Сигнал УЗ луча')

    plt.subplot(2, 2, 2)
    plt.plot(x_input_mass, spikes_mass)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Сигнал УЗ луча со спайками')

    plt.subplot(2, 2, 3)
    plt.plot(x_input_mass, antispikes(spikes_mass, N, s))
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Отфильтрованный сигнал УЗ луча')

    plt.tight_layout()
    plt.show()

# Применим шифт к входному массиву
    x1, x2, c = 1, N, 100
    shift_mass = shift(input_mass, x1, x2, c)
    exp_val = expected_value(shift_mass)

# Шифт и антишифт
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x_input_mass, input_mass)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Сигнал УЗ луча')

    plt.subplot(2, 2, 2)
    plt.plot(x_input_mass, shift_mass)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Сигнал УЗ луча со сдвигом')

    plt.subplot(2, 2, 3)
    plt.plot(x_input_mass[1:], anti_shift(shift_mass, N, exp_val)[1:])
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Отфильтрованный сигнал УЗ луча')

    plt.tight_layout()
    plt.show()


# Применим тренд
    k, b = 0.1, 1
    L = 10
    trend = lin_trends(N, k, b)
    trend_mass = mass_trend(trend, input_mass)
    antitrend_mass_x, antitrend_mass_y = antitrend(input_mass, N, L)

# Тренд и антитренд
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x_input_mass, input_mass)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Сигнал УЗ луча')

    plt.subplot(2, 2, 2)
    plt.plot(x_input_mass, trend_mass)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Сигнал УЗ луча с трендом')

    plt.subplot(2, 2, 3)
    plt.plot(antitrend_mass_x, antitrend_mass_y)
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Отфильтрованный сигнал УЗ луча')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
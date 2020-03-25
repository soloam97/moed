import struct

from pylab import *
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image, ImageDraw
import matplotlib.image as mpimg
import pandas as pd
import scipy.integrate as integrate


# Функции для линейных трендов
def lin_trends(N, k, b):
    '''
    Функция генерирует зависимость линейного тренда от времени
    :param N: количество значений в массиве
    :param k: коэффициент наклона линейной заивисимости
    :param b: параметр сдвига линейной зависимости
    :return: ось времени и ось значений линейной зависимости
    '''
    x = np.arange(N)
    y = k * x + b
    return x, y


# Функции для экспоненциальных трендов
def exp_trends(N, betta, alpha):
    '''

    :param N: количество элементов массива
    :param betta: коэффициент
    :param alpha: степенной коэффициент
    :return: ось времени и ось значений экспоненциальной зависимости
    '''
    x = np.arange(N)
    y = betta * e ** (alpha * x)
    return x, y


# Функция генерирует масиив для встроенного генератора случайных чисел
def rand_stand(N, s):
    '''
    :param N: количество случайных чисел
    :param s: разброс случайных чисел
    :return: массив от 0 до 1000 и массив со случайными числами
    '''
    x = np.arange(N)
    y = np.random.uniform(-s, s, (N))
    return x, y


# Функция генерирует массив случайных чисел с помощью самодельного генератора
def rndmm(s):
    tim = time.time()
    while tim != tim // 1:
        tim *= 10
    tim = tim // 10
    t = str(tim)
    t = t[::-1]
    t = float(t)
    a, c = (s / 2 + 1), (s / 3 + 2)
    for i in range(100):
        b = sin(t)
        if b > 0:
            t = (t * a + c) % (s)
        else:
            t = -((t * a + c) % (s))
    return t


def my_rand(N, s):
    y = []
    for i in range(N):
        y.append(rndmm(s))
        time.sleep(0.00001)
    return y


# Сдвиг данных - функция shift
def shift(mass, x1, x2, c):
    '''
    :param mass: массив значений, часть которого нужно сдвинуть
    :param x1: первая координата сдвига
    :param x2: вторая координата сдвига
    :param c: константа сдвига
    :return: массив значений со сдвигом
    '''
    for i in range(x1, x2):
        mass[i] += c
    return mass


# Добавление спайков
def spikes(mass, s=10, m=10):
    '''
    Функция добавляет спайки ко входному массиву значений
    :param mass:
    :param s:
    :param m:
    :return:
    '''
    N = len(mass)
    y = [0 for i in range(N)]
    for i in range(m):
        k = np.random.randint(N)
        y[k] = np.random.randint(s * (10 ** 1), 1.1 * s * (10 ** 1))
        y[k] *= np.random.choice([-1, 1])

    mass += y
    return mass


# Добавление аддитивного шума
def additive_noise(trend, s=10):
    '''
    Функция добавляет к тренду trend аддитивный шум
    :param trend: значения данной зависимости
    :return: зависимость зашумленного тренда от времени
    '''
    N = len(trend)
    length_mass = [i for i in range(N)]
    add_noise = np.random.uniform(-s, s, (N))
    trend_noise = add_noise + trend
    return length_mass, trend_noise


# Добавление мультитпликативного шума
def multiplicative_noise(trend, s=10):
    '''
    Функция добавляет к тренду trend мультипликативный шум
    :param trend: значения данной зависимости
    :return: зависимость зашумленного тренда от времени
    '''
    N = len(trend)
    length_mass = [i for i in range(N)]
    add_noise = np.random.uniform(-s, s, (N))
    trend_noise = add_noise * trend
    return length_mass, trend_noise


# Переход от непрерывной функции к дискретной
def garm_processes(i, f, a0, dt, N):
    '''
    :param i: количество гармоничесикх процессов
    :param f: массив, хранящий частоты
    :param a0: амплитуда гармонического процесса
    :param dt: шаг дискретизации
    :param N: количество элементов для вывода
    :return: массив значений гармонического процесса
    '''
    garm = []
    for index in range(i):
        if type(a0) != list:
            garm.append([(a0 * sin(2 * np.pi * f[index] * k * dt)) for k in range(N)])
        else:
            garm.append([(a0[index] * sin(2 * np.pi * f[index] * k * dt)) for k in range(N)])
    return garm


# Преобразование Фурье - амплитудный спектр
def ampl_spectr(mass, N):
    '''
    Функция производит преобразование Фурье,
    Возвращает амплитудный спектр
    :param mass: массив значений
    :param N: количество шагов дискретизации
    :return: амплитудный спектр, аргументы
    '''

    # Считаем дейстивтельную и мнимую части набора значений для гармонического процесса
    re = [0 for i in range(N)]
    im = [0 for i in range(N)]
    for m in range(N):
        for k in range(N):
            re[m] += (mass[k] * cos((2 * math.pi * m * k) / N))
            im[m] += (mass[k] * sin((2 * math.pi * m * k) / N))
        re[m] *= (1 / N)
        im[m] *= (1 / N)

    # Заполняем массив значениями модулей гармонического процесса
    C = [0 for i in range(N)]
    for m in range(N):
        C[m] = sqrt((re[m]) ** 2 + (im[m]) ** 2)
    return C


# Полигармонический процесс
def poligarm_procceses(i, f, a, dt, N):
    '''
    :param i: количество гармоничесикх процессов
    :param f: массив, хранящий частоты
    :param a: массив, хранящий амплитуды
    :param dt: шаг дискретизации
    :param N: количество значений
    :return: полигармоническую функцию
    '''

    # Реализуем i гармонических функций
    garm = [0 for j in range(i)]
    for index in range(i):
        garm[index] = [(a[index] * sin(2 * np.pi * f[index] * k * dt)) for k in range(N)]

    # Сложим i гармнонических функций и получим полигармонический процесс
    poligarm = [0 for i in range(N)]
    for i in range(N):
        poligarm[i] = garm[0][i] + garm[1][i] + garm[2][i]
        # poligarm[i] = garm[0][i] + garm[1][i]

    return poligarm


# Гистограмма - распределние плотности вероятности
def historam(y, s, N, number_of_step=100):
    '''
    Весь диапазон от -s до s разбить на итервалы (на m=40 штук)
    Пробегая все значения в соответствующий интеравал - количество значений,
    Попадающий в это значение
    :param y: входная зависимость от абсциссы
    :param s: разброс значений ординаты
    :param N: количество значений
    :param number_of_step: количество делений для гистограммы
    :return:
    '''
    step = 2 * s / number_of_step  # размер одного деления гистограммы
    x_hist = np.arange(-s, s, step)  # линейка гистограммы

    # Вычисляем значения гистограммы по ординате
    index_hist = 0
    y_hist = [0 for i in range(number_of_step)]
    for i in x_hist:
        for j in range(N):
            if i < y[j] < (i + 1):
                y_hist[index_hist] += 1
        index_hist += 1

    return x_hist, y_hist


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


# Автокорреляционная функция
def acf(mass, N):
    sr_mass = expected_value(mass)
    r_xx = [0 for i in range(N)]
    r_xx_zn = 0

    # Считаем знаменатель для АКФ
    for k in range(N):
        r_xx_zn += (mass[k] - sr_mass) ** 2

    for L in range(N):
        for k in range(N - L):
            r_xx[L] += (mass[k] - sr_mass) * (mass[k + L] - sr_mass)
        r_xx[L] /= r_xx_zn
    return r_xx


# Функция взаимной корреляции
def mcf(mass1, mass2, N):
    '''
    Функция взаимной корреляции характеризует степень корреляции
    двух функций
    :param mass1: массив значений первой функции
    :param mass2: массив значений второй функции
    :param N: количество значений
    :return: массив значений функции взаимной корреляции
    '''
    sr_y1 = expected_value(mass1)
    sr_y2 = expected_value(mass2)
    r_xy = [0 for i in range(N)]
    r_xy_zn1 = 0.0
    r_xy_zn2 = 0.0

    for k in range(N):
        r_xy_zn1 += (mass1[k] - sr_y1) ** 2
        r_xy_zn2 += (mass2[k] - sr_y2) ** 2
    r_xy_zn = sqrt(r_xy_zn1 * r_xy_zn2)

    for L in range(N):
        for k in range(N - L):
            r_xy[L] += (mass1[k] - sr_y1) * (mass2[k] - sr_y2)
        r_xy[L] /= r_xy_zn
    return r_xy


# Функция antishift иссключает помеху сдвига из данных
def anti_shift(mass, N, exp_val):
    '''
    Функция удаляет помеху сдвига
    :param mass: входной массив данных с сдвигом
    :param N: количество элементов входного массива
    :param exp_val: математическое ожидание значений массива
    :return: значения массива без сдвига
    '''
    mass1 = copy(mass)
    mass1[0] = 0
    for i in range(1, N):
        mass1[i] -= exp_val
    return mass1


# Функция antispikes устраняет помеху нехарактерных значений
def antispikes(mass, N, s):
    '''
    Функция устраняет нехарактерные значения, экстраполируя их
    :param mass: вхожной массив значений
    :param N: количество значений в массиве
    :param s: разброс значений
    :return: отфильтрованный массив
    '''
    for i in range(1, N - 1):
        if abs(mass[i]) > (s + 0.5 * s):
            mass[i] = (mass[i - 1] + mass[i + 1]) / 2
    return mass


# Смоделировать аддитивный шум в виде тренда, а затем убрать методом скользящего окна
def mass_trend(mass, trend):
    '''
    Функция суммирует массив с шумом ввиде тренда
    :param mass: входной массив значений
    :param trend: шумовой тренд
    :return: массив значений с трендом
    '''
    mass_trend_y = mass + trend
    mass_trend_x = np.arange(len(mass_trend_y))
    return mass_trend_x, mass_trend_y


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


# Чтение массива из файла .dat
def open_reader(file):
    with open(file, "rb") as binary_file:
        figures = []

        data = binary_file.read()
        for i in range(0, len(data), 4):
            elem = struct.unpack('f', data[i:i + 4])
            figures.append(elem[0])
        return figures


# Обратное преобразование Фурье inv_fourier_transform
def summ_re_im(mass, N):
    # Считаем действительную и мниму части
    re = [0 for i in range(N)]
    im = [0 for i in range(N)]
    Xm = []
    for m in range(N):
        for k in range(N):
            re[m] += (mass[k] * cos((2 * np.pi * m * k) / N))
            im[m] += (mass[k] * sin((2 * np.pi * m * k) / N))
        re[m] *= (1 / N)
        im[m] *= (1 / N)
        Xm.append(re[m] + im[m])
    return Xm


def inv_fourier_transf(Xm, N):
    '''
    Функция производит обратное преобразование Фурье
    :param xm: входной массив
    :param N: количество элементов
    :return: преобразование Фурье
    '''
    im = [0 for i in range(N)]
    re = [0 for i in range(N)]
    xm = []
    for m in range(N):
        for k in range(N):
            re[m] += (Xm[k] * cos((2 * np.pi * m * k) / N))
            im[m] += (Xm[k] * sin((2 * np.pi * m * k) / N))
        # im[m] *= (1 / N)
        # re[m] *= (1 / N)
        xm.append(re[m] + im[m])
    return xm


# Функиця обнуляет 50 последних значений
def zero_last_50(mass, N, zero_last_m):
    for i in range(N - zero_last_m, N):
        mass[i] = 0
    return mass


# ------------------------------------------------------
# Модуль для работы с ЭКГ
# ------------------------------------------------------

# Функция, генерирующая примерный ЭКГ сигнал
def ecg(x, alpha=30, f0=10, dt=0.005):
    '''
    Функция генерирует примерный сигнал ЭКГ
    :param x: массив значений по абсциссе
    :param alpha: коэффициент степени экспоненты
    :param f0: частота гармочниеской составляющей
    :param dt: шаг дискретизации
    :return: ЭКГ массив
    '''
    ecg = np.sin(2 * np.pi * f0 * dt * x) * np.exp(-alpha * x * dt)
    return ecg


# Функция, генерирующая примерный ЭКГ сигнал
def ecg1(x, N, alpha=30, f0=10, dt=0.005):
    '''
    Функция генерирует примерный сигнал ЭКГ
    :param x: массив значений по абсциссе
    :param alpha: коэффициент степени экспоненты
    :param f0: частота гармочниеской составляющей
    :param dt: шаг дискретизации
    :return: ЭКГ массив
    '''
    ecg = []
    for i in range(N):
        ecg.append(math.sin(2 * math.pi * f0 * dt * x[i]) * math.e ** (-alpha * x[i] * dt))
    return ecg


# Функция генерирует тики, то есть задает ЧСС
def ticks(N, l):
    '''
    Функция рисует тики (повторяющиеся одинаковые спайки).
    На вход программа получает общее уоличество точек N
    Количество тиков ticks
    Уровень тиков ticks_level
    :param N: количество элементов
    :param l: первая координата расположения тика
    :return:
    '''
    ticks_strength = 120  # Уровень тиков
    ticks_count = int(N / l)
    ticks_mass = [0 for i in range(N)]
    for number in range(1, ticks_count):
        ticks_mass[number * l - 1] = ticks_strength
    return ticks_mass


# Функция связывает входной сигнал с управляющим - свертка
def convolution(input_mass, control_mass):
    '''
    Функция связывает входной сигнал с управляющим
    :param input_mass: массив входного сигнала
    :param control_mass: массив управляющего сингала
    :return: элементы массива свертки
    '''
    N, M = len(input_mass), len(control_mass)  # Размер входного и управляющего массивов
    conv_mass = []  # Массив, заполняемый элементами свертки
    sum_of_conv = 0
    for k in range(N + M - 1):
        for m in range(M):
            if k - m < 0:
                pass
            if k - m > N - 1:
                pass
            else:
                sum_of_conv += input_mass[k - m] * control_mass[m]

        conv_mass.append(sum_of_conv)
        sum_of_conv = 0
    return conv_mass


# ------------------------------------------------------

# Фильтр назких частот
def low_pass_filter(m=32, dt=0.001, fc=100):
    '''
    Функция для фильтрации низких частот
    :param m: ширина окна (чем больше m, тем круче переходная характеристика фильтра
    :param dt: шаг дискретизации
    :param lpw: вес фильтра
    :return: отфильтрованный массив
    '''

    # расчитвываем веса прямоугольной функции
    arg = 2 * fc * dt
    lpw = []
    lpw.append(arg)
    arg *= np.pi
    for i in range(1, m + 1):
        lpw.append(np.sin(arg * i) / (np.pi * i))

    # Для трапецивидной
    lpw[m] /= 2
    # Применяем окно Поттера сглаживания окно p310 для сглаживания
    # Это окно требует 4 константы:
    d = [0.35577019, 0.24369830, 0.07211497, 0.00630165]
    summ = lpw[0]
    for i in range(1, m + 1):
        summ2 = d[0]
        arg = (np.pi * i) / m
        for k in range(1, 4):
            summ2 += 2 * d[k] * np.cos(arg * k)
        lpw[i] *= summ2
        summ += 2 * lpw[i]
    # Делаем нормировку
    for i in range(m + 1):
        lpw[i] /= summ

    # for i in range(100):
    #     lpw.append(0)
    # Доделать m+1 в 2m+1
    lpw = lpw[::-1] + lpw[1:]
    # for i in range(len(lpw)):
    #     lpw[i] *= len(lpw)
    return lpw


# Фильтр высоких чатсот
def high_pass_filter(m, dt, fc):
    '''
    Фильтр высоких частот
    :param m: отвечает за крутизну фильтра
    :param dt: шаг дискретизации
    :param fc: частота среза
    :return: характеристику фильтра
    '''
    lpw = low_pass_filter(m, dt, fc)
    hpw = []
    for i in range(2 * m + 1):
        if i == m:
            hpw.append(1 - lpw[i])
        else:
            hpw.append(-lpw[i])
    return hpw


# Полосовой фильтр
def bend_pass_filter(m, dt, fc1, fc2):
    '''
    Полосовой фильтр
    :param m: отвечает за крутизну
    :param dt: шаг дискретизации
    :param fc1: входная частота фильтра
    :param fc2: выходная частота фильтра
    :return: характеристику фильтра
    '''
    lpw1 = low_pass_filter(m, dt, fc1)
    lpw2 = low_pass_filter(m, dt, fc2)
    bpw = []
    for i in range((2 * m) + 1):
        bpw.append(lpw2[i] - lpw1[i])
    return bpw


# РЕжекторный фильтр
def bend_stop_filter(m, dt, fc1, fc2):
    '''
    Режекторный фильтр
    :param m: отвечает за крутизну
    :param dt: шаг дискретизации
    :param fc1: входная частота фильтра
    :param fc2: выходная частота фильтра
    :return: характеристика фильтра
    '''
    lpw1 = low_pass_filter(m, dt, fc1)
    lpw2 = low_pass_filter(m, dt, fc2)
    bsw = []
    for i in range((2 * m) + 1):
        if i == m:
            bsw.append(1 + lpw1[i] - lpw2[i])
        else:
            bsw.append(lpw1[i] - lpw2[i])
    return bsw


# Чтение массива из файла .wav
def wav_reader(file):
    with open(file, "rb") as binary_file:
        figures = []

        data = binary_file.read()
        for i in range(44, len(data), 2):
            elem = struct.unpack('h', data[i:i + 2])
            figures.append(elem[0])
        return figures


# Действительная и мнимая части функции
def re_and_im(mass, N):
    '''
    Функция производит преобразование Фурье,
    Возвращает амплитудный спектр
    :param mass: массив значений
    :param N: количество шагов дискретизации
    :return: действительную и мнимую части
    '''

    # Считаем дейстивтельную и мнимую части набора значений для гармонического процесса
    re = [0 for i in range(N)]
    im = [0 for i in range(N)]
    for m in range(N):
        for k in range(N):
            re[m] += (mass[k] * cos((2 * np.pi * m * k) / N))
            im[m] += (mass[k] * sin((2 * np.pi * m * k) / N))
        re[m] *= (1 / N)
        im[m] *= (1 / N)

    return re, im


def complex_ratio(x_re, x_im, y_re, y_im, N):
    '''
       Функция считает отношение двух комплексных величин
    :param x: первая коммплексная величина
    :param y: вторая коммплексная величина
    :return: отношение
    '''
    ratio_of_x_y_re, ratio_of_x_y_im = [], []
    for i in range(N):
        ratio_of_x_y_re.append((x_re[i] * y_re[i] + x_im[i] * y_im[i]) / (y_re[i] ** 2 + y_im[i] ** 2))
        ratio_of_x_y_im.append((y_re[i] * x_im[i] - x_re[i] * y_im[i]) / (y_re[i] ** 2 + y_im[i] ** 2))

    return ratio_of_x_y_re, ratio_of_x_y_im


# ------------------------------------------------------------------------------------------------------------------------
# Второй семестр
# Работа с изображениями
# ------------------------------------------------------------------------------------------------------------------------

# Функция нормировки
def normalization(mass, N=255):
    '''
    Функция нормировки
    :param mass: список значений оттенков серости пикселей
    :param N: количество элементов
    :return: норма
    '''
    norm = []
    min_mass = min(mass)
    max_mass = max(mass)
    for num in mass:
        norm.append(((num - min_mass) / (max_mass - min_mass)) * N)
    return norm


def image_read():
    mode = int(input('mode:'))  # Считываем номер преобразования.
    image = Image.open("grace.jpg")  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.
    return pix


def read_png(file):
    img = mpimg.imread(file)
    return img


def read_jpg_gray(file):
    image = Image.open(file).convert('L')
    return image


def read_xcr():
    pass


def image_scale(image, const, type, mode):
    pix = image.load()  # Выгружаем значения пикселей
    w, h = image.size[0], image.size[1]

    if mode == 'increased':
        new_w, new_h = int(w * const), int(h * const)
    elif mode == 'decreased':
        new_w, new_h = int(w / const), int(h / const)
    else:
        pass

    if type == 'nn':
        image_nearest_resized = Image.new('L', (new_w, new_h))
        draw = ImageDraw.Draw(image_nearest_resized)  # Создаем инструмент для рисования
        for col in range(new_w):
            for row in range(new_h):
                if mode == 'increased':
                    p = pix[int(col / const), int(row / const)]
                elif mode == 'decreased':
                    p = pix[int(col * const), int(row * const)]
                else:
                    pass
                draw.point((col, row), p)
        image_resized = image_nearest_resized

    elif type == 'bi':
        image_bilinear_rows = Image.new('L', (new_w, new_h))
        draw = ImageDraw.Draw(image_bilinear_rows)  # Создаем инструмент для рисования
        for col in range(1, (new_w - 1)):
            for row in range(1, (new_h - 1)):
                if mode == 'increased':
                    r1 = pix[int(col / const), int((row - 1) / const)]
                    r2 = pix[int(col / const), int((row + 1) / const)]
                elif mode == 'decreased':
                    r1 = pix[int(col * const), int((row - 1) * const)]
                    r2 = pix[int(col * const), int((row + 1) * const)]
                else:
                    pass
                p = int((r1 + r2) / 2)
                draw.point((col, row), p)
            if mode == 'increase':
                draw.point((col, 0), pix[int(col / const), int(0 / const)])
                draw.point((col, new_h), pix[int(col / const), int((new_h - 1) / const)])
            elif mode == 'decreased':
                draw.point((col, 0), pix[int(col * const), int(0 * const)])
                draw.point((col, new_h), pix[int(col * const), int((new_h - 1) * const)])
            else:
                pass

        pix_bilinear_rows = image_bilinear_rows.load()
        image_bilinear_resized = Image.new('L', (new_w, new_h))
        draw2 = ImageDraw.Draw(image_bilinear_resized)  # Создаем инструмент для рисования

        for row in range(1, (new_h - 1)):
            for col in range(1, (new_w - 1)):
                r1 = pix_bilinear_rows[int((col - 1)), int(row)]
                r2 = pix_bilinear_rows[int((col + 1)), int(row)]
                p = int((r1 + r2) / 2)
                draw2.point((col, row), p)
            draw2.point((0, row), pix_bilinear_rows[int(0), int(row)])
            draw2.point((new_w, row), pix_bilinear_rows[int((new_w - 1)), int(row)])

        image_resized = image_bilinear_resized

    else:
        pass

    return image_resized


def pillow_image_grayscale_negative(image):
    pix = image.load()
    w, h = image.size[0], image.size[1]

    image_negative = Image.new('L', (w, h))
    draw = ImageDraw.Draw(image_negative)

    for col in range(w):
        for row in range(h):
            draw.point((col, row), 255 - pix[col, row])

    return image_negative


def pillow_image_grayscale_gammacorr(image, C, Y):
    pix = image.load()
    w, h = image.size[0], image.size[1]
    gammacorr = []

    image_gammacorr = Image.new('L', (w, h))
    draw = ImageDraw.Draw(image_gammacorr)

    for col in range(w):
        for row in range(h):
            gammacorr.append(int(C * pix[col, row] ** Y))

    gammacorr_norm = normalization(gammacorr, 255)

    i = 0
    for col in range(w):
        for row in range(h):
            draw.point((col, row), int(gammacorr_norm[i]))
            i += 1

    return image_gammacorr


def pillow_image_grayscale_log(image, C):
    pix = image.load()
    w, h = image.size[0], image.size[1]
    log = []

    image_log = Image.new('L', (w, h))
    draw = ImageDraw.Draw(image_log)

    for col in range(w):
        for row in range(h):
            log.append(C * np.log(pix[col, row] + 1))

    log_norm = normalization(log, 255)

    i = 0
    for col in range(w):
        for row in range(h):
            draw.point((col, row), int(log_norm[i]))
            i += 1

    return image_log


def pillow_image_grayscale_grad_preobr(r):
    func = P()
    CDF = integrate.quad(func, 0, r)
    return 0


def pillow_image_grayscale_hist(image):
    pix = list(image.getdata())
    data = pd.Series(pix)
    return data


# Чтение массива пикселей из xcr файла
def open_reader_xcr(file):
    with open(file, "rb") as binary_file:
        figures = []

        data = binary_file.read()
        for i in range(0, len(data), 2):
            elem = struct.unpack('h', data[i:i + 2])
            figures.append(elem[0])
        return figures


def image_histogram(image):
    pixels = image.load()
    w, h = image.size[0], image.size[1]

    i = 0
    pixels_1d = []
    for col in range(w):
        for row in range(h):
            pixels_1d.append(pixels[col, row])

    # нормализация данных
    S = col * row
    pixels_1d = normalization(pixels_1d)
    for i in range(S):
        pixels_1d[i] = int(pixels_1d[i])

    # создаем список гистограммы
    image_hist_y = [0 for i in range(255)]
    image_hist_x = [i for i in range(255)]

    index_hist = 0
    for i in image_hist_x:
        for pix in pixels_1d:
            if pix == i:
                image_hist_y[index_hist] += 1
        index_hist += 1

    return image_hist_x, image_hist_y, pixels_1d


# Создание картинки по считываемым данным по одномерному массиву пикселей
def drawing_image_new(pixels, w, h):
    '''
    Функция рисует картинку в оттенках серого по вхожному одномерному списку пикселей
    :param pixels: одномерный список оттенков серого каждого пикселя
    :param w: ширина создаваемой картинки
    :param h: высота создаваемой картинки
    :return: image_new - картинка в оттенках серого
    '''
    image_new = Image.new('L', (w, h))  # создаем пустую картинку в оттенках серого с шириной w и высотой h
    draw = ImageDraw.Draw(image_new)  # Запускаем инструмент для рисования

    # нормализуем значения оттенков серого
    S = 255
    image_new_norm = normalization(pixels, S)
    # image_new_norm = pixels
    # заполняем значения пикселей новой картинки оттенками серого входного списка
    i = 0
    for col in range(w):
        for row in range(h):
            draw.point((col, row), int(image_new_norm[i]))
            i += 1

    return image_new


def equalization(image):
    # Считываем  файл
    w, h = image.size[0], image.size[1]
    pix = image.load()
    # Строим гистограмму
    x, y, pixels_1d = image_histogram(image)
    plt.plot(x, y)
    plt.show()

    # Производим интегрирование
    cdf = [0]
    for i in range(1, len(y)):
        cdf.append(cdf[i - 1] + ((y[i - 1] + y[i]) * 0.5))
    max_cdf = max(cdf)
    for i in range(255):
        cdf[i] /= max_cdf

    plt.plot(x, cdf)
    plt.show()

    # new_image = drawing_image_new(pixels_1d, w, h)
    # new_image.show()

    for i in range(w * h):
        for j in range(255):
            if pixels_1d[i] == j:
                pixels_1d[i] = cdf[j] * 255

    new_image = drawing_image_new(pixels_1d, w, h)
    return new_image


# Функция, генерирующая список производных
def derivative(matrix_pix, w, h):
    '''
        Функция заполняет массив значениями производных по строкам
    :param matrix_pix: входной двумерный массив, элементы которого - яркости пикселей
    :param w: ширина изображения
    :param h: высота изображения
    :return: матрицу производных
    '''
    data = []
    for i in range(w):
        row = []
        for j in range(h - 1):
            row.append(matrix_pix[i][j + 1] - matrix_pix[i][j])
        data.append(row)

    return data


# Свертка для изображений
def image_conv(input_mass, control_mass, w, h, m):
    '''
        Функция реализует свертку изображения и управляющего сигнала
    :param input_mass: матрица изображения для обработки
    :param control_mass: массив управляющего сигнала
    :param w: ширина изображения
    :param h: высота изображения
    :param m: крутость переходной характеристики управляющего массива
    :return: отфильтрованное изображения
    '''
    data_conv = []
    for i in range(w):
        temp = convolution((input_mass[i]), control_mass)
        data_conv.append(temp[m:(h + m)])

    return data_conv


# Функция нормировки пикселей
def normalize_pixels(mass, N):
    norm = []
    min_pix = min(mass)
    max_pix = max(mass)
    for elem in mass:
        norm.append((((elem - min_pix) / (max_pix - min_pix)) - 0.5) * 2 * N)
    return norm


# Функция добавляет аддитивный шум Гаусса на картинку
def add_gauss_noise(file, level=20):
    '''
        Функция реализует добавление аддитивного шума Гауссовского распределения
    :param file: входной файл
    :param level: уровень шума
    :return: изображение с нормально распределенным шумом
    '''
    if type(file) == str:
    # Загружаем картинку
        image = read_jpg_gray(file)
    else:
        image = file
    w, h = image.size[0], image.size[1]
    pixels = image.load()  # Выгружаем значения пикселей

    # моделируем Гауссовский шум
    mu, sigma = 0.2, level
    gaussNoise = np.random.normal(mu, sigma, size=[w, h])

    # Добавление шума на изображение
    noisedImage = []
    for col in range(w):
        for row in range(h):
            noisedImage.append(pixels[col, row] + gaussNoise[col, row])

    # Рисование изображения с шумом
    image_add_noise = drawing_image_new(noisedImage, w, h)

    return image_add_noise


def add_impulse_noise(file, Pa=0.05, Pb=0.1):
    '''
        Функция реализует добавление аддитивного шума Гауссовского распределения
    :param file: входной файл
    :param Pa: вероятность для черного шума
    :param Pb: вероятность для белого шума
    :return:
    '''
    # Загружаем изображение
    if type(file) == str:
        image = read_jpg_gray(file)
    else:
        image = file
    w, h = image.size[0], image.size[1]
    pixels = image.load()  # Выгружаем значения пикселей
    pixels_1d = []
    for col in range(w):
        for row in range(h):
            pixels_1d.append(pixels[col, row])

    # Моделируем импульсный шум
    a = 0
    b = 255
    randVals = np.random.uniform(low=0.0, high=1.0,
                                 size=(w * h))  # массив случайных значений с нормальным распределением
    noisedImpulse = pixels_1d.copy()

    for i in range(w * h):
        if randVals[i] < Pa:
            noisedImpulse[i] = a
        elif Pa < randVals[i] < (Pa + Pb):
            noisedImpulse[i] = b

    # # Добавление шума на изображение
    # noisedImage2 = []
    # for i in range(w * h):
    #     noisedImage2.append(pixels_1d[i] + noisedImpulse[i])

    # Рисование изображения с шумом
    image_impulse_noise = drawing_image_new(noisedImpulse, w, h)

    return image_impulse_noise


def from_2d_to_1d(mass_2d, w, h):
    mass_1d = []
    for i in range(w):
        for j in range(h):
            mass_1d.append(mass_2d[i][j])
    return mass_1d


def from_1d_to_2d(mass_1d, w, h):
    mass_2d = []
    for col in range(h):
        elem = []
        for row in range(w):
            elem.append(mass_1d)
        mass_2d.append(elem)
    return mass_2d


def diff_by_row_for_trend(matrix_pixels, w, h):
    data = []
    for i in range(h):
        row = []
        for j in range(w - 1):
            row.append(matrix_pixels[i, j + 1] - matrix_pixels[i, j])
        data.append(row)

    return data
from matplotlib import pyplot as plt
from func import *
import soundfile as sf
import sounddevice as sd
import wavio
from PIL import *
import cv2


# -----------------------------------------
# Графики четырех трендов
# -----------------------------------------

def trends():
    N = 1000
    k1, b1 = 3, 4
    k2, b2 = -3, 4
    x1, y1 = lin_trends(N, k1, b1)
    x2, y2 = lin_trends(N, k2, b2)

    betta1, alpha1 = 0.1, 0.004
    betta2, alpha2 = 0.05, -0.005
    x3, y3 = exp_trends(N, betta1, alpha1)
    x4, y4 = exp_trends(N, betta2, alpha2)

    fig = plt.figure()

    subplot(2, 2, 1)
    plot(x1, y1, 'b-')
    xlabel('x')
    ylabel('y')
    title('y = kx+b, k > 0')

    subplot(2, 2, 2)
    plot(x2, y2, 'b-')
    xlabel('x')
    ylabel('y')
    title('y = kx+b, k < 0')

    subplot(2, 2, 3)
    plot(x3, y3, 'b-')
    xlabel('t')
    ylabel('y')
    title(r'$y = \beta \cdot e ^ {\alpha t} $, $\alpha > 0 $')

    subplot(2, 2, 4)
    plot(x4, y4, 'b-')
    xlabel('t')
    ylabel('y')
    title(r'$y = \beta \cdot e ^ {\alpha t} $, $\alpha < 0 $')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# График случайной последовательности значений (встроенный генератор)
# -----------------------------------------


def task_rand_stand():
    N, s = 1000, 10
    x, y = rand_stand(N, s)

    fig = plt.figure()

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Встроенный генератор случайных чисел')

    plt.show()


# -----------------------------------------
# График случайной последовательности значений (свой генератор)
# -----------------------------------------

def task_rand_my():
    N, s = 1000, 10
    x = np.arange(N)
    y = my_rand(N, s)

    # if abs(max(y)) > abs(min(y)):
    #     maxi = max(y)
    # else:
    #     maxi = abs(min(y))
    #
    # for i in range(N):
    #     y[i] = y[i] / maxi

    fig = plt.figure()

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Свой генератор случайных чисел')

    plt.show()


# -----------------------------------------
# Сдвиг данных - функция shift
# -----------------------------------------

def task_shift():
    N = 1000
    s = 10  # разброс значений
    x1, x2 = 200, 500  # первая и вторая координаты сдвига
    c = 1000  # константа сдвига
    x = [i for i in range(N)]
    mass = np.random.uniform(-s, s, N)

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x, mass)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Несвдинутые данные')

    mass_shift = shift(mass, x1, x2, c)
    plt.subplot(1, 2, 2)
    plt.plot(x, mass_shift)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Со сдвигом')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Добавление нехарактерных значений - функция spikes
# -----------------------------------------

def task_spikes():
    N = 1000
    s = 10
    m = 10
    x = [i for i in range(N)]
    mass = np.random.uniform(-s, s, N)

    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(x, mass)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Случайные числа')

    plt.subplot(1, 2, 2)
    plt.plot(x, spikes(mass, s, m))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Со спайками')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Добавление аддитивного шума
# -----------------------------------------

def task_additive_noise():
    N = 1000
    k1, b1 = -0.1, 1000
    trend1_x, trend1_y = lin_trends(N, k1, b1)
    k2, b2 = 0.1, 0
    trend2_x, trend2_y = lin_trends(N, k2, b2)
    trend1_noise_x, trend1_noise_y = additive_noise(trend1_y, s=10)
    trend2_noise_x, trend2_noise_y = additive_noise(trend2_y, s=10)

    fig = plt.figure('Аддитивный шум')

    subplot(2, 2, 1)
    plot(trend1_x, trend1_y, 'b-')
    xlabel('x')
    ylabel('y1')
    title('y1 = kx1+b, k < 0')

    subplot(2, 2, 2)
    plot(trend1_noise_x, trend1_noise_y)
    xlabel('x')
    ylabel('y2')
    title('y=kx+b, k < 0 + шум')

    subplot(2, 2, 3)
    plot(trend2_x, trend2_y, 'b-')
    xlabel('x')
    ylabel('y3')
    title('y3 = kx1+b, k > 0')

    subplot(2, 2, 4)
    plot(trend2_noise_x, trend2_noise_y)
    xlabel('x')
    ylabel('y4')
    title('y=kx+b, k > 0 + шум')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Добавление мультипликативного шума
# -----------------------------------------

def task_multiplicative_noise():
    N = 1000
    k1, b1 = -0.1, 100
    trend1_x, trend1_y = lin_trends(N, k1, b1)
    k2, b2 = 0.1, 0
    trend2_x, trend2_y = lin_trends(N, k2, b2)
    trend1_noise_x, trend1_noise_y = multiplicative_noise(trend1_y, s=10)
    trend2_noise_x, trend2_noise_y = multiplicative_noise(trend2_y, s=10)

    fig = plt.figure('Мультипликативный шум')

    plt.subplot(2, 2, 1)
    plt.plot(trend1_x, trend1_y, 'b-')
    plt.xlabel('x')
    plt.ylabel('y1')
    plt.title('y1 = kx1+b, k < 0')

    plt.subplot(2, 2, 2)
    plt.plot(trend1_noise_x, trend1_noise_y)
    plt.xlabel('x')
    plt.ylabel('y2')
    plt.title('y=kx+b, k < 0 + шум')

    plt.subplot(2, 2, 3)
    plt.plot(trend2_x, trend2_y, 'b-')
    plt.xlabel('x')
    plt.ylabel('y3')
    plt.title('y3 = kx1+b, k > 0')

    plt.subplot(2, 2, 4)
    plt.plot(trend2_noise_x, trend2_noise_y)
    plt.xlabel('x')
    plt.ylabel('y4')
    plt.title('y=kx+b, k > 0 + шум')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Гармонический процесс дискретный
# -----------------------------------------
def task_garm_process():
    N = 1000  # Количество реализаций
    f = [11, 110, 259, 410]  # Частоты, для четырех гармонических процессов
    f_count = len(f)  # Количество гармоник
    dt = 0.001  # Шаг дискретизации
    a0 = 100  # Амплитудное значение для гармонических процессов
    # Реализуем 4 гармонических процесса
    t = np.arange(0, N, 1)
    x = garm_processes(f_count, f, a0, dt, N)

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(t, x[0], 'b-')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('x = a0 * sin(2 * pi * f01 * t)')

    plt.subplot(2, 2, 2)
    plt.plot(t, x[1], 'b-')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('x = a0 * sin(2 * pi * f02 * t)')

    plt.subplot(2, 2, 3)
    plt.plot(t, x[2], 'b-')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('x = a0 * sin(2 * pi * f03 * t)')

    plt.subplot(2, 2, 4)
    plt.plot(t, x[3], 'b-')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('x = a0 * sin(2 * pi * f04 * t)')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Прямое преобразование Фурье
# -----------------------------------------

def task_ampl_spectr():
    N = 1000
    f = [11, 110, 250, 410]
    dt = 0.001
    a0 = 100
    f_count = 4
    t = np.arange(N)
    t1 = t[:(N // 2)]
    x = garm_processes(f_count, f, a0, dt, N)

    x1 = []
    for i in range(f_count):
        x[i] = ampl_spectr(x[i], N)
        x1.append(x[i][:(N // 2)])

    fig = plt.figure()

    subplot(2, 2, 1)
    plot(t1, x1[0], 'b')
    xlabel('t')
    ylabel('x')
    title('|Xm| для fo = 11 Гц')

    subplot(2, 2, 2)
    plot(t1, x1[1], 'b')
    xlabel('t')
    ylabel('x')
    title('|Xm| для fo = 110 Гц')

    subplot(2, 2, 3)
    plot(t1, x1[2], 'b')
    xlabel('t')
    ylabel('x')
    title('|Xm| для fo = 250 Гц')

    subplot(2, 2, 4)
    plot(t1, x1[3], 'b')
    xlabel('t')
    ylabel('x')
    title('|Xm| для fo = 410 Гц')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Получить полигармонический процесс
# -----------------------------------------

def task_poligarm_processes():
    N = 1000
    count = 3  # Количество гармонических процессов
    a = [25, 35, 30]  # амлитуды гармонических процессов
    f = [11, 41, 141]  # Частоты гармонических процессов
    dt = 0.001
    t = np.arange(N)
    t1 = np.arange(N // 2)

    # Реализуем гармонические функции
    garm_x = garm_processes(count, f, a, dt, N)

    # Реализуем полигармоническую функцию
    poligarm_x = poligarm_procceses(count, f, a, dt, N)

    # Производим преобразование Фурье
    x1 = []
    for i in range(count):
        garm_x[i] = ampl_spectr(garm_x[i], N)
        x1.append(garm_x[i][:(N // 2)])
    poligarm_x = ampl_spectr(poligarm_x, N)[:(N // 2)]

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plot(t1, x1[0], 'b')
    xlabel('t')
    ylabel('x')
    title('|Xm| для fo = 11 Гц')

    plt.subplot(2, 2, 2)
    plot(t1, x1[1], 'b')
    xlabel('t')
    ylabel('x')
    title('|Xm| для fo = 41 Гц')

    plt.subplot(2, 2, 3)
    plot(t1, x1[2], 'b')
    xlabel('t')
    ylabel('x')
    title('|Xm| для fo = 141 Гц')

    plt.subplot(2, 2, 4)
    plot(t1, poligarm_x, 'b')
    xlabel('t')
    ylabel('x')
    title('|Xm| для fo =  Гц')

    tight_layout()
    plt.show()


# -----------------------------------------
# Гистограмма - распределние плотности вероятности:
# Гистаграмму для N=1000, 4 окошка: свой генератор, встроенный и две гистограммы
# Весь диапазон от -s до s разбить на итервалы (на m=40 штук)
# Пробегая все значения, в соответствующий интеравал -
# количество значений, попадающий в это значение
# -----------------------------------------

def task_historam():
    N = 1000
    s = 10
    x = np.arange(N)
    y1 = np.random.uniform(-s, s, N)
    y2 = my_rand(N, s)
    hist1_x, hist1_y = historam(y1, s, N)
    hist2_x, hist2_y = historam(y2, s, N)

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x, y1)
    plt.xlabel('x')
    plt.ylabel('y1')
    plt.title('Встроенный генератор')

    plt.subplot(2, 2, 2)
    plt.plot(x, y2)
    plt.xlabel('x')
    plt.ylabel('y2')
    plt.title('Свой генератор')

    plt.subplot(2, 2, 3)
    plt.plot(hist1_x, hist1_y)
    plt.xlabel('x')
    plt.ylabel('y1')
    plt.title('Гистоограмма для встроенного')

    plt.subplot(2, 2, 4)
    plt.plot(hist2_x, hist2_y)
    plt.xlabel('x')
    plt.ylabel('y1')
    plt.title('Гистоограмма для своего')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Посчитать АКФ для генераторов и взаимную АКФ для своего и стандартного генератора
# Посчитать АКФ
# -----------------------------------------

def task_acf():
    N = 1000
    s = 10
    x = np.arange(N)
    rand_st_x, rand_st_y = rand_stand(N, s)
    rand_my_y = my_rand(N, s)
    hist1_x, hist1_y = historam(rand_st_y, s, N)
    hist2_x, hist2_y = historam(rand_my_y, s, N)

    fig = plt.figure('Гистограмма')

    plt.subplot(2, 2, 1)
    plt.plot(rand_st_x, rand_st_y)
    plt.xlabel('x')
    plt.ylabel('y1')
    plt.title('Встроенный генератор')

    plt.subplot(2, 2, 2)
    plt.plot(x, rand_my_y)
    plt.xlabel('x')
    plt.ylabel('y2')
    plt.title('Свой генератор')

    plt.subplot(2, 2, 3)
    plt.plot(hist1_x, hist1_y)
    plt.xlabel('y1')
    plt.ylabel('p(y1)')
    plt.title('Гистоограмма 1')

    plt.subplot(2, 2, 4)
    plt.plot(hist2_x, hist2_y)
    plt.xlabel('y2')
    plt.ylabel('p(y2)')
    plt.title('Гистоограмма 2')

    fig.tight_layout()
    plt.show()

    acf_x = x
    acf1_y = acf(rand_st_y, N)
    acf2_y = acf(rand_my_y, N)
    mcf_x = x
    mcf_y = mcf(rand_st_y, rand_my_y, N)

    fig = plt.figure('Функции автокорреляции и взаимной корреляции')

    plt.subplot(2, 2, 1)
    plt.plot(acf_x, acf1_y)
    plt.xlabel('x')
    plt.ylabel('y1')
    plt.title('АКФ для встроенного генератора')

    plt.subplot(2, 2, 2)
    plt.plot(acf_x, acf2_y)
    plt.xlabel('x')
    plt.ylabel('y2')
    plt.title('АКФ для своего генератора')

    plt.subplot(2, 2, 3)
    plt.plot(mcf_x, mcf_y)
    plt.xlabel('x')
    plt.ylabel('y2')
    plt.title('Взаимная АКФ для своего генератора')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Функция antishift устраняет помеху сдвига
# -----------------------------------------

def task_antishist():
    N = 1000
    s = 10
    shift_y = 100 * s
    rand_st_x, rand_st_y = rand_stand(N, s)
    rand_st_y = shift(rand_st_y, 1, N, shift_y)
    exp_val = expected_value(rand_st_y)
    rand_st_y[0] = 0

    fig = plt.figure()

    # plt.style.use('ggplot')
    plt.subplot(1, 2, 1)
    plt.plot(rand_st_x, rand_st_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Последовательность со сдвигом')

    mass_antishift_x = rand_st_x
    mass_antishift_y = anti_shift(rand_st_y, N, exp_val)

    plt.subplot(1, 2, 2)
    plt.plot(mass_antishift_x, mass_antishift_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Последовательность без сдвига')

    tight_layout()
    plt.show()


# -----------------------------------------
# Функция antispikes устраняет помеху нехарактерных значений
# -----------------------------------------
def task_antispikes():
    N = 1000
    s = 10
    m = 10
    # Генерируем массив случайных чисел в интервале от -s до s
    rand_st_x, rand_st_y = rand_stand(N, s)

    # Добавляем спайки к массиву
    mass_spikes_x = rand_st_x
    mass_spikes_y = spikes(rand_st_y)

    fig = plt.figure('Устранение спайков')

    plt.subplot(1, 2, 1)
    plt.plot(mass_spikes_x, mass_spikes_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Со спайками')

    # Генерируем массив без спайков
    mass_antispikes_x = rand_st_x
    mass_antispikes_y = antispikes(mass_spikes_y, N, s)

    plt.subplot(1, 2, 2)
    plt.plot(mass_antispikes_x, mass_antispikes_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Без спайков')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Функция antistrend исключает шум ввиде тренда
# -----------------------------------------

def task_antitrend():
    N = 1000
    L = 100
    s = 10
    k, b = -0.1, 1000
    m = 1000 * s

    # Генерируем входной массив
    rand_st_x, rand_st_y = rand_stand(N, s)

    # Генерируем массив линейного тренда
    trend_x, trend_y = lin_trends(N, k, b)

    # Генерируем массив аддитивного шума
    mass_trend_x, mass_trend_y = mass_trend(rand_st_y, trend_y)

    # Вычленяем тренд из зависимости
    mass_antitrend_x, mass_antitrend_y = antitrend(mass_trend_y, N, L)

    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(rand_st_x, rand_st_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Входной массив значений')

    plt.subplot(2, 2, 2)
    plt.plot(mass_trend_x, mass_trend_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Входной массив с шумом')

    plt.subplot(2, 2, 3)
    plt.plot(mass_antitrend_x, mass_antitrend_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Массив без шума')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Функция производит чтение из файла и выявляет частоту и амплитуду гармоник
# -----------------------------------------

def task_open_reader():
    # y = open_reader('C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat', 'f')
    y = open_reader('C:/Users/soloa/OneDrive/Документы/MATLAB/matrica.dat', 'f')

    N = len(y)
    x = np.arange(N)

    x_spectr = ampl_spectr(y, N)
    ind = []
    count = 0
    for i in range(N // 2):
        if x_spectr[i] > 1:
            ind.append(i)
            count += 1

    print('Количество гармоник: ', count)

    for i in ind:
        print('Частота первой гармоники: f = ', i, 'Амплитуда A = ', round(x_spectr[i] * 2))

    x_acf = acf(y, N)

    a0 = 30
    f01 = 5
    dt = 0.001
    x_garm = [(a0 * sin(2 * np.pi * f01 * k * dt)) for k in range(N)]
    x_garm_mcf = mcf(x_garm, y, N)

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Данные из файла')

    plt.subplot(2, 2, 2)
    plt.plot(x[:(N // 2)], x_spectr[:(N // 2)], 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр')

    plt.subplot(2, 2, 3)
    plt.plot(x, x_acf, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('АКФ ')

    plt.subplot(2, 2, 4)
    plt.plot(x, x_garm_mcf, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Взаимная АКФ')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Обратное преобразование Фурье
# -----------------------------------------

def task_inv_fourier_transf():
    N = 1000
    count = 3  # Количество гармонических процессов
    a = [25, 35, 30]  # амлитуды гармонических процессов
    f = [11, 41, 141]  # Частоты гармонических процессов
    dt = 0.001
    t = np.arange(N)
    t1 = np.arange(N // 2)

    # Реализуем гармонические функции
    garm_x = garm_processes(count, f, a, dt, N)

    # Реализуем полигармоническую функцию
    poligarm_x = poligarm_procceses(count, f, a, dt, N)

    # Производим преобразование Фурье для полигармонического сигнала
    x1 = []
    for i in range(count):
        garm_x[i] = ampl_spectr(garm_x[i], N)
        x1.append(garm_x[i][:(N // 2)])
    poligarm_fft = ampl_spectr(poligarm_x, N)[:(N // 2)]

    # Считаем сумму действительной и мнимой частей полигармонического сигнала
    Xm = summ_re_im(poligarm_x, N)
    # Производим обратное преобразование Фурье для спектра полигармонического сигнала
    opf_garm = inv_fourier_transf(Xm, N)

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(t, poligarm_x, 'b')
    plt.xlabel('Время, t')
    plt.ylabel('Полигармонический процесс, x(t)')
    plt.title('Гармоника')

    plt.subplot(2, 2, 2)
    plt.plot(t[: (N // 2)], poligarm_fft[:(N // 2)], 'b')
    plt.xlabel('Частоты, f')
    plt.ylabel('Значения гармоник, x(f)')
    plt.title('Гармоника')

    plt.subplot(2, 2, 3)
    plt.plot(t, opf_garm, 'b')
    plt.xlabel('Время, t')
    plt.ylabel('Полигармонический процесс, x(t)')
    plt.title('ОПФ')

    fig.tight_layout()
    plt.show()


# -----------------------------------------
# Обнуление последних 50 значений и спектр этого массива
# -----------------------------------------

def task_zero_last_50():
    N = 1000  # Количество реализаций
    f = [11, 110]  # Частоты, для четырех гармонических процессов
    # f_count = len(f)
    dt = 0.001  # Шаг дискретизации
    a0 = [100, 50]  # Амплитудное значение для гармонических процессов
    # Реализуем 4 гармонических процесса
    t = np.arange(0, N, 1)
    garm = garm_processes(1, f, a0, dt, N)
    garm_zero = zero_last_50(garm[0], N, 50)
    fft_garm = ampl_spectr(garm_zero, N)

    fig = plt.figure()

    subplot(1, 2, 1)
    plt.plot(t, garm_zero)

    subplot(1, 2, 2)
    plt.plot(t[:(N // 2)], fft_garm[:(N // 2)])

    plt.show()


# ------------------------------------------------------
# Модуль для работы с ЭКГ
# ------------------------------------------------------

def task_ecg():
    N = 1000
    l = 250  # Координата первого вхождения тиков
    x = np.arange(N)  # Массив абсциссы времени
    alpha = 30  # степенной коэффициент экспоненты ЭКГ
    f0 = 10  # Частота гармонической составляющей ЭКГ
    dt = 0.005  # Шаг дискретизации
    ecg_mass = ecg(x, alpha, f0, dt)  # Входной массив - одиночный ЭКГ сигнал
    ticks_mass = ticks(N, l)
    conv_mass = convolution(ecg_mass, ticks_mass)
    conv_mass_x = np.arange(len(conv_mass))

    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x, ecg_mass)

    plt.subplot(2, 2, 2)
    plt.plot(x, ticks_mass)

    plt.subplot(2, 2, 3)
    plt.plot(conv_mass_x, conv_mass)

    fig.tight_layout
    plt.show()


# ------------------------------------------------------
# Спектр Фурье для ЭКГ сигнала
# ------------------------------------------------------

def task_acg_furie():
    N = 1000
    l = 250  # Координата первого вхождения тиков
    x = np.arange(N)  # Массив абсциссы времени
    alpha = 30  # степенной коэффициент экспоненты ЭКГ
    f0 = 10  # Частота гармонической составляющей ЭКГ
    dt = 0.005  # Шаг дискретизации
    ecg_mass = ecg(x, alpha, f0, dt)  # Входной массив - одиночный ЭКГ сигнал
    ticks_mass = ticks(N, l)
    conv_mass = convolution(ecg_mass, ticks_mass)
    conv_mass_x = np.arange(len(conv_mass))

    ecg_spectrum = ampl_spectr(conv_mass, len(conv_mass))

    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x, ecg_mass)

    plt.subplot(2, 2, 2)
    plt.plot(x, ticks_mass)

    plt.subplot(2, 2, 3)
    plt.plot(conv_mass_x, conv_mass)

    plt.subplot(2, 2, 4)
    plt.plot(conv_mass_x[:(N // 2)], ecg_spectrum[:(N // 2)])

    fig.tight_layout
    plt.show()


# ------------------------------------------------------
# Фильтр низких частот
# -----------------------------------------------------

def norm_filter(filter_mass):
    N = len(filter_mass)
    for i in range(N):
        filter_mass[i] *= N
    return filter_mass


def task_low_pass_filter():
    m = 32
    dt = 0.001
    fc = 100
    lpw = low_pass_filter(m, dt, fc)
    step = (fc) / len(lpw)
    x_lpw = np.arange(0, fc, step)
    N = len(x_lpw)

    lpw_spectr = ampl_spectr(lpw, len(lpw))
    norm_filter(lpw_spectr)

    fig = plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(x_lpw, lpw)

    plt.subplot(2, 1, 2)
    plt.plot((x_lpw)[:(N // 2)], lpw_spectr[:(N // 2)])
    plt.show()
    print(len(lpw))


# ------------------------------------------------------
# Чтение из файла и обработка на линейное преобразование с помощью свертки
# -----------------------------------------------------

def task_open_file_convolution():
    input_mass = open_reader('C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat', 'f')

    N = len(input_mass)
    x = np.arange(N)

    # Преобразование Фурье входного массива
    x_spectr = ampl_spectr(input_mass, N)

    # Управляющий массив - фильтр низких частот
    control_mass = low_pass_filter(m=32, dt=0.001, fc=100)
    # Свертка: входного массива и управляющего
    conv_mass = convolution(input_mass, control_mass)[:1000]
    x_conv_mass = np.arange(len(conv_mass))
    # Спектр отфильтрованного массива
    conv_mass_spectr = ampl_spectr(conv_mass, len(conv_mass))

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x, input_mass, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Данные из файла')

    plt.subplot(2, 2, 2)
    plt.plot(x[:(N // 2)], x_spectr[:(N // 2)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр массива')

    plt.subplot(2, 2, 3)
    plt.plot(x_conv_mass, conv_mass)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Отфильтрованный массив')

    plt.subplot(2, 2, 4)
    plt.plot(x_conv_mass[:(N // 2)], conv_mass_spectr[:(N // 2)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр отфильтрованного массива')

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------
# Фильтры высоких частот, полосовой и режекторный
# -----------------------------------------------------

def task_filters():
    m = 32
    dt = 0.001

    # Фильтр высоких частот
    hpw = high_pass_filter(m, dt, fc=100)
    x_hpw = np.arange(len(hpw))
    hpw_spectr = ampl_spectr(hpw, len(hpw))
    N = len(hpw)
    norm_filter(hpw_spectr)

    # Полосовой фильтр
    bpw = bend_pass_filter(m=32, dt=0.001, fc1=100, fc2=200)
    x_bpw = np.arange(len(bpw))
    bpw_spectr = ampl_spectr(bpw, len(bpw))
    norm_filter(bpw_spectr)

    # Режекторный фильтр
    bsw = bend_stop_filter(m, dt, fc1=100, fc2=200)
    x_bsw = np.arange(len(bsw))
    bsw_spectr = ampl_spectr(bsw, len(bsw))
    norm_filter(bsw_spectr)

    fig = plt.figure()

    plt.subplot(3, 2, 1)
    plt.plot(x_hpw, hpw)
    plt.title('Фильтр высоких частот')

    plt.subplot(3, 2, 2)
    plt.plot(x_hpw[:(N // 2)], hpw_spectr[:(N // 2)])
    plt.title('Фильтр высоких частот')

    plt.subplot(3, 2, 3)
    plt.plot(x_bpw, bpw)
    plt.title('Полосовой фильтр')

    plt.subplot(3, 2, 4)
    plt.plot(x_bpw[:(N // 2)], bpw_spectr[:(N // 2)])
    plt.title('Полосовой фильтр')

    plt.subplot(3, 2, 5)
    plt.plot(x_bsw, bsw)
    plt.title('Режекторный фильтр')

    plt.subplot(3, 2, 6)
    plt.plot(x_bsw[:(N // 2)], bsw_spectr[:(N // 2)])
    plt.title('Режекторный фильтр')

    plt.tight_layout()
    plt.show()


def task_open_file_filter():
    # считывание файла для его обработки, N - длина массива из файла
    input_mass_y = open_reader('C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat', 'f')  # Частоты 5, 55, 250
    N = len(input_mass_y)
    input_mass_x = np.arange(N)
    # Делаем преобразование Фурье для входного массива
    input_mass_furie = ampl_spectr(input_mass_y, N)

    # Фильтр низких частот
    # m, dt, fc = 64, 0.001, 200  # Параметры фильтра
    lpf = low_pass_filter(m=64, dt=0.001, fc=60)
    # Применяем фильтр низких частот входного массива ФНЧ
    conv_input_mass_lpf = convolution(input_mass_y, lpf)
    N_lpf = len(conv_input_mass_lpf)
    x_lpf = np.arange(N_lpf)
    # Преобразование Фурье для отфильтрованного массива
    mass_lpf_furie = ampl_spectr(conv_input_mass_lpf, N_lpf)

    # Фильтр высоких частот
    hpf = high_pass_filter(m=64, dt=0.001, fc=240)

    # Применяем фильтр высоких частот
    conv_input_mass_hpf = convolution(input_mass_y, hpf)
    x2 = np.arange(len(conv_input_mass_hpf))

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(input_mass_x, input_mass_y)
    plt.title('Входной массив')

    plt.subplot(2, 2, 2)
    plt.plot(input_mass_x, input_mass_furie)
    plt.title('Преобразование Фурье входного массива')

    plt.subplot(2, 2, 3)
    plt.plot(x_lpf, conv_input_mass_lpf)
    plt.title('Массив без верхней частоты f=250')

    plt.subplot(2, 2, 4)
    plt.plot(x_lpf, mass_lpf_furie)
    plt.title('Спектр массива без f=250')

    plt.tight_layout()
    plt.show()


# Применение фильтра высоких частот
def task_file_convolution_hpf():
    input_mass = open_reader('C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat', 'f')  # Частоты 5, 55, 250

    N = len(input_mass)
    x = np.arange(N)

    # Преобразование Фурье входного массива
    x_spectr = ampl_spectr(input_mass, N)

    # Управляющий массив - фильтр высоких частот
    control_mass = high_pass_filter(m=32, dt=0.001, fc=240)
    # Свертка: входного массива и управляющего
    conv_mass = convolution(input_mass, control_mass)[:1000]
    N_conv_mass = len(conv_mass)
    x_conv_mass = np.arange(len(conv_mass))
    # Спектр отфильтрованного массива
    conv_mass_spectr = ampl_spectr(conv_mass, len(conv_mass))

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x, input_mass, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Данные из файла')

    plt.subplot(2, 2, 2)
    plt.plot(x, x_spectr)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр массива')

    plt.subplot(2, 2, 3)
    plt.plot(x_conv_mass, conv_mass)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Отфильтрованный массив')

    plt.subplot(2, 2, 4)
    plt.plot(x_conv_mass[:(N // 2)], conv_mass_spectr[:(N // 2)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр отфильтрованного массива')

    plt.tight_layout()
    plt.show()


# Применение полосового фильтра bpf
def task_file_convolution_bpf():
    input_mass = open_reader('C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat', 'f')  # Частоты 5, 55, 250

    N = len(input_mass)
    x = np.arange(N)

    # Преобразование Фурье входного массива
    x_spectr = ampl_spectr(input_mass, N)

    # Управляющий массив - фильтр высоких частот
    control_mass = bend_pass_filter(m=32, dt=0.001, fc1=50, fc2=60)
    # Свертка: входного массива и управляющего
    conv_mass = convolution(input_mass, control_mass)[:1000]
    N_conv_mass = len(conv_mass)
    x_conv_mass = np.arange(len(conv_mass))
    # Спектр отфильтрованного массива
    conv_mass_spectr = ampl_spectr(conv_mass, len(conv_mass))

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x, input_mass, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Данные из файла')

    plt.subplot(2, 2, 2)
    plt.plot(x, x_spectr)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр массива')

    plt.subplot(2, 2, 3)
    plt.plot(x_conv_mass, conv_mass)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Отфильтрованный массив')

    plt.subplot(2, 2, 4)
    plt.plot(x_conv_mass[:(N // 2)], conv_mass_spectr[:(N // 2)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр отфильтрованного массива')

    plt.tight_layout()
    plt.show()


# Применение режекторного фильтра bsf
def task_file_convolution_bsf():
    input_mass = open_reader('pgp_f4-1K-1ms.dat', 'f')  # Частоты 5, 55, 250

    N = len(input_mass)
    x = np.arange(N)

    # Преобразование Фурье входного массива
    x_spectr = ampl_spectr(input_mass, N)

    # Управляющий массив - фильтр высоких частот
    control_mass = bend_stop_filter(m=64, dt=0.001, fc1=30, fc2=80)
    # Свертка: входного массива и управляющего
    conv_mass = convolution(input_mass, control_mass)[:1000]
    N_conv_mass = len(conv_mass)
    x_conv_mass = np.arange(len(conv_mass))
    # Спектр отфильтрованного массива
    conv_mass_spectr = ampl_spectr(conv_mass, len(conv_mass))

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x, input_mass, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Данные из файла')

    plt.subplot(2, 2, 2)
    plt.plot(x[:(N // 2)], x_spectr[:(N // 2)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр массива')

    plt.subplot(2, 2, 3)
    plt.plot(x_conv_mass, conv_mass)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Отфильтрованный массив')

    plt.subplot(2, 2, 4)
    plt.plot(x_conv_mass[:(N // 2)], conv_mass_spectr[:(N // 2)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр отфильтрованного массива')

    plt.tight_layout()
    plt.show()


# Считывание wav файла
def task_wav_file():
    filename = 'ma.wav'  # Имя файла - его месторасположение

    data, fs = sf.read(filename, dtype='float32')  # data - массив звукового файла, fs - частота дискретизации
    sd.play(data, fs)  # Воспроизведение файла с массивом data и частотой дискретизации fs
    status = sd.wait()  # Ожидание, пока идет воспроизвдение
    print(data)

    N_data = len(data)  # Количество элементов в data
    data_x = np.arange(N_data)  # Создаем ось абсцисс для графика данных wav файла
    data_spectr = np.fft.fft(data)  # Преобразование Фурье для визуализации спектра wav файла

    lpf = low_pass_filter(m=64, dt=0.001, fc=200)  # Фильтр нихзких частот как управляющий массив для свертки
    # Свертка: входного массива и управляющего
    conv_mass = convolution(data, lpf)  # Свертка для фильтрации низких частот
    conv_mass_x = np.arange(len(conv_mass))

    conv_mass_spectr = np.fft.fft(conv_mass)

    # sd.play(conv_mass, fs)
    # status=sd.wait()
    # data1 = np.append(data, data)
    # data2 = np.append(data, data[::-1])
    # for i in range(20):
    #     data1 = np.append(data1, data[:])
    #     data2 = np.append(data2, data[::-1])
    # data3 = data1 + data2

    for i in range(len(conv_mass)):
        conv_mass[i] *= 20
    sf.write("ma2.wav", conv_mass, fs)

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(data_x, data)

    plt.subplot(2, 2, 2)
    plt.plot(data_x[:N_data // 2], data_spectr[:N_data // 2])

    # plt.subplot(2, 2, 3)
    # plt.plot(conv_mass_x, conv_mass)
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(conv_mass_x, conv_mass_spectr)

    plt.show()


def task_my_wav():
    filename = 'form1.wav'  # Имя файла - его месторасположение

    data, fs = sf.read(filename, dtype='float32')  # data - массив звукового файла, fs - частота дискретизации
    # sd.play(data, fs)                               # Воспроизведение файла с массивом data и частотой дискретизации fs
    # status = sd.wait()                              # Ожидание, пока идет воспроизвдение

    # data = data1[:6000]
    N_data = len(data)  # Количество элементов в data
    data_x = np.arange(N_data)  # Создаем ось абсцисс для графика данных wav файла
    data_spectr = np.fft.fft(
        data)  # Преобразование Фурье для визуализации спектра wav файла 120-140, 250-270, 300-320, 450-470

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(data_x, data)

    plt.subplot(2, 2, 2)
    plt.plot(data_x[:N_data // 2], data_spectr[:N_data // 2])

    plt.show()

    m, dt, fc1, fc2 = 32, 0.001, 120, 140
    bpf = bend_pass_filter(m, dt, fc1, fc2)
    #
    conv_mass = convolution(data, bpf)

    # m, dt, fc = 32, 0.001, 30
    # lpf = low_pass_filter(m, dt, fc)
    # conv_mass = convolution(data, lpf)
    N1 = len(conv_mass)
    x_conv_mass = np.arange(N1)
    #
    # plt.plot(x_conv_mass, conv_mass)
    # plt.plot(x_conv_mass, np.fft.fft(conv_mass))
    # plt.show()
    #
    #
    for i in range(N1):
        conv_mass[i] *= 40
    sf.write("conv.wav", conv_mass, fs)


def task_exam():
    # Открыть файл и отобразить данные
    # проанализировать файл, на налоичие составляющих тренд, шум, тренд, определить амплитуды и частоты гармоники в районе 70 Гц

    input_mass = open_reader('v1x2.dat', 'f')  # Открытие файла
    N = len(input_mass)  # Количество элементов входного массива
    x_input_mass = np.arange(N)
    print(N)

    # Вычленяем тренд из зависимости
    L = 30
    mass_antitrend_x, mass_antitrend_y = antitrend(input_mass, N, L)

    # Спектр входного массива
    spectr = ampl_spectr(input_mass, N)
    x_spectr = np.arange(N)

    # Спектр антитренда
    antitrend_spectr = ampl_spectr(mass_antitrend_y, (N - L))

    # # Тренд
    # a = 0  # переменная для усреднения значений в окне
    # mass_antitrend_y = []
    # for i in range(N - L):
    #     for j in range(L):
    #         a += input_mass[i + j]
    #     a /= L
    #     mass_antitrend_y.append(a)
    # mass_antitrend_x = np.arange(N - L)

    # Вычленение частоты и амплитуды
    ind = []
    count = 0
    for i in range(20, (N - L) // 2):
        if antitrend_spectr[i] > 20:
            ind.append(i)
            count += 1
    print('Количество гармоник: ', count)
    for i in ind:
        print('Частота первой гармоники: f = ', i, 'Амплитуда A = ', (antitrend_spectr[i] * 2))

    # Построение графиков
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x_input_mass, input_mass)

    plt.subplot(2, 2, 2)
    plt.plot(x_spectr[:(N // 2)], spectr[:(N // 2)])

    plt.subplot(2, 2, 3)
    plt.plot(mass_antitrend_x, mass_antitrend_y)

    plt.subplot(2, 2, 4)
    plt.plot(x_spectr[:((N - L) // 2)], antitrend_spectr[:((N - L) // 2)])

    plt.tight_layout()
    plt.show()

    # Второй способ устранения тренда
    # Фильтр высоких частот
    hpf = high_pass_filter(m=64, dt=0.001, fc=1)

    # Применяем фильтр высоких частот
    conv_input_mass_hpf = convolution(input_mass, hpf)[:N]
    x2 = np.arange(len(conv_input_mass_hpf))[:N]

    # Построение графиков
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x_input_mass, input_mass)

    plt.subplot(2, 2, 2)
    plt.plot(x_spectr[:(N // 2)], spectr[:(N // 2)])

    plt.subplot(2, 2, 3)
    plt.plot(x2, conv_input_mass_hpf)

    x2_spectr = ampl_spectr(x2, N)
    conv_spectr = ampl_spectr(conv_input_mass_hpf, N)

    plt.subplot(2, 2, 4)
    plt.plot(x2_spectr[:(N // 2)], conv_spectr[:(N // 2)])

    plt.tight_layout()
    plt.show()


def polig_pr():
    i = 3
    f = [1, 5, 66]
    a = (1817 * 2), 10, 70
    dt = 0.001
    N = 1000
    input_mass = poligarm_procceses(i, f, a, dt, N)
    x_input_mass = np.arange(N)
    noise = rand_stand(N, s=100)

    # спектр
    spectr = ampl_spectr(input_mass, N)

    # Управляющий массив - фильтр высоких частот
    control_mass = high_pass_filter(m=64, dt=0.001, fc=40)
    # Свертка: входного массива и управляющего
    conv_mass = convolution(input_mass, control_mass)[:1000]
    N_conv_mass = len(conv_mass)
    x_conv_mass = np.arange(len(conv_mass))
    # Спектр отфильтрованного массива
    conv_mass_spectr = ampl_spectr(conv_mass, len(conv_mass))

    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(x_input_mass, input_mass, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Данные из файла')

    plt.subplot(2, 2, 2)
    plt.plot(x_input_mass[:(N // 2)], spectr[:(N // 2)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр массива')

    plt.subplot(2, 2, 3)
    plt.plot(x_conv_mass, conv_mass)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Отфильтрованный массив')

    plt.subplot(2, 2, 4)
    plt.plot(x_conv_mass[:(N // 2)], conv_mass_spectr[:(N // 2)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Спектр отфильтрованного массива')

    plt.tight_layout()
    plt.show()


# Реализуем свертку ЭКГ, затем добавляем для управляющего сигнала
# в конец массива нули, потом берем преобразование Фурье,
# считаем действительную и мнимую части и складываем, делаем ОПФ
def task_deconvolution_of_ecg():
    # N = 1000
    # l = 250  # Координата первого вхождения тиков
    # x = np.arange(N)  # Массив абсциссы времени
    # alpha = 30  # степенной коэффициент экспоненты ЭКГ
    # f0 = 10  # Частота гармонической составляющей ЭКГ
    # dt = 0.005  # Шаг дискретизации
    # ecg_mass = ecg(x, alpha, f0, dt)  # Входной массив - одиночный ЭКГ сигнал
    # ticks_mass = ticks(N, l)
    # conv_mass = convolution(ecg_mass, ticks_mass)
    # conv_mass_x = np.arange(len(conv_mass))
    #
    #
    # # Преобразование Фурье от задающего сигнала и выходного
    # N2 = len(conv_mass_x)
    # spectr_x = np.arange(N2)
    # spectr_y = ampl_spectr(conv_mass, len(conv_mass))
    # spectr_h = ampl_spectr(ecg_mass, len(ecg_mass))
    # print(len(spectr_h), len(spectr_y))
    #
    # # дейстивтельная и мнимая части задающего и выхожного сигнала
    # re_spectr_h, im_spectr_h = re_and_im(spectr_h, N2)
    # re_spectr_y, im_spectr_y = re_and_im(spectr_y, N2)
    #
    # # Получение входного сигнала
    # re_x, im_x = [], []
    # for i in range(N):
    #     re_x.append((re_spectr_h[i] * re_spectr_y[i] + im_spectr_h[i] * im_spectr_y[i]) / (re_spectr_h[i] ** 2 + im_spectr_h[i] ** 2))
    #     im_x.append((re_spectr_h[i] * im_spectr_y[i] - re_spectr_y[i] * im_spectr_h[i]) / (re_spectr_h[i] ** 2 + im_spectr_h[i] ** 2))
    #
    # # Заполняем массив значениями модулей гармонического процесса
    # C = [0 for i in range(N2)]
    # for m in range(N2):
    #     C[m] = sqrt(re_x[m] + im_x[m])
    #
    #
    # inv_x = fft.ifft(C)
    # plt.plot(spectr_x, inv_x)
    # plt.show()

    N = 1000
    l = 250  # Координата первого вхождения тиков
    x = [i for i in range(N)]  # Массив абсциссы времени
    # x = np.arange(N)
    alpha = 30  # степенной коэффициент экспоненты ЭКГ
    f0 = 10  # Частота гармонической составляющей ЭКГ
    dt = 0.005  # Шаг дискретизации
    ecg_mass = ecg1(x, N, alpha, f0, dt)  # Управляющий массив - одиночный ЭКГ сигнал
    ticks_mass = ticks(N, l)  # Входной сигнал
    conv_mass = convolution(ecg_mass, ticks_mass)  # свертка, получаем сигнал ЭКГ

    N_conv = len(conv_mass)
    conv_mass_x = [i for i in range(N_conv)]

    # Дополнить нулями ecg функию
    ecg_mass += [0 for i in range(N - 1)]

    # Взять преобразование Фурье от выходного сигнала conv_mass и управляющего сигнала ecg_mass
    spectr_ecg_mass = ampl_spectr(ecg_mass, N_conv)
    spectr_conv_mass = ampl_spectr(conv_mass, N_conv)

    # Вычислить действительную и мнимую части управляющего и выходного сигналов
    ecg_mass_spectr_re, ecg_mass_spectr_im = re_and_im(spectr_ecg_mass, N_conv)
    conv_mass_spectr_re, conv_mass_spectr_im = re_and_im(spectr_conv_mass, N_conv)

    # Вычисляем отношение conv_mass_spectr / ecg_mass_spectr
    compl_re, compl_im = complex_ratio(ecg_mass_spectr_re, ecg_mass_spectr_im, conv_mass_spectr_re, conv_mass_spectr_im,
                                       N_conv)

    # Вычисляем моудль комплексного числа
    module_of_compl = []
    for i in range(N_conv):
        module_of_compl.append(compl_re[i] + compl_im[i])

    # Берем обратное преобразование Фурье от модуля комплексного числа
    compl_spectr = inv_fourier_transf(module_of_compl, N_conv)

    fig = plt.figure()

    plt.plot(conv_mass_x[:1000], compl_spectr[:1000])

    plt.show()


# ----------------------------------------------------------------------------------------------------
# Мастшибирование изображения
def task_scaling_image_1():
    # метод ближайшего соседа

    image = read_jpg_gray('grace.jpg')  # открытие изображения

    const = 2  # коэффициент масштабироваиня
    image_resized_1 = image_scale(image, const, type='nn', mode='increased')
    image_resized_1.show()

    const = 2  # коэффициент масшабирования
    image_resized_2 = image_scale(image, const, type='nn', mode='decreased')
    image_resized_2.show()


def task_scaling_image_2():
    # метод билинейной интерполяции

    image = read_jpg_gray('grace.jpg')  # открытие изображения
    const = 2  # коэффициент масштабирование
    image_resized_1 = image_scale(image, const, type='bi', mode='increased')
    image_resized_1.show()

    const = 2  # коэффициент масштабирования
    image_resized_1 = image_scale(image, const, type='bi', mode='decreased')
    image_resized_1.show()


# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# Отображение изображения в негативе
def task_negative_image():
    '''
    Считываем изображение с помощью read_jpg_gray()
    Сохраняем в двумерную numpy матрицу
    Делаем норировку матрицы
    Делаем преобразование всех пикселей 255-pix[i]
    Выводим изображение
    :return:
    '''
    file_names = ['image_contrast/image1.jpg', 'image_contrast/image2.jpg']  # Список названий изображений
    for file_name in file_names:
        image = read_jpg_gray(file_name)  # Открываем изображение
        image.show()
        width, height = image.size[0], image.size[1]  # Количество столбцов и строк матрицы пикселей
        matrix_pixels = np.array(image).reshape(height, width)  # Сохраняем значения пикселей как numpy массив
        a = np.array(range(width * height)).reshape((height, width))

        # Вызов функции для негатива
        negative_pixels = negative_matrix_pix(matrix_pixels)

        # Отображение numpy массива в массив pil
        # image_negative = Image.fromarray(negative_pixels)  # В качестве аргумента numpy массив
        image_negative = drawing_image_new(negative_pixels, width, height)
        image_negative.show()


# Гамма коррекция изображения
def task_gamma_corrextion():
    file_names = ['image_contrast/image1.jpg', 'image_contrast/image2.jpg']  # Список названий изображений
    for file_name in file_names:
        image = read_jpg_gray(file_name)  # Открываем изображение
        image.show()
        width, height = image.size[0], image.size[1]  # Количество столбцов и строк матрицы пикселей
        matrix_pixels = np.array(image).reshape(height, width)  # Сохраняем значения пикселей как numpy массив

        # Вызов функции для негатива
        const, gamma = 7, 0.8
        gamma_corr_pixels = gamma_correction(matrix_pixels, const, gamma)

        # Отображение numpy массива в массив pil
        # image_gamma = Image.fromarray(gamma_corr_pixels)
        image_gamma = drawing_image_new(gamma_corr_pixels, width, height)
        image_gamma.show()


# Логарифмическая коррекция изображения
def task_log_corrextion():
    file_names = ['image_contrast/image1.jpg', 'image_contrast/image2.jpg']  # Список названий изображений
    for file_name in file_names:
        image = read_jpg_gray(file_name)  # Открываем изображение
        image.show()
        width, height = image.size[0], image.size[1]  # Количество столбцов и строк матрицы пикселей
        matrix_pixels = np.array(image).reshape(height, width)  # Сохраняем значения пикселей как numpy массив

        # Вызов функции для негатива
        const = 50
        log_corr_pixels = log_correction(matrix_pixels, const)

        # Отображение numpy массива в массив pil
        # image_log = Image.fromarray(log_corr_pixels)
        image_log = drawing_image_new(log_corr_pixels, width, height)
        image_log.show()


# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# Эквализация изображения - автоматическая настройка яркости
def task_equalization():
    # Считываем  файл
    image = read_jpg_gray('image_contrast/HollywoodLC.jpg')
    # image = read_jpg_gray('image_contrast/image2.jpg')

    image.show()
    x, y, pix = image_histogram(image)

    plt.plot(x, y)
    plt.show()

    image_new = equalization(image)
    image_new.show()

    x_new, y_new, pix_new = image_histogram(image_new)

    plt.plot(x_new, y_new)
    plt.show()

    # im = cv2.imread('image_contrast/HollywoodLC.jpg')
    # # calculate mean value from RGB channels and flatten to 1D array
    # vals = im.mean(axis=2).flatten()
    # # plot histogram with 255 bins
    # b, bins, patches = plt.hist(vals, 255)
    # plt.xlim([0, 255])
    # plt.show()


# ---------------------------------------------------------------------------------------
''' На фрагменте изображения выполнить следующие преобразования:
    1) АКФ
    2) Спектр АКФ (на которой нужно увидеть пик на какой-то частоте
    3) Построчный вывод
    4) Производная строка (подавление тренда): от производной берем АКФ ->
    -> обнаруживаем период -> берем спектр -> модуль строки производной АКФ
    5) Часоту среза задавать в нормируемой шкале
    6) Частота дискретизации 1 px
    7) Параметры режекторного фильтра M = 16/32/...
       dx = 1
       fs1, fs2 - частоты среза режекторного фильтра
       w - веса
    8) Достаточно отфильтровать каждую строку в 1 направлении
    1. Прочитать файл.xcr построчно с каким-то инкриментом по y dy=10 (каждая 10-я строка)
    2. Считаем производную x по k
    3. Считаем АКФ Rx'x'
    4. Фурье спектр (модуль F[Rx'x']) и взаимной корреляции первых двух строчек (модуль F[Rx'y'])
    5. Определяем параметры пика (f_grid)
    6. Выбираем частоты среза fs1 и fs2
    7. Реализуем режекторный фильтр bend_stop_filter (его веса)
    8. Отфильтровать изображение
    9. Сделать контрастным (lod, автоматическая настройка - эквализация)
    10. image 400x300
'''


# -----------------------------------------------------------------------------------------------------
def task_read_xcr():
    # Считываем изображение
    image = open_reader_xcr('h400x300.xcr')
    width, height = 400, 300
    matrix_pix = np.array(image).reshape(height, width)

    # Создаем входное изображение 400x300
    input_image = drawing_image_new(matrix_pix, width, height)
    input_image.save('moed_test.jpg')
    input_image.show()

    # Гистограмма для входного изображения
    image_hist_x, image_hist_y, pixels_1d = image_histogram(input_image)

    plt.plot(image_hist_x, image_hist_y)
    plt.title('Гистограмма input_image')
    # plt.savefig('moed2.jpg')
    plt.show()

    derivative_mass = derivative(matrix_pix, width, height)
    # pix = from_2d_to_1d(derivative_mass, width, height - 1)

    # Создаем изображение производной 400x300
    input_image = drawing_image_new(derivative_mass, width - 1, height)
    # input_image.save('moed3.jpg')
    input_image.show()

    # Создаем спектр для производной
    spectrum = ampl_spectr(derivative_mass[0], width - 1)
    # spectrum_x = range((width - 1))

    spectrum_x = np.linspace(0, 1, (width - 1))

    len_spectr = (width - 1) // 2

    plt.plot(spectrum_x[:len_spectr], spectrum[:len_spectr])
    plt.title('Спектр')
    plt.xlabel('Пискель')
    plt.ylabel('Яркость')
    # plt.savefig('moed4.jpg')
    plt.show()

    max_spectr = max(spectrum)
    for i in range(len_spectr):
        if spectrum[i] == max_spectr:
            fc = i
            break

    # Организуем режекторный фильтр
    fs = width
    dx = 1 / fs
    m, fc1, fc2 = 32, fc - 15, fc + 15
    input_mass = matrix_pix
    control_mass = bend_stop_filter(m, dx, fc1, fc2)

    conv_matrix_pix = image_conv(input_mass, control_mass, width - 1, height, m)

    # Создаем изображение 400x300
    conv_image = drawing_image_new(conv_matrix_pix, width - 1, height)
    # conv_image.save('moed5.jpg')
    conv_image.show()

    # Создаем АКФ для производной
    w_acf = width - 1
    acf_of_derivative = acf(derivative_mass[0], w_acf)
    # Считаем спектр АКФ для производной
    # mass_x = range(w_acf)

    mass_x = np.linspace(0, 1, (width - 1))

    spectr_acf_of_derivative = ampl_spectr(acf_of_derivative, w_acf)
    # mass_x = normalization(mass_x, dim=1, N=0.5)

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(mass_x, acf_of_derivative)  # График АКФ производной
    plt.title('График АКФ производной')

    plt.subplot(1, 2, 2)
    plt.plot(mass_x[:(w_acf // 2)], spectr_acf_of_derivative[:(w_acf // 2)])  # График спектра АКФ производной
    plt.title('Спектр АКФ производной')
    # plt.savefig('moed6.jpg')
    plt.show()

    #
    # # Создаем ВКФ для производной
    # w_mcf = w_acf
    # mass1_for_mcf = derivative_mass[0]
    # mass2_for_mcf = derivative_mass[1]
    # mcf_of_derivative = mcf(mass1_for_mcf, mass2_for_mcf, w_mcf)
    # # Считаем сектр ВКФ для производной
    # spectr_mcf_of_derivative = ampl_spectr(mcf_of_derivative, w_mcf)
    #
    # fig = plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.plot(mass_x, mcf_of_derivative)
    # plt.title('График ВКФ производной')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(mass_x[:(w_mcf // 2)], spectr_mcf_of_derivative[:(w_mcf // 2)])
    # plt.title('Спектр ВКФ производной')
    # # plt.savefig('moed7.jpg')
    # plt.show()
    #
    image_equal = equalization(conv_image)
    image_equal.show()


# ----------------------------------------------------------------------------------------------------
# Фильтрация изображения с помощью фильтра низких частот
'''
Подавить 3 вида аддитивных шумов уровня 1%, 5%, 15%
1) Нормально распределенные шумы
2) Биполярные шумы "соль и перец"
3) Сумма первых двух
  на модельном изображении model.jpg с помощью фильтра низких частот
  Частоты среза задаются в нормированной шкале Найквиста от 0 до 0.5, параметр m
  подобрать самостоятельно до лучшего результата
  
  1. Считываем матрицу пикселей входного изображения
  2. Строим гистограмму входного изображения
  3. Добавляем модельный шум в матрицу изображения 1%, 5%, 15%
  4. Строим Гистограмму для зашумленных изображений
  5. Считаем спектры зашумленных изображений
  6. Реализуем фильтр низких частот
  7. Производим фильтрацию
  8. Визуализируем гистограммы и спектр отфильтрованных изображений
'''


# ----------------------------------------------------------------------------------------------------
def task_add_model_noise():
    # Загружаем картинку
    file = 'image_model_noise/MODEL.jpg'  # Входное модельное изображение
    image = read_jpg_gray(file)
    width, height = image.size[0], image.size[1]
    image.show()  # Выводим на экран входную картинку

    # гистограмма для входного изображения
    hist_image_x, hist_image_y, hist_image_pix = image_histogram(image)

    plt.bar(hist_image_x, hist_image_y)
    plt.show()

    # Добавление шума
    types_of_noise = ['gauss', 'impulse', 'gauss + impulse']
    percent_noise = [1, 5, 15]

    for type_of_noise in types_of_noise:
        for percent in percent_noise:
            if type_of_noise == 'gauss':
                mu, level = 10, percent
                image_noised = add_gauss_noise(file, mu, level)
            elif types_of_noise == 'impulse':
                Pa, Pb = percent / 200, percent / 200
                image_noised = add_impulse_noise(file, Pa, Pb)
            else:
                mu, level = 10, percent
                Pa, Pb = percent / 200, percent / 200
                image_noised = add_gauss_noise(file, mu, level)
                image_noised = add_impulse_noise(image_noised, Pa, Pb)

            image_noised.show()

            # Строим гистограмму для зашумленного изображения
            hist_image_noised_x, hist_image_noised_y, pix_noised = image_histogram(image_noised)

            plt.bar(hist_image_noised_x, hist_image_noised_y)
            plt.title(f'Гистограмма зашумленного изображения {percent}%')
            plt.xlabel(f'{type_of_noise}')
            plt.show()

            # Считаем спектр зашумленного сигнала
            noised_pixels = np.array(image_noised).reshape(height, width)

            image_noised_spectrum = ampl_spectr(noised_pixels[100], width)

            plt.plot(image_noised_spectrum[:width // 2])
            plt.title(f'Спектр зашумленного изображения {percent}%')
            plt.xlabel(f'{type_of_noise}')
            plt.show()

            # Считаем АКФ зашумленного изображения
            acf_image_noised = acf(image_noised_spectrum, width)
            spectrum_acf_image_noised = ampl_spectr(acf_image_noised, width)

            plt.subplot(1, 2, 1)
            plt.plot(acf_image_noised[:width // 2])
            plt.title(f'АКФ зашумленного изображения {percent}%')
            plt.xlabel(f'{type_of_noise}')

            plt.subplot(1, 2, 2)
            plt.plot(spectrum_acf_image_noised[:width // 2])
            plt.xlabel(f'{type_of_noise}')

            plt.show()

            fs = width
            m = 32
            dx = 1 / fs
            fc = 20
            input_mass = noised_pixels
            control_mass = low_pass_filter(m, dx, fc)
            conv_matrix = image_conv(input_mass, control_mass, width, height, m)

            conv_image = drawing_image_new(conv_matrix, width, height)
            conv_image.show()

            hist_image_filt_x, hist_image_filt_y, filt_pix = image_histogram(conv_image)

            plt.bar(hist_image_filt_x, hist_image_filt_y)
            plt.title(f'Гистограмма отфильтрованного изображения {percent}%')
            plt.xlabel(f'{type_of_noise}')
            plt.show()


# -----------------------------------------------------------------------------------------------

# Реализация пространственного фильтра среднего
'''
Подавить 3 аддитивных шума уровня 1%, 5%, 15%
  1) случайные шумы
  2) биполярные шумы
  3) сумма первых двух
на модельном изображении model.jpg пространственными фильтрами с возможностью
изменения размера масок:
  а) усредняющий арифметический фильтр (*адаптивный)
  б) медианный (*адаптивный)
  *- опционно повышенной трудности
'''


# -----------------------------------------------------------------------------------------------
def task_spatial_filter_average():
    pref = 'image_model_noise/'
    files = ['gauss_1.jpg', 'gauss_5.jpg', 'gauss_15.jpg', 'impulse_1.jpg', 'impulse_5.jpg', 'impulse_15.jpg',
             'gauss_impulse_1.jpg', 'gauss_impulse_5.jpg', 'gauss_impulse_15.jpg']
    for i in range(len(files)):
        files[i] = f'{pref}{files[i]}'
    mask_sizes = [3, 5]
    for file in files:
        image = read_jpg_gray(file)
        image.show()

        # Гистограмма входного зашумленного изображения
        image_hist_x, image_hist_y, pixels_1d = image_histogram(image)

        for mask_size in mask_sizes:
            image_new = spatial_filter_average(image, mask_size)
            image_new.show()
            # image_new.save(f'aver_mask_{mask_size}_{file}')

            # Гистограмма отфильтрованного изображения
            image_new_hist_x, image_new_hist_y, pixels_new_1d = image_histogram(image_new)

            fig = plt.figure('Гистограмма зашумленного и отфильтрованного изображений')
            plt.subplot(121)
            plt.bar(image_hist_x, image_hist_y)
            plt.title('Hist of noise image')
            plt.xlabel(f'{file}')

            plt.subplot(122)
            plt.bar(image_new_hist_x, image_new_hist_y)
            plt.title('Hist of filtered image')
            plt.xlabel(f'filtered {file}')

            plt.tight_layout()
            plt.show()


# -----------------------------------------------------------------------------------------------------

# Реализация пространственного медианного фильтра

# -----------------------------------------------------------------------------------------------------

def task_spatial_filter_median():
    pref = 'image_model_noise/'
    files = ['gauss_1.jpg', 'gauss_5.jpg', 'gauss_15.jpg', 'impulse_1.jpg', 'impulse_5.jpg', 'impulse_15.jpg',
             'gauss_impulse_1.jpg', 'gauss_impulse_5.jpg', 'gauss_impulse_15.jpg']
    for i in range(len(files)):
        files[i] = f'{pref}{files[i]}'
    mask_sizes = [3, 5]
    for file in files:
        image = read_jpg_gray(file)
        image.show()

        # Гистограмма входного зашумленного изображения
        image_hist_x, image_hist_y, pixels_1d = image_histogram(image)

        for mask_size in mask_sizes:
            image_new = spatial_filter_median(image, mask_size)
            image_new.show()
            # image_new.save(f'med_mask_{mask_size}_{file}')

            # Гистограмма отфильтрованного изображения
            image_new_hist_x, image_new_hist_y, pixels_new_1d = image_histogram(image_new)

            fig = plt.figure('Гистограмма зашумленного и отфильтрованного изображений')
            plt.subplot(121)
            plt.bar(image_hist_x, image_hist_y)
            plt.title('Hist of noise image')
            plt.xlabel(f'{file}')

            plt.subplot(122)
            plt.bar(image_new_hist_x, image_new_hist_y)
            plt.title('Hist of filtered image')
            plt.xlabel(f'filtered {file}')

            plt.tight_layout()
            plt.show()


# -----------------------------------------------------------------------------------------------------
# Реализация деконволюции изображения
'''
    1) Считываем dat файл изображения
    2) Считываем dat файл с ядром функции
    3) Выводим изображение
    4) Дополняем нулями функицю ядра
    5) Построчно производим Фурье преобразование изображения
    6) Производим преобразование Фурье функции ядра
    7) Считаем действительные и мнимые части функции ядра и каждой строки изображения
    8) Производим построчно комплексное деление изображения и функции ядра
    9) Считаем модули
    9) Берем обратное преобразование Фурье и получаем отфильтрованное изображение от размытия
'''


# -----------------------------------------------------------------------------------------------------

def task_image_deconvolution():
    file_name_1 = 'data_deconvolution/blur307x221D.dat'  # Изображение без шумов
    file_name_2 = 'data_deconvolution/blur307x221D_N.dat'  # Изображение с шумами
    file_name_3 = 'data_deconvolution/kernD76_f4.dat'  # Массив значений ядра смазывающей функции

    data = open_reader(file_name_2, 'f')  # считываем данные из файла изображения и сохраняем в массив
    function_core = open_reader(file_name_3,
                                'f')  # считываем данные из файла ядра смазывающей функции и сохраняем в массив

    length_core = len(function_core)  # Длина ядра смазывающей функции
    print(len(data))

    width, height = 307, 221
    for i in range(width - length_core):
        function_core.append(0)
    print(len(function_core))

    matrix_pix = np.array(data).reshape(height, width)
    pix = from_2d_to_1d(matrix_pix, height, width)

    plt.imshow(matrix_pix, cmap='gist_gray', origin='lower')
    plt.show()

    # Применим деконволюцию
    # deconv_matrix_pix = image_deconvolution(matrix_pix, function_core)

    k = 15  # Для изображения с шумами
    # k = 0.0005  # Для изображения без  шумов
    deconv_matrix_pix = optimal_image_deconvolution(matrix_pix, function_core, k)

    # Вывод отфильтрованного изображения
    plt.imshow(deconv_matrix_pix, cmap='gist_gray', origin='lower')
    plt.title(f'Отфильтрованное изображение {k}')
    plt.show()


# -----------------------------------------------------------------------------------------------------
# Релизация выделения контуров объекта
'''
Сегментировать контуры объектов в изображении model.jpg без шумов и с шумами 15% двумя способами:
    1) ФНЧ
    2) ФВЧ
В обоих случаях можно применять пороговые преобразования и арифметические операции с изображениями.
Обосновать последовательность применения всех преобразований и их параметры
Оценить результаты
1) пороговое преобразование, получаем бинарное изображение (можно выбрать два порога или один)
2) Бинарное изображение ФНЧ с правильным параметром (он немного размажет края)
3) Вычесть это изображение и изображение с пороговым преобразованием (остануться только размытые контуры
они будут довольно толстые, но зато будут без разрывов
4*) пороговое преобразование (останется объект с толстым контуром и шум)
*5) Медианный фильтр
для ФВЧ
Высокие частоты - мелкие объекты
контур - резкий переход (пиксели фона, скачок в пару пикселей и затем объект)
фильтрация - после порогового преобразования
'''


# -----------------------------------------------------------------------------------------------------
def task_contour_segmentation():
    pref = 'image_model_noise/'
    # files = ['MODEL.jpg', 'gauss_5.jpg', 'impulse_5.jpg', 'gauss_impulse_5.jpg']
    # files = ['MODEL.jpg']
    files = ['impulse_5.jpg']

    for i in range(len(files)):
        files[i] = f'{pref}{files[i]}'

    for file in files:
        # Считываем матрицу пикселей изображения
        image = read_jpg_gray(file)
        image.show()
        width, height = image.size[0], image.size[1]
        matrix_pixels = np.array(image).reshape(height, width)

        # Произодим пороговую фильтрацию
        count = 0
        for row in range(height):
            for col in range(width):
                if matrix_pixels[row][col] < 200:
                    matrix_pixels[row][col] = 0
                else:
                    matrix_pixels[row][col] = 255
                    count += 1

        # Рисуем изображение после порогового преобразования
        image_new = drawing_image_new(matrix_pixels, width, height)
        image_new.show()

        #  Гистограмма порогового изображения
        hist_image_new_x, hist_image_new_y, pix_new = image_histogram(image_new)
        print(hist_image_new_y[255])

        plt.bar(hist_image_new_x, hist_image_new_y)
        plt.title(f'Гистограмма  {file}')
        plt.show()

        # --------------------------------------------------------------- потом удалить
        spectr = ampl_spectr(matrix_pixels[100], width)
        plt.plot(spectr[:width // 2])
        plt.show()

        acf_f = acf(matrix_pixels[100], width)
        spectr_acf = ampl_spectr(acf_f, width)

        plt.plot(spectr_acf[:width // 2])
        plt.title('ACF')
        plt.show()

        # --------------------------------------------------------------- до этого места

        # Фильтр низких частот для порогового изображения, чтобы размыть изображение
        m, dx = 64, 1 / width
        if file == f'{pref}MODEL.jpg':
            fc = 15  # Чем меньше fc, тем больше размытие контуров, в итоге шире контуры для без шумов
            porog = 50
        elif file == f'{pref}impulse_5.jpg':
            fc = 10  # Для импульсного
            porog = 50
        elif file == f'{pref}gauss_5.jpg':
            fc = 10  # Для импульсного
            porog = 50
        elif file == f'{pref}gauss_impulse_5.jpg':
            fc = 20  # Для импульсного
            porog = 30

        control_mass = low_pass_filter(m, dx, fc)  # Веса фильтра низких частот
        conv_mass = image_conv(matrix_pixels, control_mass, width, height, m)  # размытое изображение

        image_convol = drawing_image_new(conv_mass, width, height)
        image_convol.show()

        # Теперь попробуем вычесть из размытого бинарное изображение, чтобы получить контуры
        matrix_pixels_1 = conv_mass - matrix_pixels
        filt_image = drawing_image_new(matrix_pixels_1, width, height)
        filt_image.show()

        # Гистограмма Разницы изображений
        hist_image_new_x, hist_image_new_y, pix_new = image_histogram(filt_image)
        plt.plot(hist_image_new_x, hist_image_new_y)
        plt.title(f'Гистограмма  {file}')
        plt.show()

        # Произодим пороговую фильтрацию для четких контуров
        for row in range(height):
            for col in range(width):
                if matrix_pixels_1[row][col] > porog:
                    matrix_pixels_1[row][col] = 255
                else:
                    matrix_pixels_1[row][col] = 0

        matrix_pixels_1 = np.array(matrix_pixels_1).reshape(height, width)
        contour_image = drawing_image_new(matrix_pixels_1, width, height)
        contour_image.show()

        # Фильтруем изображения медианным фильтром
        if file == f'{pref}impulse_5.jpg' or file == f'{pref}gauss_impulse_5.jpg':
            matrix_pixels_1 = spatial_filter_median(contour_image, 3)
            matrix_pixels_1 = np.array(matrix_pixels_1).reshape(height, width)
            contour_image = drawing_image_new(matrix_pixels_1, width, height)
            contour_image.show()


# -----------------------------------------------------------------------------------------------------
'''
1) Считываем изображение
2) Гистограмма изображения
3) Пороговое преобразование
4) Считаем градиент бинарного изображения
5) выводим модуль градиента бинарного изображения
'''


# Сегментация изображения с помощью градианта
def task_segmentation_gradient():
    # files = ['MODEL.jpg', 'gauss_15.jpg', 'impulse_15.jpg', 'gauss_impulse_15.jpg']
    file = 'image_model_noise/gauss_impulse_5.jpg'

    # Считываем изображение
    image = read_jpg_gray(file)
    width, height = image.size[0], image.size[1]
    image.show()

    # Используем медианный фильтр
    image = spatial_filter_median(image, 5)
    image.show()
    # image.save(f'image_contours/gauss_med_filt.jpg')
    # image.save(f'image_contours/gauss_impulse_med_filt.jpg')

    matrix_pixels = np.array(image).reshape(height, width)

    # Гистограмма входного изображения
    hist_image_x, hist_image_y, image_pix = image_histogram(image)

    plt.bar(hist_image_x, hist_image_y)
    plt.title(f'Гистограмма Изображения с шумом')
    # plt.savefig(f'image_contours/model_hist.jpg')
    # plt.savefig(f'image_contours/gauss_hist.jpg')
    # plt.savefig(f'image_contours/gauss_impulse_hist.jpg')
    plt.show()

    # Произодим пороговую фильтрацию
    for row in range(height):
        for col in range(width):
            if 190 < matrix_pixels[row][col] < 250:
                matrix_pixels[row][col] = 255
            else:
                matrix_pixels[row][col] = 0

    image_bin = drawing_image_new(matrix_pixels, width, height)
    image_bin.show()
    # image_bin.save(f'image_contours/gauss_bin.jpg')
    # image_bin.save(f'image_contours/gauss_impulse_bin.jpg')

    # Считаем модуль градиента изображения
    gradient_matrix_x = gradient(matrix_pixels, 'row')
    image_new = drawing_image_new(gradient_matrix_x, width - 1, height)
    image_new.show()
    # image_new.save(f'image_contours/gauss_bin_grad_x.jpg')
    # image_new.save(f'image_contours/gauss_impulse_bin_grad_x.jpg')

    gradient_matrix_y = gradient(matrix_pixels, 'column')
    image_new = drawing_image_new(gradient_matrix_y, width, height - 1)
    image_new.show()
    # image_new.save(f'image_contours/gauss_bin_grad_y.jpg')
    # image_new.save(f'image_contours/gauss_impulse_bin_grad_y.jpg')

    gradient_matrix = []
    for row in range(height - 1):
        new_row = []
        for col in range(width - 1):
            new_row.append(sqrt(gradient_matrix_x[row][col] ** 2 + gradient_matrix_y[row][col] ** 2))
        gradient_matrix.append(new_row)

    gradient_matrix = np.array(gradient_matrix).reshape(height - 1, width - 1)

    image_mod_grad = drawing_image_new(gradient_matrix, width - 1, height - 1)
    image_mod_grad.show()
    # image_mod_grad.save(f'image_contours/gauss_bin_grad_module.jpg')
    # image_mod_grad.save(f'image_contours/gauss_impulse_bin_grad_module.jpg')

    hist_x, hist_y, pix = image_histogram(image_mod_grad)

    plt.plot(hist_x, hist_y)
    plt.show()


def task_segmentation_laplasian():
    # files = ['MODEL.jpg', 'gauss_15.jpg', 'impulse_15.jpg', 'gauss_impulse_15.jpg']
    file = 'image_model_noise/impulse_5.jpg'

    # Считываем изображение
    image = read_jpg_gray(file)
    width, height = image.size[0], image.size[1]
    image.show()
    # image.save('image_contours/2input_image.jpg')

    hist_image_x, hist_image_y, image_pix = image_histogram(image)

    plt.bar(hist_image_x, hist_image_y)
    plt.title(f'Гистограмма Входного')
    plt.show()

    matrix_pixels = np.array(image).reshape(height, width)
    # Произодим пороговую фильтрацию
    for row in range(height):
        for col in range(width):
            if 110 < matrix_pixels[row][col] < 210:
                matrix_pixels[row][col] = 255
            else:
                matrix_pixels[row][col] = 0

    drawing_image_new(matrix_pixels, width, height).show()

    # Используем медианный фильтр
    image = spatial_filter_median(image, 5)
    image.show()
    # image.save('image_contours/2filt_image.jpg')

    matrix_pixels = np.array(image).reshape(height, width)

    # Гистограмма входного изображения
    hist_image_x, hist_image_y, image_pix = image_histogram(image)

    plt.bar(hist_image_x, hist_image_y)
    plt.title(f'Гистограмма Изображения шумом')
    plt.show()
    # plt.savefig('image_contours/2hist.jpg')

    # # Произодим пороговую фильтрацию
    # for row in range(height):
    #     for col in range(width):
    #         if 100 < matrix_pixels[row][col] < 200:
    #             matrix_pixels[row][col] = 255
    #         else:
    #             matrix_pixels[row][col] = 0

    image_bin = drawing_image_new(matrix_pixels, width, height)
    image_bin.show()
    # image_bin.save('image_contours/2image_bin.jpg')

    # Считаем лапласиан
    matrix_bin = np.array(image_bin).reshape(height, width)
    laplas_matrix = laplasian(matrix_bin)

    image_laplas = drawing_image_new(laplas_matrix, width - 2, height - 2)
    image_laplas.show()
    # image_laplas.save('image_contours/2laplas.jpg')
    # image_laplas.save('image_contours/lapl_gauss_15_cont.jpg')

    matrix_pixels = np.array(image_laplas).reshape(height - 2, width - 2)

    hist_x, hist_y, pix = image_histogram(image_laplas)

    plt.plot(hist_x, hist_y)
    plt.show()


# ----------------------------------------------------------------------------------------------------
# Реализация выделение контура с помощью эрозии
'''
1) считываем изображение (модельное и с шумом)
2) производим бинаризацию изображения
3) применяем эрозию к изображению
4) вычитаем из бинарного изображения резильтат эрозии
'''


def task_erosion():
    # files = ['MODEL.jpg', 'impulse_15.jpg']
    # file = 'image_model_noise/impulse_5.jpg'
    # file = 'image_model_noise/MODEL.jpg'
    file = 'image_model_noise/impulse_5.jpg'
    # file = 'image_model_noise/gauss_impulse_1.jpg'

    # Считываем изображение
    image = read_jpg_gray(file)
    width, height = image.size[0], image.size[1]

    matrix_pixels = np.array(image).reshape(height, width)

    # Выводим входное изображение
    image.show()

    # Произодим пороговую фильтрацию
    for row in range(height):
        for col in range(width):
            if matrix_pixels[row][col] < 200:
                matrix_pixels[row][col] = 0
            else:
                matrix_pixels[row][col] = 255

    # Выводим бинарное изображение
    image_bin = drawing_image_new(matrix_pixels, width, height)
    image_bin.save(f'image_er_dil/bin_{file[18:]}')
    image_bin.show()

    # Если в изображении есть шумы, то применяем медианный фильтр
    if file != 'image_model_noise/MODEL.jpg':
        image_bin = spatial_filter_median(image_bin, 5)
        image_bin.save(f'image_er_dil/filtered_{file[18:]}')
        image_bin.show()
        filt_matrix = np.array(image_bin).reshape(height, width)

    # Производим эрозию или дилатацию изображения
    # new_matrix = erosion(image_bin, 9, 9)
    new_matrix = dilatation(image_bin, 9, 9)
    er_image = drawing_image_new(new_matrix, len(new_matrix[0]), len(new_matrix))
    er_image.save(f'image_er_dil/er_{file[18:]}')
    er_image.show()

    if file == 'image_model_noise/MODEL.jpg':
        pix = new_matrix - matrix_pixels  # Для дилатации
        # pix = matrix_pixels - new_matrix  # Для эрозии
    else:
        pix = new_matrix - filt_matrix  # Для дилатации с шумом
        # pix = filt_matrix - new_matrix  # Для эрозии

    # Выводим контур объекта
    img = drawing_image_new(pix, width, height)
    img.save(f'image_er_dil/cont_{file[18:]}')
    img.show()


# ----------------------------------------------------------------------------------------------------
'''
Используя методы:
- изменения размеров;
- сегментации;
- пространственной и частотной обработки;
- градационных преобразований.

Разработать и реализовать максимально автоматизированный или автоматический алгоритм 
настройки оптимальной яркости и конрастности четырех изображений 
вертикальных и горизонтальных МРТ срезов:
2 для позвоночника и 2 для головы, приведя изображения к размерам 400х400.

Формат данных двоичный, целочисленный 2-хбайтовый (short).
'''


def task_mrt():
    # files = ['mrt/spine-H_x256.bin', 'mrt/spine-V_x512.bin', 'mrt/brain-H_x512.bin', 'mrt/brain-V_x256.bin']
    files = ['mrt/spine-H_x256.bin']
    new_width = new_height = 400

    for file in files:
        # Вычленяем размеры изображения из названия файла
        width = height = int(re.search(r'\d+', file).group(0))

        # Считываем целые числа по 2 байта
        bin_file = open_reader(file, format='h')

        # Строим изображение
        matrix_pixels = np.array(bin_file).reshape(height, width)
        image = drawing_image_new(matrix_pixels, width, height)
        image.save(f'mrt/input_{file[4:16]}.jpg')
        image.show()

        # Строим гистограмму изображения
        hist_x, hist_y, pixels = image_histogram(image)

        plt.plot(hist_x, hist_y)
        plt.title(f'Гистограмма {file[4:16]}')

        plt.show()

        # Применяем эвкализацию изображния
        contr_image = equalization(image)
        contr_image.save(f'mrt/equal_{file[4:16]}.jpg')
        contr_image.show()

        # Масштабируем изображение до размеров 400х400
        if width > new_width:
            const = width / new_width
            type_scale = 'nn'
            mode = 'decreased'
        else:
            const = new_width / width
            type_scale = 'nn'
            mode = 'increased'

        scale_image = image_scale(contr_image, const, type_scale, mode)
        scale_image.save(f'mrt/scale_{file[4:16]}.jpg')
        scale_image.show()
        scale_matrix = np.array(scale_image).reshape(new_height, new_width)

        # Строим гистограмму изображения после эквализации
        hist_x, hist_y, pixels = image_histogram(scale_image)

        plt.plot(hist_x, hist_y)
        plt.title(f'Гистограмма после эквализации {file[4:16]}')

        plt.show()

        # for i in range(1, 10):

        # gamma_matrix = gamma_correction(scale_matrix, const=10, gamma=20)
        # # gamma_matrix = log_correction(scale_matrix, const=20)
        # #
        # gamma_image = drawing_image_new(gamma_matrix, new_width, new_height)
        # gamma_image.show()

        # # Реализуем Спектр
        # spectrum_image = ampl_spectr(scale_matrix[100], new_width)
        # acf_image = acf(spectrum_image, new_width)
        # spectrum_acf = ampl_spectr(acf_image, new_width)
        # x_spectrum = np.linspace(0, 1, new_width)
        #
        #
        # # plt.subplot(1, 2, 1)
        # # plt.plot(x_spectrum[:new_width // 2], spectrum_image[:new_width // 2])
        # # plt.title(f'Спектр изображения {file[4:16]}')
        # #
        # # plt.subplot(1, 2, 2)
        # # plt.plot(x_spectrum[:new_width // 2], spectrum_acf[:new_width // 2])
        # #
        # # plt.title(f'Спектр АКФ')
        # #
        # # plt.show()
        # #
        # # # m, dt, fc = 64, 1 / 400, 1
        # # # hpw = high_pass_filter(m, dt, fc)
        # # # conv_matrix = image_conv(scale_matrix, hpw, new_width, new_height, m)
        # # #
        # # #
        # # #
        # # # conv_image = drawing_image_new(conv_matrix, new_width, new_height)
        # # # conv_image.show()
        # # #
        # # # filt_image = spatial_filter_median(conv_image, 5)
        # # # filt_image.show()


'''
Применяя все реализованные методы обработки и анализа изображений,
а также любые сторонние методыбиблиотеки помимо реализованных
выделить и автоматически подсчитать на изображении stones.jpg
камни заданного размера S в двух вариантах:
1) Выделить только те объекты, у которых размер по каждому из направлений равен S
2) Выделить камни, у которых размер хотя бы по одному направлению равен 
 остальных направлениях меньше S
 S = 11
'''


def count_of_stones():
    file = 'size_of_object/stones.jpg'
    image = read_jpg_gray(file)
    width, height = image.size[0], image.size[1]
    matrix_pixels = np.array(image).reshape(height, width)

    # Показать входное изображение
    input_image = drawing_image_new(matrix_pixels, width, height)
    input_image.show()

    # Гистограмма
    hist_x, hist_y, pixels = image_histogram(input_image)

    plt.plot(hist_x, hist_y)
    plt.title('Гистограмма входного изображения')
    plt.show()

    # Произодим пороговую фильтрацию
    for row in range(height):
        for col in range(width):
            if matrix_pixels[row][col] < 120:
                matrix_pixels[row][col] = 0
            else:
                matrix_pixels[row][col] = 255

    bin_image = drawing_image_new(matrix_pixels, width, height)
    bin_image.show()

    # Гистограмма
    hist_x, hist_y, pixels = image_histogram(bin_image)

    plt.plot(hist_x, hist_y)
    plt.title('Гистограмма бинарного изображения')
    plt.show()

    # Производим эрозию или дилатацию изображения
    new_matrix = erosion(bin_image, 3, 3)
    # new_matrix = dilatation(bin_image, 3, 3)
    er_image = drawing_image_new(new_matrix, len(new_matrix[0]), len(new_matrix))

    er_image.show()

    if file == 'image_model_noise/MODEL.jpg':
        pix = new_matrix - matrix_pixels  # Для дилатации
        # pix = matrix_pixels - new_matrix  # Для эрозии
    else:
        pix = new_matrix - matrix_pixels  # Для дилатации с шумом
        # pix = filt_matrix - new_matrix  # Для эрозии

    # Выводим контур объекта
    img = drawing_image_new(pix, width, height)

    img.show()





    # mask = [
    #     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    #     [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    #     [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    #     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    # ]
    #
    # for row in range(11, height - 11):
    #     for col in range(11, width - 11):




if __name__ == '__main__':
    # trends()
    # task_rand_stand()
    # task_rand_my()
    # task_shift()
    # task_spikes()
    # task_additive_noise()
    # task_multiplicative_noise()
    # task_garm_process()
    # task_ampl_spectr()
    # task_poligarm_processes()
    # task_historam()
    # task_acf()
    # task_antishist()
    # task_antispikes()
    # task_antitrend()
    # task_open_reader()
    # task_inv_fourier_transf()
    # task_zero_last_50()
    # task_ecg()
    # task_acg_furie()
    # task_low_pass_filter()
    # task_open_file_convolution()
    # task_filters()
    # task_open_file_filter()
    # task_file_convolution_hpf()
    # task_file_convolution_bpf()
    # task_file_convolution_bsf()
    # task_wav_file()
    # task_my_wav()
    # task_exam()
    # polig_pr()
    # task_deconvolution_of_ecg()
    # task_read_image()
    # task_scaling_image_1()
    # task_scaling_image_2()
    # task_negative_image()
    # task_gamma_corrextion()
    # task_log_corrextion()
    # task_equalization()
    # task_read_xcr()
    # task_add_model_noise()
    # task_spatial_filter_average()
    # task_spatial_filter_median()
    # task_image_deconvolution()
    # task_contour_segmentation()
    # task_segmentation_gradient()
    # task_segmentation_laplasian()
    # task_erosion()
    # task_mrt()
    count_of_stones()

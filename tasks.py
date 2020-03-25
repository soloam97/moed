from matplotlib import pyplot as plt
from func import *
import soundfile as sf
import sounddevice as sd
import wavio
from PIL import *


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
    # y = open_reader("C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat")
    y = open_reader("C:/Users/soloa/OneDrive/Документы/MATLAB/matrica.dat")

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
    input_mass = open_reader("C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat")

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
    input_mass_y = open_reader("C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat")  # Частоты 5, 55, 250
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
    input_mass = open_reader("C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat")  # Частоты 5, 55, 250

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
    input_mass = open_reader("C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat")  # Частоты 5, 55, 250

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
    input_mass = open_reader("pgp_f4-1K-1ms.dat")  # Частоты 5, 55, 250

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

    input_mass = open_reader("v1x2.dat")  # Открытие файла
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


def task_negative_image():
    # mode = int(input('mode:'))  # Считываем номер преобразования.
    image = read_jpg_gray('image1.jpg')  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.

    # pixels = []
    # for i in range(width):
    #     for j in range(height):
    #         pixels.append(pix[i, j])
    # L_max = max(pixels)
    #
    # new_width, new_height = width, height
    # new_image = Image.new('L', (new_width, new_height))
    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            draw.point((x, y), (255 - r))
    image.save("result1.1.jpg", "JPEG")  # сохранить изображение
    image.show()

    image = read_jpg_gray('image2.jpg')  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.

    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            draw.point((x, y), (255 - r))
    image.save("result2.1.jpg", "JPEG")  # сохранить изображение
    image.show()

    image = read_jpg_gray('image3.jpg')  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.

    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            draw.point((x, y), (255 - r))
    image.save("result3.1.jpg", "JPEG")  # сохранить изображение
    image.show()


def task_gamma_corrextion():
    # mode = int(input('mode:'))  # Считываем номер преобразования.
    image = read_jpg_gray('image1.jpg')  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.
    const = 7
    gamma = 0.7
    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            draw.point((x, y), (math.ceil(const * (r ** gamma))))
    image.save("result1.2.jpg", "JPEG")  # сохранить изображение
    image.show()

    image = read_jpg_gray('image2.jpg')  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.
    const = 7
    gamma = 0.7
    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            draw.point((x, y), (math.ceil(const * (r ** gamma))))
    image.save("result2.2.jpg", "JPEG")  # сохранить изображение
    image.show()

    image = read_jpg_gray('image3.jpg')  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.
    const = 7
    gamma = 0.7
    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            draw.point((x, y), (math.ceil(const * (r ** gamma))))
    image.save("result3.2.jpg", "JPEG")  # сохранить изображение
    image.show()


def task_log_corrextion():
    # mode = int(input('mode:'))  # Считываем номер преобразования.
    image = read_jpg_gray('image1.jpg')  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.
    const = 20
    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            draw.point((x, y), (math.ceil(const * log2(r + 1))))
    image.save("result1.3.jpg", "JPEG")  # сохранить изображение
    image.show()

    image = read_jpg_gray('image2.jpg')  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.
    const = 20
    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            draw.point((x, y), (math.ceil(const * log2(r + 1))))
    image.save("result2.3.jpg", "JPEG")  # сохранить изображение
    image.show()

    image = read_jpg_gray('image3.jpg')  # Открываем изображение.
    draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    pix = image.load()  # Выгружаем значения пикселей.
    const = 20
    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            draw.point((x, y), (math.ceil(const * log2(r + 1))))
    image.save("result3.3.jpg", "JPEG")  # сохранить изображение
    image.show()


def task_equalization():
    # Считываем  файл
    image = read_jpg_gray('HollywoodLC.jpg')
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
    new_image.save("HollywoodLC2.jpg", "JPEG")  # сохранить изображен
    new_image.show()


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
    w, h = 300, 400
    matrix_pix = np.array(image).reshape(w, h)
    pix = from_2d_to_1d(matrix_pix, w, h)

    # Создаем входное изображение 400x300 (поворот на 90 градусов)
    input_image = drawing_image_new(pix, w, h).transpose(Image.ROTATE_90)
    input_image.show()

    derivative_mass = derivative(matrix_pix, w, h)
    pix = from_2d_to_1d(derivative_mass, w, h - 1)

    # Создаем изображение производной 400x300 (поворот на 90 градусов)
    input_image = drawing_image_new(pix, w, h - 1).transpose(Image.ROTATE_90)
    input_image.show()


    # Создаем спектр для производной
    spectrum = ampl_spectr(derivative_mass[0], h - 1)
    spectrum_x = range(h - 1)
    plt.plot(spectrum_x, spectrum)
    plt.title('Спектр')
    plt.xlabel('Пискель')
    plt.ylabel('Яркость')
    plt.show()

    fs = 400
    dt = 1 / fs
    m, fc1, fc2 = 64, 100, 130
    input_mass = matrix_pix
    control_mass = bend_stop_filter(m, dt, fc1, fc2)

    conv_matrix_pix = image_conv(input_mass, control_mass, w, h, m)
    conv_pix = from_2d_to_1d(conv_matrix_pix, w, h)

    # Создаем изображение 400x300 (поворот на 90 градусов)
    conv_image = drawing_image_new(conv_pix, w, h).transpose(Image.ROTATE_90)
    conv_image.show()

    # Создаем АКФ для производной
    w_acf = h - 1
    acf_of_derivative = acf(derivative_mass[0], w_acf)
    # Считаем спектр АКФ для производной
    mass_x = range(w_acf)
    spectr_acf_of_derivative = ampl_spectr(acf_of_derivative, w_acf)

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(mass_x, acf_of_derivative)  # График АКФ производной
    plt.title('График АКФ производной')

    plt.subplot(1, 2, 2)
    plt.plot(mass_x[:(w_acf // 2)], spectr_acf_of_derivative[:(w_acf // 2)])  # График спектра АКФ производной
    plt.title('Спектр АКФ производной')
    plt.show()

    # Создаем ВКФ для производной
    w_mcf = w_acf
    mass1_for_mcf = derivative_mass[0]
    mass2_for_mcf = derivative_mass[1]
    mcf_of_derivative = mcf(mass1_for_mcf, mass2_for_mcf, w_mcf)
    # Считаем сектр ВКФ для производной
    spectr_mcf_of_derivative = ampl_spectr(mcf_of_derivative, w_mcf)

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(mass_x, mcf_of_derivative)
    plt.title('График ВКФ производной')

    plt.subplot(1, 2, 2)
    plt.plot(mass_x[:(w_mcf // 2)], spectr_mcf_of_derivative[:(w_mcf // 2)])
    plt.title('Спектр ВКФ производной')
    plt.show()

    image_equal = equalization(conv_image)
    image_equal.show()


def task_adding_noise():
    # Загружаем картинку
    file = 'MODEL.jpg'
    image = read_jpg_gray(file)
    w, h = image.size[0], image.size[1]
    image.show()  # Выводим на экран входную картинку

    # гистограмма для входного изображения
    hist_image_x, hist_image_y, hist_image_pix = image_histogram(image)

    # Добавление гауссовского шума
    level = 1000
    image_gauss_noise = add_gauss_noise(file, level)
    image_gauss_noise.show()
    image_gauss_noise.save('gauss_noise.jpg')
    image_gauss_noise = read_jpg_gray('gauss_noise.jpg')

    # гистограмма для зашумленного изображения
    hist_gauss_x, hist_gauss_y, hist_gauss_pix = image_histogram(image_gauss_noise)

    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(hist_image_x, hist_image_y)

    plt.subplot(1, 2, 2)
    plt.plot(hist_gauss_x, hist_gauss_y)

    plt.show()

    # Добавление импульсного шума
    image_impulse_noise = add_impulse_noise(file)
    image_impulse_noise.show()

    # зашумление изображения гауссовским шумом + "соль и перец"
    image_gauss_impulse_noise = add_impulse_noise(image_gauss_noise)
    image_gauss_impulse_noise.show()

    pixels = image_impulse_noise.load()
    pix = []
    for i in range(w):
        for j in range(h):
            pix.append(pixels[i, j])
    matrix_pix = np.array(pix).reshape(w, h)

    # Создаем спектр для производной
    spectrum = ampl_spectr(matrix_pix[100], h)
    spectrum_x = range(h)
    plt.plot(spectrum_x, spectrum)
    plt.title('Спектр')
    plt.xlabel('Пискель')
    plt.ylabel('Яркость')
    plt.show()
    #
    # acf_mass = acf(spectrum, h)
    # plt.plot(spectrum_x, acf_mass)
    # plt.show()

    m = 32
    dt = 0.004
    input_mass = matrix_pix
    control_mass = low_pass_filter(m, dt, fc=100)
    conv_matrix = image_conv(input_mass, control_mass, w, h, m)
    conv_pixels = from_2d_to_1d(conv_matrix, w, h)

    conv_image = drawing_image_new(conv_pixels, w, h)
    conv_image.show()

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
    task_read_xcr()
    # task_adding_noise()

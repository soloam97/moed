from func import *
import numpy as np

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


def main():

    i = 3
    f = [5, 55, 110]
    a = [5, 10, 16]
    dt = 0.001
    N = 1000
    input_mass = poligarm_procceses(i, f, a, dt, N)
    x = [i for i in range(N)]

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

if __name__ == '__main__':
    main()
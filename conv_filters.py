def task_open_file_filter():
    # считывание файла для его обработки, N - длина массива из файла
    input_mass_y = open_reader("C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat") # Частоты 5, 55, 250
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
    input_mass = open_reader("C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat") # Частоты 5, 55, 250

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
    input_mass = open_reader("C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat") # Частоты 5, 55, 250

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
    input_mass = open_reader("C:/Users/soloa/PycharmProjects/univer1/2/pgp_f4-1K-1ms.dat") # Частоты 5, 55, 250

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
import numpy as np

x_size = 100


def step(c, d):

    c_t_1 = np.zeros(x_size, dtype=float)

    for x in range(1, x_size-1, 1):

        # perm = 1.0
        # if d[x+1] == 0.0 or  d[x] == 0.0:
        #     perm = 0.0

        #OK
        # flux = d[x+1]*(c[x+1]-c[x]) - d[x]*(c[x]-c[x-1])
        if d[x + 1] == 0.0 or d[x] == 0.0:
            p_1 = 0.0
        else:
            p_1 = (d[x + 1] + d[x])/2

        if d[x] == 0.0 or d[x-1] == 0.0:
            p_0 = 0.0
        else:
            p_0 = (d[x] + d[x-1])/2


        # flux = (d[x + 1] + d[x]) * (c[x + 1] - c[x])/2.0 - (d[x]+d[x-1]) * (c[x] - c[x - 1])/2.0
        flux_x = p_1 * (c[x + 1] - c[x]) - p_0 * (c[x] - c[x - 1])

        c_t_1[x] = c[x] + flux_x

    return c_t_1


def balance_check(c):
    return np.sum(c)


if __name__ == '__main__':

    c = np.zeros(x_size, dtype=float)

    c[50] = 2000.0

    d = np.zeros(x_size, dtype=float)
    d[:53] = 0.3

    num_step = 40
    for s in range(num_step):

        initial_check = balance_check(c)

        c_t_1 = step(c, d)

        step_check = balance_check(c_t_1)

        c = c_t_1

        print('initial_check=',initial_check)
        print('step_check=', step_check)

    print(c)
import numpy as np
import numpy.linalg
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
from scipy.sparse import linalg
from time import process_time


def main():
    # iteration count

    it_num = int(200)

    # geometry

    # nx = 440
    # ny = 440
    h = float(1)

    # Neutron group yield fractions

    del_g_1 = float(1)
    del_g_2 = float(0)

    # Fuel neutronic data

    f_s_12 = float(0.018193)  # group 1
    f_d_1 = float(1.0933622)
    f_a_1 = float(0.0092144)
    f_f_1 = float(0.0065697)
    f_r_1 = f_a_1 + f_s_12

    f_s_21 = float(0.0013089)  # group 2
    f_d_2 = float(0.3266693)
    f_a_2 = float(0.0778104)
    f_f_2 = float(0.13126)
    f_r_2 = f_a_2 + f_s_21

    # Reflector neutronic data

    r_s_12 = float(0.0255380)  # group 1
    r_d_1 = float(1.1245305)
    r_a_1 = float(0.0008996)
    r_f_1 = float(0)
    r_r_1 = r_a_1 + r_s_12

    r_s_21 = float(0.0001245)  # group 2
    r_d_2 = float(0.7503114)
    r_a_2 = float(0.0255590)
    r_f_2 = float(0)
    r_r_2 = r_a_2 + r_s_21

    # Control rod neutronic data

    # In control rods, all fluxes are assumed to be 0

    # Reference geometry matrix

    ref_2D = np.ones((440, 440), dtype=int)

    ref_2D[1:20, :] = 2  # reflector
    ref_2D[420:439, :] = 2
    ref_2D[:, 1:20] = 2
    ref_2D[:, 420:439] = 2

    ref_2D[210:230, 210:230] = 0  # control rod

    ref_2D[0, :] = 3  # edges (data will be from reflector)
    ref_2D[:, 439] = 4
    ref_2D[439, :] = 5
    ref_2D[:, 0] = 6

    ref_2D[0, 0] = 7  # corners (data will be from reflector)
    ref_2D[0, 439] = 8
    ref_2D[439, 439] = 9
    ref_2D[439, 0] = 10

    ref_1D = np.reshape(ref_2D, (193600, 1), order='C')

    # Coefficient Matrices using the reference geometry matrices

    A_1 = make_big_ugly_matrix(ref_1D, h, f_d_1, r_d_1, f_r_1, r_r_1)
    A_2 = make_big_ugly_matrix(ref_1D, h, f_d_2, r_d_2, f_r_2, r_r_2)

    F_1 = make_diagonal_matrix(ref_1D, float(0), f_f_1, r_f_1)
    F_2 = make_diagonal_matrix(ref_1D, float(0), f_f_2, r_f_2)
    s_12 = make_diagonal_matrix(ref_1D, float(0), f_s_12, r_s_12)
    s_21 = make_diagonal_matrix(ref_1D, float(0), f_s_21, r_s_21)

    # Initial guesses

    phi_1_old = np.ones((440 * 440, 1), dtype=float)
    phi_2_old = np.ones((440 * 440, 1), dtype=float)
    k_old = float(1)

    # Solve

    start_time = process_time()
    for i in range(0, it_num):
        S_1 = (del_g_1 / k_old) * (F_1 * phi_1_old + F_2 * phi_2_old) + s_21 * phi_2_old
        S_2 = (del_g_2 / k_old) * (F_1 * phi_1_old + F_2 * phi_2_old) + s_12 * phi_1_old

        phi_1_new, x_1 = sp.linalg.gmres(A_1, S_1)
        phi_2_new, x_2 = sp.linalg.gmres(A_2, S_2)
        k_new = k_old * np.sum(h * (F_1 * phi_1_new + F_2 * phi_2_new)) / np.sum(h * (F_1 * phi_1_old + F_2 * phi_2_old))

        phi_1_old = phi_1_new
        phi_2_old = phi_2_new
        k_old = k_new

    # reshape outputs

    p1 = np.reshape(phi_1_old, (440, 440), order='C')
    p2 = np.reshape(phi_2_old, (440, 440), order='C')

    end_time = process_time()
    print(end_time - start_time, "seconds")

    # Output

    print(k_old)

    x = np.arange(0, 440, 1)
    y = np.arange(0, 440, 1)
    x_, y_ = np.meshgrid(x, y)

    fig = plt.figure(figsize=(13.5, 7))
    a1 = fig.add_subplot(121, projection='3d')
    a2 = fig.add_subplot(122, projection='3d')

    a1.plot_surface(x_, y_, p1, label='Group 1', cmap='PuRd_r')
    a2.plot_surface(x_, y_, p2, label='Group 1', cmap='PuBuGn_r')

    plt.title(f'k_effective = {k_old}')
    plt.show()


def make_diagonal_matrix(ref, data_0, data_1, data_2):
    a = np.zeros((len(ref),), dtype=float)
    for i in range(0, len(ref)):
        if ref[i, 0] == 0:
            a[i] = data_0
        elif ref[i, 0] == 1:
            a[i] = data_1
        else:
            a[i] = data_2

    x = sp.diags(a, offsets=0, format="csr", dtype=float)

    return x


def make_big_ugly_matrix(ref, h, f_d, r_d, f_r, r_r):
    x = sp.lil_matrix((len(ref), len(ref)), dtype=float)

    for k in range(0, len(ref)):
        if ref[k, 0] == 0:
            x[k, k] = 1     # control rod

        elif ref[k, 0] == 1:
            x[k, k - 440] = (- f_d) / (h ** 2)      # fuel
            x[k, k - 1] = (- f_d) / (h ** 2)
            x[k, k] = ((4 * f_d) / (h ** 2)) + f_r
            x[k, k + 1] = (- f_d) / (h ** 2)
            x[k, k + 440] = (- f_d) / (h ** 2)

        elif ref[k, 0] == 2:
            x[k, k - 440] = (- r_d) / (h ** 2)      # reflector
            x[k, k - 1] = (- r_d) / (h ** 2)
            x[k, k] = ((4 * r_d) / (h ** 2)) + r_r
            x[k, k + 1] = (- r_d) / (h ** 2)
            x[k, k + 440] = (- r_d) / (h ** 2)

        elif ref[k, 0] == 3:
            x[k, k - 1] = (- r_d) / (h ** 2)        # top edge
            x[k, k] = ((4 * r_d) / (h ** 2)) + r_r
            x[k, k + 1] = (- r_d) / (h ** 2)
            x[k, k + 440] = (- r_d) / (h ** 2)

        elif ref[k, 0] == 4:
            x[k, k - 440] = (- r_d) / (h ** 2)      # right edge
            x[k, k - 1] = (- r_d) / (h ** 2)
            x[k, k] = ((4 * r_d) / (h ** 2)) + r_r
            x[k, k + 440] = (- r_d) / (h ** 2)

        elif ref[k, 0] == 5:
            x[k, k - 440] = (- r_d) / (h ** 2)      # bottom edge
            x[k, k - 1] = (- r_d) / (h ** 2)
            x[k, k] = ((4 * r_d) / (h ** 2)) + r_r
            x[k, k + 1] = (- r_d) / (h ** 2)

        elif ref[k, 0] == 6:
            x[k, k - 440] = (- r_d) / (h ** 2)      # left edge
            x[k, k] = ((4 * r_d) / (h ** 2)) + r_r
            x[k, k + 1] = (- r_d) / (h ** 2)
            x[k, k + 440] = (- r_d) / (h ** 2)

        elif ref[k, 0] == 7:
            x[k, k] = ((4 * r_d) / (h ** 2)) + r_r  # top-left corner
            x[k, k + 1] = (- r_d) / (h ** 2)
            x[k, k + 440] = (- r_d) / (h ** 2)

        elif ref[k, 0] == 8:
            x[k, k - 1] = (- r_d) / (h ** 2)        # top-right corner
            x[k, k] = ((4 * r_d) / (h ** 2)) + r_r
            x[k, k + 440] = (- r_d) / (h ** 2)

        elif ref[k, 0] == 9:
            x[k, k - 440] = (- r_d) / (h ** 2)      # bottom-right corner
            x[k, k - 1] = (- r_d) / (h ** 2)
            x[k, k] = ((4 * r_d) / (h ** 2)) + r_r

        elif ref[k, 0] == 10:
            x[k, k - 440] = (- r_d) / (h ** 2)      # bottom-left corner
            x[k, k] = ((4 * r_d) / (h ** 2)) + r_r
            x[k, k + 1] = (- r_d) / (h ** 2)

        else:
            pass

    x = x.tocsr()

    return x


main()


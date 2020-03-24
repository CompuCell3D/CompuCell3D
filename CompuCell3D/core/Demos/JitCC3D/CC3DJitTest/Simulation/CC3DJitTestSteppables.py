# For testing basic Numba integration functionality
from cc3d.core.PySteppables import *
from cc3d.cpp.CompuCell import Point3D

import math
import time
import numpy as np
from numba import prange, njit

from cc3d.core.JitCC3D import *


@njit
def get_cell_id_global(cell):
    return cell.id


@njit(debug=True)
def construct_point_3d(x, y, z):
    return Point3D(x, y, z)


@njit
def add_point_3d(pt1, pt2):
    return pt1 + pt2


@njit
def sub_point_3d(pt1, pt2):
    return pt1 - pt2


@njit
def cell_list_id_sq(cell_list):
    # return [cell for cell in cell_list]
    return [cell.id ** 2 for cell in cell_list]


@njit
def cell_pixel_list_pt2(pixel_list):
    pixel_list2 = []
    for ptd in pixel_list:
        pixel = ptd.pixel
        pixel_list2.append(add_point_3d(pixel, pixel))

    return pixel_list2


# Do the same as cell_pixel_list_pt2, but with indexing and len() of cell pixel list
@njit
def cell_pixel_list_pt2_idx(pixel_list):
    pixel_list2 = []
    for i in prange(len(pixel_list)):
        pixel = pixel_list[i].pixel
        pixel_list2.append(add_point_3d(pixel, pixel))

    return pixel_list2


# Get a cell pixel list and extract each cell's pixels inside a jitted function
@njit
def get_cell_pixels(cc3d_jit_utils):
    pixel_list = []
    for cell in cc3d_jit_utils.cell_list:
        for ptd in cc3d_jit_utils.get_cell_pixel_list(cell):
            pixel = ptd.pixel
            pixel_list.append(pixel + pixel)

    return pixel_list


def mean_cell_distance_py(steppable):
    pt_com_list = [[cell.xCOM, cell.yCOM, cell.zCOM] for cell in steppable.cell_list]
    n = len(pt_com_list)
    mean_dist_list = []
    for pt in pt_com_list:
        this_dist = 0
        for pt_i in pt_com_list:
            this_dist += math.sqrt((pt[0] - pt_i[0])**2 + (pt[1] - pt_i[1])**2 + (pt[2] - pt_i[2])**2)

        mean_dist_list.append(this_dist / (n - 1))

    return mean_dist_list


@njit(parallel=True)
def mean_cell_distance_jit(cc3d_jit_utils):
    pt_com_list = []
    mean_dist_list = []
    for cell in cc3d_jit_utils.cell_list:  # Can't currently make this parallel
        pt_com_list.append([cell.xCOM, cell.yCOM, cell.zCOM])
        mean_dist_list.append(0.0)

    n = len(pt_com_list)
    for i in prange(n):
        pt = pt_com_list[i]
        this_dist = 0
        for pt_i in pt_com_list:
            this_dist += math.sqrt((pt[0] - pt_i[0])**2 + (pt[1] - pt_i[1])**2 + (pt[2] - pt_i[2])**2)

        mean_dist_list[i] = this_dist / (n - 1)

    return mean_dist_list


class CC3DJitTestSteppable(SteppableBasePy):

    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self, frequency)

        self.lambda_volume = 2.0
        self.target_volume = 50

        self.benchmark_time_py = 0.0
        self.benchmark_time_jit = 0.0
        self.benchmark_steps = 0

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

        # Attach an instance of CC3DJitUtils to this steppable
        init_cc3d_jit(self)

        for cell in self.cell_list:
            cell.lambdaVolume = self.lambda_volume
            cell.targetVolume = self.target_volume

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """

        # Cell Accessor Test: get cell id by built-in steppable method and static and global jitted methods
        #   Global and static methods should both be able to retrieve and return a cell parameter in nopython mode
        cell1 = self.fetch_cell_by_id(1)
        id_py_cc3d = cell1.id
        id_static = self.get_cell_id_static(cell1)
        id_global = get_cell_id_global(cell1)
        assert id_py_cc3d == id_static and id_py_cc3d == id_global
        print('Cell Accessor Test: Passed! ( {}={}={} )'.format(cell1.id, id_static, id_global))

        # Cell Accessor Test 2: modify a cell parameter in a jitted method
        #   Changes to a cell parameter by a jitted method in nopython mode should appear here
        lambda_volume_tmp = 1.0
        self.set_cell_lambda_volume(cell1, lambda_volume_tmp)
        assert cell1.lambdaVolume == lambda_volume_tmp
        print('Cell Accessor Test 2: Passed! ( {}->{} : {}->{} )'.format(self.lambda_volume,
                                                                         lambda_volume_tmp,
                                                                         self.lambda_volume,
                                                                         cell1.lambdaVolume))
        cell1.lambdaVolume = self.lambda_volume  # Restore original value

        # Point Constructor Test: create a Point3D instance in a jitted method
        x, y, z = (1, 2, 3)
        pt_py = Point3D()
        pt_py.x, pt_py.y, pt_py.z = x, y, z
        pt_jit = construct_point_3d(x, y, z)
        assert pt_py == pt_jit
        # print('Point Constructor Test: Passed! ( {}={} )'.format(pt_py, pt_jit))
        print('Point Constructor Test: Passed!')

        # Point Arithmetic Test: do operations on Point3D instances in a jitted method
        pt_1 = Point3D(x+1, y+2, z+3)
        pt_2 = Point3D(x, y, z)
        pt_add_py = Point3D(pt_1.x + pt_2.x,
                            pt_1.y + pt_2.y,
                            pt_1.z + pt_2.z)
        pt_sub_py = Point3D(pt_1.x - pt_2.x,
                            pt_1.y - pt_2.y,
                            pt_1.z - pt_2.z)
        pt_add_jit = add_point_3d(pt_1, pt_2)
        pt_sub_jit = sub_point_3d(pt_1, pt_2)
        assert pt_add_py == pt_add_jit and pt_sub_py == pt_sub_jit
        # print('Point Arithmetic Test: Passed! ( {}={}; {}={} )'.format(pt_add_py, pt_add_jit, pt_sub_py, pt_sub_jit))
        print('Point Arithmetic Test: Passed!')

        # CellList Test: return a list of a calculation on all cells in a CellList
        id2_list_jit = cell_list_id_sq(self.cell_list)
        id2_list_py = [cell.id ** 2 for cell in self.cell_list]
        assert id2_list_py == id2_list_jit
        # print('CellList Test: Passed! ( {}.^2={} )'.format([cell.id for cell in self.cell_list], id2_list_jit))
        print('CellList Test: Passed!')

        # CellPixelList Test: return a list of a calculation on all pixels in a PixelList
        cell1_pixel_list = self.get_cell_pixel_list(cell1)
        pixel_list_py = [Point3D(2*ptd.pixel.x, 2*ptd.pixel.y, 2*ptd.pixel.z) for ptd in cell1_pixel_list]
        pixel_list_jit = cell_pixel_list_pt2(cell1_pixel_list)
        pixel_list_idx = cell_pixel_list_pt2_idx(cell1_pixel_list)
        assert pixel_list_jit == pixel_list_py and pixel_list_idx == pixel_list_py
        print('CellPixelList Test: Passed!')

        # CellPixelList2 Test: return a list of all pixels occupied by cells
        all_cells_pixel_list_jit = get_cell_pixels(self.cc3d_jit_utils)
        all_cells_pixel_list_py = []
        for cell in self.cell_list:
            for ptd in self.get_cell_pixel_list(cell):
                pixel = ptd.pixel
                all_cells_pixel_list_py.append(Point3D(2 * pixel.x, 2 * pixel.y, 2 * pixel.z))
        assert all_cells_pixel_list_jit == all_cells_pixel_list_py
        print('CellPixelList Test2: Passed!')

        # Benchmark: calculate mean distances between all cells with and without acceleration
        #   Measured average speedup over 100 MCS on T.J.'s primary machine: 9-12x.
        t1_i_py = time.time()
        mean_cell_distance_list_py = mean_cell_distance_py(self)
        t1_f_py = time.time()
        t1_py = t1_f_py - t1_i_py
        t1_i_jit = time.time()
        mean_cell_distance_list_jit = mean_cell_distance_jit(self.cc3d_jit_utils)
        t1_f_jit = time.time()
        t1_jit = t1_f_jit - t1_i_jit
        if mean_cell_distance_list_py != mean_cell_distance_list_jit:
            err_vals = [(mean_cell_distance_list_py[i], mean_cell_distance_list_jit[i]) for i in
                        range(len(mean_cell_distance_list_py)) if
                        mean_cell_distance_list_py[i] != mean_cell_distance_list_jit[i]]
            print('Check values: ', err_vals)

        if t1_jit > 0:
            print('Benchmark: Python: {}, Jit: {}, Speedup: {}x'.format(t1_py, t1_jit, t1_py / t1_jit))
        else:
            print('Benchmark: Python: {}, Jit: {}'.format(t1_py, t1_jit))

        if mcs > 0:  # Don't record first result, since Jit methods are compiled at first call
            self.benchmark_time_py += t1_py
            self.benchmark_time_jit += t1_jit
            self.benchmark_steps += 1

    def finish(self):
        """
        Finish Function is called after the last MCS
        """

        mean_benchmark_time_py = self.benchmark_time_py / self.benchmark_steps
        mean_benchmark_time_jit = self.benchmark_time_jit / self.benchmark_steps

        print('\n********************************************************')
        print('Mean Python time: {} s'.format(mean_benchmark_time_py))
        print('Mean Jit time   : {} s'.format(mean_benchmark_time_jit))
        print('Speedup         : {} x'.format(mean_benchmark_time_py/mean_benchmark_time_jit))
        print('********************************************************\n')


    @jit_steppable_method(nopython=True)
    def get_cell_id_static(cell):
        return cell.id

    @jit_steppable_method(nopython=True)
    def set_cell_lambda_volume(cell, lambda_volume):
        cell.lambdaVolume = lambda_volume


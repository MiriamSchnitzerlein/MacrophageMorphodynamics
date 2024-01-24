import numpy as np  # v 1.24.3
import pandas as pd  # v 2.0.1
import re

import Generate_files


def read_handpicked_excel_file(path):
    """
    :param path: pathlib movie_path to either
                    > the parent folder of a certain .tif movie file
                    > a '.tif' movie within a certain condition folder
    :return: Excel file with cell 'state' handpicked for all movies of the certain condition folder:
                check if labelled object is in fact a 'normal' RTM or if the cell was cut-off/ two cells were merged...
    """
    file_path = path / 'CellsHandpicked.xlsx'
    if path.name[-4:] == '.tif':
        file_path = path.parent / 'CellsHandpicked.xlsx'
    data = pd.read_excel(file_path, header=None)
    data = np.array(data)
    return data


def read_excel_file_seed_labels(path):
    """
        :param path: pathlib movie_path to either
                        > the Originals folder of a certain condition folder
                        > a '.tif' movie within a certain condition folder
        :return: Excel file with cell labels for initial seeds corrected if fixed cell part was separated
        """
    file_path = path / 'LabelSeedsHandpicked.xlsx'
    if path.name[-4:] == '.tif':
        file_path = path.parent / 'LabelSeedsHandpicked.xlsx'
    # CHECK IF FILE EXISTS; IF NOT, CREATE A DUMMY DATA ARRAY
    if not file_path.is_file():
        data = np.array([[path.name, '-1', '-1']])
    else:
        data = pd.read_excel(file_path)
        data = np.array(data)
    return data


def read_scaling(movie_path):
    """
    :param movie_path: Pathlib movie_path to .tif movie file
    :return length_scaling: length scaling [pxl / µm],
    :return time_scaling: time points [s] for the single frames, type int
    Info: if 'Scaling_...' file does not exist, the function to generate it is called
    """

    data_path = movie_path.parent.parent / 'NpyData'
    file_path = data_path / (re.sub(movie_path.name[-4:], '_Scalings.npy', movie_path.name))
    if not file_path.is_file():
        Generate_files.store_scaling(movie_path)

    scaling = np.load(file_path)
    length_scaling = scaling[0]
    time_scaling = scaling[1:].astype(int)

    return length_scaling, time_scaling


def read_grayscale(movie_path):
    """
    :param movie_path: Pathlib path to .tif movie file
    :return gray_arr: corresponding grayscale np array [x, y, t], type np.uint8
    :return amount_of_frames: amount of frames of movie t, type int
    Info: if grayscale file does not exist, the function to generate it is called
    """

    gray_arr_path = movie_path.parent.parent / 'NpyData' / re.sub(movie_path.name[-4:], '_gray.npy', movie_path.name)
    if not gray_arr_path.is_file():
        Generate_files.store_grayscale(movie_path)

    gray_arr = np.load(gray_arr_path)
    amount_of_frames = gray_arr.shape[2]

    return gray_arr, amount_of_frames


def read_binary(movie_path):
    """
    :param movie_path: Pathlib path to .tif movie file
    :return bin_arr: corresponding binary np array [x, y, t], type bool
    :return amount_of_frames: amount of frames of movie t, type int
    Info: if binary file does not exist, the function to generate it is called
    """

    bin_arr_path = movie_path.parent.parent / 'NpyData' / re.sub(movie_path.name[-4:], '_binary.npy', movie_path.name)
    if not bin_arr_path.is_file():
        Generate_files.store_binary(movie_path)

    bin_arr = np.load(bin_arr_path)
    amount_of_frames = bin_arr.shape[2]

    return bin_arr, amount_of_frames


def read_labels(path):
    """
    :param path: Pathlib path to .tif movie file
    :return label_arr: corresponding np array [x, y, t] with each cell labelled individually, type int8
    :return amount_of_frames: amount of frames of movie , type int
    Info: if label file does not exist, the function to generate it is called
    """

    label_arr_path = path.parent.parent / 'NpyData' / re.sub(path.name[-4:], '_labels.npy', path.name)
    if not label_arr_path.is_file():
        Generate_files.store_label_array(path)

    label_arr = np.load(label_arr_path)
    amount_of_frames = label_arr.shape[2]

    return label_arr, amount_of_frames


def read_cell_size_properties(path):
    """
    :param path: Pathlib path to .tif movie file
    :return areas: size of cells in µm for each time point t [µm^2], shape: [[amount_of_cells, t]
    :return fixed_areas: size of fixed portion of cells [µm^2], shape: [amount_of_cells]
    :return mobile_areas: size of mobile potion of each cell for each time point t [µm^2], shape: [amount_of_cells, t]
    :return perimeters: length of perimeter of cells for each time point t in µm [amount_of_cells, t]
    :return pixel_areas: size of cell as amount of pixels, shape: [amount_of_cells, t]
    :return intensity_areas: pixel_areas * intensity of each pixel, shape: [amount_of_cells, t]
    Info: if the CellSizeProperties file does not exist, the function to generate it is called
    """

    data_path = path.parent.parent / 'NpyData'
    prop_path = data_path / re.sub(path.name[-4:], '_CellSizeProperties.npy', path.name)
    if not prop_path.is_file():
        Generate_files.store_cell_properties(path)

    cell_size_properties = np.load(prop_path)
    area = cell_size_properties[:, :, 0]
    fixed_area = cell_size_properties[:, 0, 1].astype(int)
    mobile_area = area - fixed_area[:, None]
    perimeter = cell_size_properties[:, :, 2]
    pixel_area = cell_size_properties[:, :, 3]
    intensity_area = cell_size_properties[:, :, 4]

    return area, fixed_area, mobile_area, perimeter, pixel_area, intensity_area


def read_cell_shape_properties(path):
    """
    :param path: Pathlib path to .tif movie file
    :return solidities: shape [amount_of_cells, t]
    :return convexities: shape [amount_of_cells, t]
    :return eccentricities: shape [amount_of_cells, t]
    :return major axis: in [µm], shape [amount_of_cells, t]
    :return minor axis: in [µm], shape [amount_of_cells, t]
    :return circularity: shape [amount_of_cells, t] (form factor/ roundness)
    Info: if the CellShapeProperties file does not exist, the function to generate it is called
    """

    data_path = path.parent.parent / 'NpyData'
    prop_path = data_path / re.sub(path.name[-4:], '_CellShapeProperties.npy', path.name)
    if not prop_path.is_file():
        Generate_files.store_cell_properties(path)

    cell_shape_properties = np.load(prop_path)
    solidities = cell_shape_properties[:, :, 0]
    convexities = cell_shape_properties[:, :, 1]
    eccentricities = cell_shape_properties[:, :, 2]
    major_axis = cell_shape_properties[:, :, 3]
    minor_axis = cell_shape_properties[:, :, 4]
    circularity = cell_shape_properties[:, :, 5]

    return solidities, convexities, eccentricities, major_axis, minor_axis, circularity

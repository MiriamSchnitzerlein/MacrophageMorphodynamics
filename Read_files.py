import numpy as np  # v 1.25.0
import pandas as pd  # v 2.2.2
import re
import imageio  # v 2.31.1
import cv2 as cv  # opencv-python v 4.9.0.80

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


def read_tif_file(movie_path):
    """
        IDEA:   > read in .tif file of movie (MOVIENAME) (using imageio)
                > transform movie to grayscale (using cv)
                > store grayscale numpy array (type np.unit8) as MOVIENAME_gray.npy
        :param movie_path: Pathlib movie_path to .tif movie file
        :return: 0
        """

    # movie information:
    if movie_path.name[-4:] == '.tif':  # to read in as .tif file
        reader = imageio.v2.get_reader(movie_path)
        amount_of_frames = len(reader)
    else:  # if movie has another file format, still try to read it in
        reader = imageio.get_reader(movie_path, format='ffmpeg')
        movie_meta_data = reader.get_meta_data()
        amount_of_frames = int(movie_meta_data['fps'] * movie_meta_data['duration'])

    if len(list(reader.get_data(0).shape)) == 3:
        frame_dimension = list(reader.get_data(0).shape)  # gets shape of first image in video
        amount_of_frames = frame_dimension[0]
        gray_frame_dimensions = frame_dimension[1:3]
    else:
        frame_dimension = list(reader.get_data(0).shape)
        gray_frame_dimensions = frame_dimension

    # create numpy array to be filled with movie
    gray_frame_dimensions.append(amount_of_frames)
    gray_arr = np.zeros(gray_frame_dimensions, dtype=np.uint8)

    for i, color_frame in enumerate(reader):
        for frame_index in range(amount_of_frames):
            if len(color_frame.shape) > 3:
                gray_frame = cv.cvtColor(color_frame[frame_index, :, :], cv.COLOR_BGR2GRAY)
                gray_arr[:, :, frame_index] = gray_frame
            elif len(color_frame.shape) == 2:
                gray_arr[:, :, i] = color_frame
            else:
                gray_arr[:, :, frame_index] = color_frame[frame_index, :, :]

    reader.close()

    mat = re.match(r'(\S+)_([\a-zA-Z]+)\.tif', movie_path.name)
    # read file about removing frames from movies
    file_path = movie_path.parent.parent.parent / 'FrameRemoval.xlsx'
    if mat.groups()[1] == 'DeepCad' or mat.groups()[1] == 'Raw':
        file_path = movie_path.parent.parent.parent.parent / 'FrameRemoval.xlsx'
    data = pd.read_excel(file_path, header=None)
    data = np.array(data)
    removal = np.where(data[:, 0] == (mat.groups()[0]+'.tif'))[0]  # this is the line corresponding to current movie, if applicable
    if len(removal) != 0:
        removal = data[removal, :][0][1]
        removal = [int(s) for s in removal.split(',')]
        gray_arr = np.delete(gray_arr, removal, axis=2)

    return gray_arr
    

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

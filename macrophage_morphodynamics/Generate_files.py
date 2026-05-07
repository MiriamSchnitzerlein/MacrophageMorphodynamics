import numpy as np  # v 1.25.0
import pandas as pd  # v 2.2.2

import matplotlib.pyplot as plt  # matplotlib v 3.8.4
from matplotlib.colors import ListedColormap

# for image analysis/ manipulation
import imageio  # v 2.31.1
import cv2 as cv  # in generate_grayscale  # opencv-python v 4.9.0.80
import skimage  # scikit-image v 0.22.0

import scipy  # v 1.12.0

from pathlib import Path  # v 1.0.1
import shutil
import re

import Read_files  # reading in .npy files with data (movie files, cell properties etc)


def generate_color_map(n):
    """
    IDEA: use the colormap "tab10" and add colors by tinting (making brighter) the original colors a bit
    :param n: integer - minimal amount of colors the map should have
    :return: matplotlib colormap with int(np.ceil(n) / 10) colors
    """

    n = int(np.ceil(n / 10))
    original_colormap = plt.get_cmap('tab10')
    original_colorarray = np.array([original_colormap(i) for i in range(10)])
    my_colorarray = np.copy(original_colorarray)
    for tint in range(0, n - 1):
        add_cols = np.ones((10, 4))
        add_cols[:, :3] = original_colorarray[:, :3] + (1 - original_colorarray)[:, :3] / (n - tint)
        my_colorarray = np.append(my_colorarray, add_cols, axis=0)

    return ListedColormap(my_colorarray)


def store_scaling(movie_path):
    """
    IDEA:   store meta-data (i.e. time and length scaling of movie) in file MOVIENAME_Scaling.npy
            > read in metadata file 'InfoFor_...)
            > get amount of time points via 'SizeT'
            > find and store time of each frame via 'TimePoint(x) = ...'
            > find and store length scaling/ Resolution via 'Resolution'
    :param movie_path: Pathlib movie_path to .tif movie file
    :return: 0
    """

    print('For movie', movie_path.name, 'reading and storing length and time scaling.')

    file_name = movie_path.parent / ('InfoFor_' + re.sub(movie_path.name[-3:], 'txt', movie_path.name))

    data_path = movie_path.parent.parent / 'NpyData'
    Path(data_path).mkdir(parents=True, exist_ok=True)
    save_path = data_path / (re.sub(movie_path.name[-4:], '_Scalings.npy', movie_path.name))
    helper_path = data_path / (re.sub(movie_path.name[-4:], '_ScalingsOriginal.npy', movie_path.name))

    if not file_name.is_file() and not save_path.is_file() and not helper_path.is_file():
        raise SystemExit('Error: Neither the file', file_name.name, ' nor', save_path.name, 'was found. Please provide '
                                                                                            'at least one of the two '
                                                                                            'files! Program aborted!')
    elif helper_path.is_file():
        shutil.copy2(helper_path, save_path)
        return 0

    elif not file_name.is_file() and save_path.is_file():
        print('Info: The file ', file_name.name, ' was not found, but the scaling is already stored anyway.')
        return 0

    length_scale, time_points = np.nan, np.nan
    file_type = 0
    with open(file_name) as info_file:
        my_string = info_file.read(8)
        if my_string == ' BitsPer':
            file_type = 1
            for item in info_file:
                # matching amount of data points
                mat = re.match(r' SizeT = (\d+)', item)
                if mat:
                    vid_length = int(mat.groups()[0])
                    time_points = np.zeros(vid_length, dtype='datetime64[ms]')

                # matching string starting with 'TimePoint'
                mat = re.match(r'TimePoint(\d+) = (\d+-\d+-\d+) (\d+:\d+:\d+.\d+)', item)
                if mat:
                    tp_num = int(mat.groups()[0])
                    time_str = np.datetime64(mat.groups()[1] + 'T' + mat.groups()[2])

                    time_points[tp_num - 1] = time_str

                # matching length scaling
                mat1 = re.match(r'Resolution:\s+(\d+.\d+) pixels per micron', item)
                mat2 = re.match(r'Resolution:\s+(\d+) pixels per micron', item)
                if mat1:
                    length_scale = (float(mat1.groups()[0]))
                elif mat2:
                    length_scale = (float(mat2.groups()[0]))
        elif my_string == '(Fiji Is':
            file_type = 2
            for item in info_file:
                # matching length scaling
                mat1 = re.match(r'Resolution:\s+(\d+.\d+) pixels per micron', item)
                mat2 = re.match(r'Resolution:\s+(\d+) pixels per micron', item)
                if mat1:
                    length_scale = (float(mat1.groups()[0]))
                elif mat2:
                    length_scale = (float(mat2.groups()[0]))

                mat = re.match(r'Frame interval:\s+(\d+.\d+) sec', item)
                if mat:
                    frame_int = float(mat.groups()[0])

                mat = re.match(r'\s*Frame:\s+\d+/(\d+)(.|\n)+', item)
                if mat:
                    amount_of_frames = int(mat.groups()[0])

    if np.isnan(length_scale):
        raise ValueError('Resolution was not found in file InfoFor_', path.name)
    if file_type == 1:
        time_points = np.round((time_points - time_points[0]).astype(float) / 1000)
    elif file_type == 2:
        time_points = np.linspace(0, frame_int * amount_of_frames, num=amount_of_frames)
    time_points = np.append(length_scale, time_points)

    np.save(save_path, time_points)
    return 0


def store_grayscale(movie_path):
    """
    IDEA:   > read in .tif file of movie (MOVIENAME) (using imageio)
            > transform movie to grayscale (using cv)
            > store grayscale numpy array (type np.unit8) as MOVIENAME_gray.npy
    :param movie_path: Pathlib movie_path to .tif movie file
    :return: 0
    """

    print('For movie', movie_path.name, 'generating and storing grayscale numpy array.')
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

    data_path = movie_path.parent.parent / 'NpyData'
    Path(data_path).mkdir(parents=True, exist_ok=True)
    np.save(data_path / re.sub(movie_path.name[-4:], '_gray.npy', movie_path.name), gray_arr)

    reader.close()

    return 0


def remove_frames_restore_data(path):
    """
    IDEA:   > check for all stored .npy files, if they were stored
            > if so, assume that they maybe were (wrongly?) manipulated already
            > restore (time) scaling and grayscale array -> then read in and remove frames specified in removal
            > if files already exist: restore following files (binary, labels, cell_properties, ...)
    :param path: pathlib Path to the .tif file of the respective movie to be treated here
    :return: 0
    """

    # I assume that I need to restore the files first every time
    # (in case I have already removed some frames and want to remove others now),
    # before I can remove the specified frames!!
    data_path = path.parent.parent / 'NpyData'

    # Read in excel file with 'frames to be removed' per movie
    file_path = path.parent.parent.parent / 'FrameRemoval.xlsx'
    data = pd.read_excel(file_path, header=None)
    data = np.array(data)
    removal = np.where(data[:, 0] == path.name)[0]  # this is the line corresponding to current movie, if applicable
    if len(removal) == 0:
        print('No frames found to be removed for movie ', path.name)
        return 0
    removal = data[removal, :][0][1]
    removal = [int(s) for s in removal.split(',')]
    print('For movie', path.name, 'removing', removal, 'time frames from all stored numpy arrays.')
    # SCALING
    file_path = data_path / (re.sub(path.name[-4:], '_Scalings.npy', path.name))
    if file_path.is_file():  # PROBLEM WHEN REPEATING FRAME REMOVAL FOR CTRL.TIF!!!
        store_scaling(path)
        len_scaling, time_scaling = Read_files.read_scaling(path)
        time_scaling = np.delete(time_scaling, removal)
        time_points = np.append(len_scaling, time_scaling)
        np.save(file_path, time_points)

    # GRAYSCALE MOVIE
    file_path = data_path / (re.sub(path.name[-4:], '_gray.npy', path.name))
    if file_path.is_file():
        store_grayscale(path)
        gray_arr, _ = Read_files.read_grayscale(path)
        gray_arr = np.delete(gray_arr, removal, axis=2)
        np.save(file_path, gray_arr)

    # BINARY MOVIE
    file_path = data_path / (re.sub(path.name[-4:], '_binary.npy', path.name))
    if file_path.is_file():
        store_binary(path)

    # LABEL MOVIE
    file_path = data_path / (re.sub(path.name[-4:], '_labels.npy', path.name))
    if file_path.is_file():
        store_label_array(path)

    # CELL PROPERTIES
    file_path = data_path / (re.sub(path.name[-4:], '_CellSizeProperties.npy', path.name))
    if file_path.is_file():
        store_cell_properties(path)

    return 0


def store_binary(movie_path):
    """
    IDEA:   > read in grayscale .npy file of movie (MOVIENAME)
            > binarise movie (imageio.filters)
            > store binary numpy array (type bool) as MOVIENAME_binary.npy
    CHANGEABLE PARAMETERS: filter used is LI THRESHOLD
    :param movie_path: Pathlib movie_path to .tif movie file
    :return: 0
    """
    print('For movie', movie_path.name, 'generating and storing binary numpy array.')

    gray_arr, amount_of_frames = Read_files.read_grayscale(movie_path)

    bin_arr = np.zeros(gray_arr.shape, dtype=bool)  # to store binary version of movie
    for frame in range(0, amount_of_frames):
        threshold = skimage.filters.threshold_li(gray_arr[:, :, frame])
        bin_arr[:, :, frame] = gray_arr[:, :, frame] > threshold

    data_path = movie_path.parent.parent / 'NpyData'
    Path(data_path).mkdir(parents=True, exist_ok=True)
    np.save(data_path / re.sub(movie_path.name[-4:], '_binary.npy', movie_path.name), bin_arr)

    return 0


def generate_binary_preprocessed(binary_array, length_scaling, opening_r=0.6):
    """
    :param binary_array: 2 D array, original binary image that is being opened, and small holes and objects are removed
    :param length_scaling: float [pxl/µm], to convert between pxl and µm
    :param opening_r: float [µm], which radius the disk for the opening operation should have
    :return: 2D array, processed binary image with smoothed outline
    PARAMETERS USED: holes and objects < 10 µm^2 are being removed
    """

    max_obj_size = 10 * (length_scaling ** 2)  # convert µm^2 to pxl;
    se_open = skimage.morphology.disk(int(opening_r * length_scaling))

    bin_arr_clea = np.copy(binary_array)
    for frame in range(binary_array.shape[-1]):  # smooth cell boundaries via closing
        bin_arr_clea[:, :, frame] = skimage.morphology.binary_closing(bin_arr_clea[:, :, frame], se_open)
        bin_arr_clea[:, :, frame] = skimage.morphology.remove_small_holes(bin_arr_clea[:, :, frame],
                                                                          int(max_obj_size))
        bin_arr_clea[:, :, frame] = skimage.morphology.remove_small_objects(bin_arr_clea[:, :, frame],
                                                                            int(max_obj_size), connectivity=2)

    return bin_arr_clea


def generate_mean_shape(movie_path):
    """
    IDEA:   > read in binary .npy file of movie (MOVIENAME)
            > generate mean of movie over time
    :param movie_path: Pathlib movie_path to .tif movie file
    :return: mean shape over time, mean shape binarized, mean shape binarized, pre-processed and labelled
    CHANGEABLE PARAMETERS: pre-processing: size of objects and holes to be removed (25 µm^2, 50 µm^2)
    """
    bin_arr, _ = Read_files.read_binary(movie_path)
    len_scaling, _ = Read_files.read_scaling(movie_path)
    mean_arr = np.mean(bin_arr, axis=2)  # this is not binary, values between 0 and 1
    mean_arr_bin = mean_arr >= 0.97

    max_obj_size = 25 * (len_scaling ** 2)  # convert µm^2 to #pxl; remove holes/ objects from seed labels in meanshape

    #####################
    # (LABEL) MEAN SHAPE OF IMAGE
    #####################
    # ## CLEANING ##
    mean_arr_bin = skimage.morphology.remove_small_holes(mean_arr_bin, max_obj_size)  # remove small holes in image
    mean_arr_bin = skimage.morphology.remove_small_objects(mean_arr_bin, max_obj_size * 2)

    # ## LABELLING ##
    mean_arr_labels = skimage.measure.label(mean_arr_bin)

    #################
    # RE-CHECK (AND POTENTIALLY RE-DEFINE) SEED LABELS VIA EXCEL FILE
    #################
    mean_arr_labels_corrected = np.copy(mean_arr_labels)
    corrected_labels = Read_files.read_excel_file_seed_labels(movie_path)
    corr_labels_movie = corrected_labels[np.where(corrected_labels[:, 0] == movie_path.name)[0], :]
    corr_labels_movie = corr_labels_movie[corr_labels_movie[:, 1].argsort()]
    corr_labels_movie = corr_labels_movie[:, 1:].astype(np.int8)  # sort by cell number
    if corr_labels_movie[0, 0] != -1:
        for cell in range(np.max(mean_arr_labels)):
            if corr_labels_movie[cell, 0] + 1 != corr_labels_movie[cell, 1] + 1:
                mean_arr_labels_corrected[mean_arr_labels == corr_labels_movie[cell, 0] + 1] = \
                    corr_labels_movie[cell, 1] + 1
        # relabel, so that labels are consecutive! (i.e. [0, 1, 2, 3] instead of [0, 2, 3, 5])
        curr_labels = np.unique(mean_arr_labels_corrected)
        wanted_labels = np.arange(0, len(curr_labels))
        if np.any(curr_labels != wanted_labels):
            for cell in range(1, len(curr_labels)):  # no need to relabel background (=0)
                mean_arr_labels_corrected[mean_arr_labels_corrected == curr_labels[cell]] = cell

    return mean_arr, mean_arr_bin, mean_arr_labels, mean_arr_labels_corrected


def store_label_array(movie_path):
    """
    IDEA:   1) read in binary .npy movie file, labelled meanshape frame, length scaling
            2) for every frame, use seeds from labelled meanshape frame in watershed algorithm to get first labels
            3) dilate binary file (so cells touch disconnected protrusions) (DIAMOND SHAPE with 2.2 µm 'radius')
            4) use first labels as seeds in watershed to label dilated cells, -> get second labels
            5) use second labels to label original binary file (so that disconnected protrusions are labelled correctly)
            6) close resulting labelled file (to smooth cell boundary etc) (CIRCULAR SHAPE with 0.9 µm radius)
                and remove small holes (< ) and small objects ( ) via fct generate_binary_preprocessed -> see generate
                binary preprocessed
            7) create 'fixed' boundaries between cells (avoid cell connection created by watershed jumping around
                between frames)
            8) store result from 7)  as MOVIENAME_labels.npy
    :param movie_path: pathlib Path to the .tif movie which is to be analysed here
    :return: 0
    CHANGEABLE PARAMETERS: preprocessing of mean shape: small holes < 25 µm^2, objects < 50µm^2
        dilation of binary: diamond shape with 2.2 µm radius
        closing of resulting label array: circle/ disk with 0.9 µm radius, small obj < 10 µm^2, holes < 10 µm^2
    """

    print('For movie', movie_path.name, 'generating and storing labelled numpy array.')

    #################
    # READ-IN
    #################
    gray_arr, _ = Read_files.read_grayscale(movie_path)
    bin_arr, amount_of_frames = Read_files.read_binary(movie_path)
    len_scaling, _ = Read_files.read_scaling(movie_path)
    _, mean_arr_bin, _, mean_arr_labels = generate_mean_shape(movie_path)
    amount_of_cells = np.max(mean_arr_labels)

    ####################
    # CHANGEABLE PARAMETERS:
    ####################
    # initialize structuring element for dilation of binary images
    se = skimage.morphology.diamond(int(2.2 * len_scaling))

    ####################
    # LABEL MOVIE
    ####################
    # INITIALIZING ARRAYS   ( initialize movie arrays (one region has one int label) )
    label_arr = np.zeros(bin_arr.shape, dtype=np.int8)  # to store labelled version

    # ACTUAL LABELLING
    for frame in range(0, amount_of_frames):
        if frame % 50 == 0:
            print('\tFrame', frame, 'of', amount_of_frames)
        # labelling the 'ORIGINAL' binary image
        rw_labels = np.copy(mean_arr_labels)
        rw_labels[np.where(bin_arr[:, :, frame] == 0)] = -1  # background (always dark) should not be labelled
        tempor_labelled_arr = skimage.segmentation.watershed(image=bin_arr[:, :, frame], markers=rw_labels,
                                                             mask=bin_arr[:, :, frame])
        # labelling the DILATED binary image (finding cutoff protrusions)
        dilated_binary = skimage.morphology.binary_dilation(bin_arr[:, :, frame], footprint=se)
        rw_labels_2 = np.zeros(dilated_binary.shape, dtype=np.int8)
        rw_labels_2[dilated_binary] = tempor_labelled_arr[dilated_binary]
        tempor_labelled_arr_dil = skimage.segmentation.watershed(image=dilated_binary, markers=rw_labels_2,
                                                                 mask=dilated_binary)
        # having CLEANED binary image with cutoff protrusions labelled correctly
        bin_arr_clea = generate_binary_preprocessed(bin_arr, len_scaling)
        if movie_path.name == 'Ctrl.tif':  # not necessary for version 2 of the data
            bin_arr_clea = generate_binary_preprocessed(bin_arr, len_scaling, opening_r=1.2)
        tempor_labelled_arr_final = bin_arr_clea[:, :, frame] * tempor_labelled_arr_dil

        # ## STORING correctly
        label_arr[:, :, frame] = tempor_labelled_arr_final

    # CORRECT RANDOM 'MISLABELLING'
    label_arr_corrected = np.copy(label_arr)
    for frame in range(0, amount_of_frames):
        # CORRECTING RANDOM MISLABELLING, introducing 'fixed' borders between touching cells
        arr = np.copy(label_arr[:, :, frame])
        for cell in range(0, amount_of_cells):
            arr[arr == 0] = -1
            cell_ids = np.where(label_arr[:, :, frame] == cell + 1)
            for pixel_id in range(0, len(cell_ids[0])):
                # 'running average' -> only care about +- 10 frames of current one!
                cut_through = label_arr[cell_ids[0][pixel_id], cell_ids[1][pixel_id], :]
                cut_through = cut_through[cut_through > 0]
                most_occ_value = np.argmax(np.bincount(cut_through))
                arr[cell_ids[0][pixel_id], cell_ids[1][pixel_id]] = most_occ_value

        marker_arr = np.zeros_like(label_arr[:, :, frame], dtype=np.int8)
        mask_arr = np.full(label_arr[:, :, frame].shape, False, dtype=bool)
        # relabel disconnected tiny regions
        relabel = False
        for cell in range(0, amount_of_cells):
            cell_arr = (arr == cell + 1).astype(np.int8)  # array where only current cell (label) exists
            cell_arr_labeled = skimage.morphology.label(cell_arr)  # label all objects in array
            if np.max(np.unique(cell_arr_labeled)) != 1:  # if there is more than 1 isolated object in the array

                reg_pro = skimage.measure.regionprops(cell_arr_labeled)  # region properties of all possible objects
                for sub_object in range(0, np.max(cell_arr_labeled)):  # all isolated objects having the current label
                    if reg_pro[sub_object].area < 8 * len_scaling:  # everything < 8 µm is counted as mislabelling
                        relabel = True
                        arr[reg_pro[sub_object].coords[:, 0], reg_pro[sub_object].coords[:, 1]] = 0
                        # these following points have to be relabelled
                        mask_arr[reg_pro[sub_object].coords[:, 0], reg_pro[sub_object].coords[:, 1]] = True
            marker_arr[arr[:, :] == cell + 1] = cell + 1  # these points are correctly labelled and are used as seeds
            # all points which are correctly labelled need to show up in mask, else watershed will delete them!
            mask_arr[arr[:, :] == cell + 1] = True

        if relabel:
            label_arr_corrected[:, :, frame] = skimage.segmentation.watershed(bin_arr_clea[:, :, frame], markers=marker_arr, mask=mask_arr)
        else:
            arr[arr == -1] = 0
            label_arr_corrected[:, :, frame] = arr

    ##############################
    # STORE MAIN RESULTS/ KEY DATA
    ##############################
    data_path = movie_path.parent.parent / 'NpyData'
    Path(data_path).mkdir(parents=True, exist_ok=True)
    np.save(data_path / re.sub(movie_path.name[-4:], '_labels.npy', movie_path.name), label_arr_corrected)

    return 0


def store_cell_properties(movie_path):
    """
    IDEA:   1) read in grayscale, binary, label .npy movie file, meanshape frame, length scalings
            2) use labelled binary mean shape to extract fixed areas
            3) use skimage.morphology.region_props to extract -> area, perimeter, cell shape parameters
                cell size parameters: area (in µm and pxl), perimeter [µm], fixed area [µm^2], intensity area [pxl^2]
                cell shape parameters:  solidity, convexity, eccentricity, major_axis, minor_axis
                                        protrusiveness, angularity, length of longest protrusion
                dynamic cell area change: change in area between frames, time points of consecutive frames
            4) store everything
    :param movie_path: pathlib movie_path to the .tif movie which is to be analysed here
    :return: 0, stores results as data arrays (.npy)

    CHANGEABLE PARAMETERS: removing small objects/ holes for binary mean shape -> fixed area
    """

    print('For movie', movie_path.name, 'generating and storing cell properties (size and shape).')

    #################
    # READ-IN
    #################
    gray_arr, amount_of_frames = Read_files.read_grayscale(movie_path)
    bin_arr, _ = Read_files.read_binary(movie_path)
    mean_arr, mean_arr_bin, _, mean_arr_labels = generate_mean_shape(movie_path)
    len_scaling, time_scaling = Read_files.read_scaling(movie_path)
    label_arr, _ = Read_files.read_labels(movie_path)

    # ## CONSTANT/ FIXED AREA ##
    mean_arr, mean_arr_bin, _, mean_arr_labels = generate_mean_shape(movie_path)
    amount_of_cells = np.max(mean_arr_labels)

    mean_arr_binary = mean_arr >= 0.97
    mean_arr_labelled = mean_arr_binary * (label_arr[:, :, 0] + 1)  # multiply with 1st frame
    mean_arr_labelled -= 1
    mean_arr_labelled[np.where(mean_arr_labelled == -1)] = 0

    mean_arr_regpro = skimage.measure.regionprops(mean_arr_labelled)

    fixed_area = np.zeros(amount_of_cells)
    for cell in range(0, amount_of_cells):
        fixed_area[cell] = mean_arr_regpro[cell].area

    #########
    # PARAMETERS FOR SHOLL ANALYSIS
    min_rad = 8 * len_scaling  # minimum radius in micron, transformed to pxl via length scaling
    max_rad = 56 * len_scaling  # maximum radius in micron
    increm = 8 * len_scaling  # increments of circle radii
    # PARAMETER FOR EROSION-DILATION
    radius = 2 * len_scaling  # in micron  # radius of disk for erosion-dilation
    my_shape = skimage.morphology.disk(radius)

    ####################
    # GENERATE CELL PROPERTIES FROM LABELLED MOVIES
    ####################
    # INITIALIZING ARRAYS
    # initialize arrays for cell properties (size and shape)
    # area, fixed_area, perimeter
    cell_size_properties = np.zeros((amount_of_cells, amount_of_frames, 3))
    # storing the fixed area (at all time points, in case I need to remove frame 0 at some point)
    cell_size_properties[:, :, 1] = np.full((amount_of_cells, amount_of_frames), fixed_area[:, None])

    # solidity, convexity, eccentricity, major_axis, minor_axis, circularity (=form factor), protrusiveness, angularity,
    # length of longest protrusion
    cell_shape_properties = np.zeros((amount_of_cells, amount_of_frames, 9))

    for frame in range(0, amount_of_frames):
        if frame % 50 == 0:
            print('\tframe ', frame)

        lab_arr = label_arr[:, :, frame]
        region_props = skimage.measure.regionprops(lab_arr)

        for cell in range(0, amount_of_cells):
            # if at some point, not all cells are detected:
            if cell < region_props[-1].label and cell < len(region_props):

                arr = (lab_arr == cell + 1)  # array where only current cell 'exists'
                if np.sum(arr) == 0:
                    print('Error 404: cell ', cell + 1, ' not found in frame', frame)
                    continue

                # CELL SIZE PROPERTIES
                cell_size_properties[cell, frame, 0] = region_props[cell].area
                cell_size_properties[cell, frame, 2] = region_props[cell].perimeter

                # CELL SHAPE PROPERTIES
                cell_shape_properties[cell, frame, 0] = region_props[cell].solidity
                cell_shape_properties[cell, frame, 2] = region_props[cell].eccentricity
                cell_shape_properties[cell, frame, 3] = region_props[cell].axis_major_length
                cell_shape_properties[cell, frame, 4] = region_props[cell].axis_minor_length

                # measure convex hull for CONVEXITY
                chull = skimage.morphology.convex_hull_image(arr)
                convex_peri = skimage.measure.regionprops(skimage.measure.label(chull))[0].perimeter
                cell_shape_properties[cell, frame, 1] = convex_peri / cell_size_properties[cell, frame, 2]

                # PROTRUSIVENESS
                circle_origin = np.round(mean_arr_regpro[cell].centroid).astype(int)  # center of mass of fixed area (for circle origin)
                # for each circle radius (increasing):
                sholl_intercept_counter = 0
                cell_arr = np.copy(arr)
                for circle_radius in np.arange(min_rad, max_rad, increm, dtype=int):
                    sholl_circle_arr = np.zeros_like(arr, dtype=np.uint8)
                    rr, cc = skimage.draw.circle_perimeter(circle_origin[0], circle_origin[1], circle_radius,
                                                           shape=sholl_circle_arr.shape)
                    sholl_circle_arr[rr, cc] = 1
                    circle = skimage.segmentation.find_boundaries(skimage.morphology.disk(circle_radius), mode='inner')
                    sholl_intercept_im = sholl_circle_arr * cell_arr
                    if np.sum(sholl_intercept_im) != np.sum(circle):  # else the whole circle is covered which is not interesting for us
                        amount_of_intercepts = np.max(skimage.morphology.label(sholl_intercept_im, connectivity=2))
                        sholl_intercept_counter += amount_of_intercepts
                cell_shape_properties[cell, frame, 6] = sholl_intercept_counter

                # ANGULARITY
                eroded_dilated_cell = skimage.morphology.binary_erosion(cell_arr, footprint=my_shape)
                eroded_dilated_cell = skimage.morphology.binary_dilation(eroded_dilated_cell, footprint=my_shape)
                # region_props = skimage.measure.regionprops(eroded_dilated_cell)
                erod_dil_cell_area = np.sum(eroded_dilated_cell != 0)
                cell_shape_properties[cell, frame, 7] = erod_dil_cell_area

                # LENGTH OF LONGEST PROTRUSION
                fixed_contour = np.where(
                    skimage.segmentation.find_boundaries(mean_arr_labels == cell + 1, mode='inner'))
                cell_contour = np.where(
                    skimage.segmentation.find_boundaries(arr, mode='inner'))
                dist = scipy.spatial.distance.cdist(np.stack([cell_contour[0], cell_contour[1]], axis=1),
                                                    np.stack([fixed_contour[0], fixed_contour[1]], axis=1))
                distances = np.min(dist, axis=1)
                cell_shape_properties[cell, frame, 8] = np.max(distances)

    # dynamic area change
    # find right 'time intervals' for integrating area
    wanted_time_interval = 14  # we want 14 seconds between 2 consecutive frames

    time_steps, time_steps_indices = np.ones_like(time_scaling) * (-1), np.ones_like(time_scaling) * (-1)
    time_steps[0], time_steps_indices[0] = 0, 0

    # initialise time steps, with step length defined by wanted_time_interval
    for time_point_ind in range(len(time_steps - 1)):
        next_time_point = time_steps[time_point_ind] + wanted_time_interval
        if next_time_point <= time_scaling[-1]:
            next_time_point_ind = np.argmin(np.abs(time_scaling - next_time_point))
            next_time_point_rounded = time_scaling[next_time_point_ind]
            if next_time_point_rounded not in time_steps:
                time_steps_indices[time_point_ind + 1] = next_time_point_ind
                time_steps[time_point_ind + 1] = next_time_point_rounded
        else:
            break

    time_steps, time_steps_indices = time_steps[time_steps > -1], time_steps_indices[time_steps_indices > -1]
    dynamic_area_change = np.zeros((amount_of_cells + 1, len(time_steps_indices)-1))
    dynamic_area_change[-1, :] = time_steps[:-1]

    for cell in range(0, amount_of_cells):
        for frame_ind in range(len(time_steps_indices[:-1])):

            lab_arr = label_arr[:, :, time_steps_indices[frame_ind]]
            arr = (lab_arr == cell + 1)  # array where only current cell 'exists'
            arr_next = (label_arr[:, :, time_steps_indices[frame_ind+1]] == cell+1)
            cell_area_difference = np.sum(arr ^ arr_next)
            dynamic_area_change[cell, frame_ind] = cell_area_difference

    ##############################
    # SCALE AND STORE MAIN RESULTS/ KEY DATA
    ##############################
    # Scaling from pxl to µm with [len_scaling] = pxl/µm
    cell_size_properties[:, :, 0] /= (len_scaling ** 2)  # scaling the area
    cell_size_properties[:, :, 2] /= len_scaling  # scaling the perimeter
    cell_size_properties[:, 0, 1] /= (len_scaling ** 2)  # scaling the fixed/ immobile

    cell_shape_properties[:, :, 3] /= len_scaling  # scaling the major axis
    cell_shape_properties[:, :, 4] /= len_scaling  # scaling the minor axis
    cell_shape_properties[:, :, 5] = (4 * np.pi * cell_size_properties[:, :, 0]) / (cell_size_properties[:, :, 2] ** 2)  # circularity
    cell_shape_properties[:, :, 7] /= (len_scaling ** 2)  # scaling the erosion-dilation-area = angularity
    cell_shape_properties[:, :, 7] = (cell_size_properties[:, :, 0] - cell_shape_properties[:, :, 7]) / cell_size_properties[:, :, 0]
    cell_shape_properties[:, :, 8] /= len_scaling  # scaling the length of the longest protrusion

    dynamic_area_change[:-1, :] /= (len_scaling ** 2)  # scaling the dynamic area change

    # storing the data arrays
    data_path = movie_path.parent.parent / 'NpyData'
    np.save((data_path / re.sub(movie_path.name[-4:], '_CellSizeProperties.npy', movie_path.name)),
            cell_size_properties)
    np.save((data_path / re.sub(movie_path.name[-4:], '_CellShapeProperties.npy', movie_path.name)),
            cell_shape_properties)
    np.save((data_path / re.sub(movie_path.name[-4:], '_DynamicAreaChange.npy', movie_path.name)),
            dynamic_area_change)

    return 0



import numpy as np  # v 1.24.3
import pandas as pd  # v 2.0.1

from matplotlib import ticker  # matplotlib v 3.7.1
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.animation as anim
import matplotlib.transforms as transforms
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import Divider, Size
from matplotlib.ticker import FormatStrFormatter

from pathlib import Path  # v 1.0.1

import scipy  # v 1.9.1
import skimage  # scikit-image v 0.20.0
from sklearn.linear_model import LinearRegression  # scikit-learn v 1.2.2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FactorAnalysis, FastICA, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import umap  # umap-learn v 0.5.5

import Read_files, Generate_files


def Fig1_CellSnapshots(movie_path, condition='BMDM'):
    """
    Create (pre-processed) Movie Snapshots for Figure 1; further processing steps are done outside of this script
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie files are chosen in the fct)
    :param condition: str - 'BMDM' or 'Ctrl', whether to plot the snapshots for
            - BMDM cells in vitro  or
            - RTM cells in vivo
    :return: 0
    """
    parent_path = movie_path.parent.parent.parent  # go to 'outermost' directory ('RTM_Data')

    ########
    # DEFINE PARAMETERS FOR IMAGE ENHANCING AND PLOTTING
    # parameters are stored for both conditions as [BMDM, Ctrl]
    ########
    movie_name = ['BMDM_2023-07-06.tif', 'Ctrl_2021-05-21_2.tif']
    # parameters for plotting the whole tissue
    fig_size_whole_tissue = [2.36, 1.57]  # in inches
    fig_width_whole_tissue = 280  # in µm
    fig_height_whole_tissue = fig_width_whole_tissue * fig_size_whole_tissue[1] / fig_size_whole_tissue[0]
    ul_corner_tissue = [[750, 750], [0, 0]]  # upper left corner of tissue snapshot (in [pxl]) for plotting
    framenr_tissue = [0, 2]  # which frame number/ time point to use for plotting the tissue snapshot

    # parameters for plotting exemplary cells
    fig_size_single_cell = np.array([[2.36, 0.4], [0.79, 1.1842]])  # in inches
    fig_width_mum_single_cell = np.array([195, 65])  # in µm, for single cell snapshots
    fig_height_mum = (fig_width_mum_single_cell * fig_size_single_cell[:, 1] / fig_size_single_cell[:, 0]).astype(int)
    ul_corner_single_cell = [[1090, 1600], [150, 180]]
    framenr_single_cell = [[0, 13, 23], [0, 40, 87]]

    c_lim = [0.025, 0.02]
    gain = [8, 12]
    cut_off = [0.4, 0.42]
    gamma = [1.5, 5]
    gamma_gain = [2, 4]
    kernel_size_factor = [20, 10]

    if condition == 'BMDM':
        index = 0
    elif condition == 'Ctrl':
        index = 1
    else:
        print('Available inputs for the summary_stat "condition" are only "BMDM" or "Ctrl"!\n'
              'No snapshots have been saved for the condition ', condition)
        return 0

    # READ-IN #
    movie_path = parent_path / condition / 'Originals' / movie_name[index]
    gray_arr, _ = Read_files.read_grayscale(movie_path)
    len_scaling, time_scaling = Read_files.read_scaling(movie_path)

    #######
    # PLOT WHOLE TISSUE
    #######

    fig = plt.figure(figsize=fig_size_whole_tissue)
    fig.patch.set_facecolor('white')
    # fig.patch.set_facecolor('black')
    ax = fig.add_subplot()
    ax.axis('off')

    # enhance image contrast
    im_to_plot = gray_arr[ul_corner_tissue[index][1]:ul_corner_tissue[index][1] + int(
        fig_height_whole_tissue * len_scaling), ul_corner_tissue[index][0]:ul_corner_tissue[index][0] + int(
        fig_width_whole_tissue * len_scaling), framenr_tissue[index]]
    im_to_plot = 255 - im_to_plot
    kernel_size = (im_to_plot.shape[0] / kernel_size_factor[index], im_to_plot.shape[1] / kernel_size_factor[index])
    im_to_plot = skimage.exposure.equalize_adapthist(im_to_plot, kernel_size=kernel_size, clip_limit=c_lim[index])
    im_to_plot = skimage.exposure.adjust_sigmoid(im_to_plot, gain=gain[index], cutoff=cut_off[index])
    im_to_plot = skimage.exposure.adjust_gamma(im_to_plot, gamma=gamma[index], gain=gamma_gain[index])

    ax.imshow(im_to_plot, cmap='gray')

    # plot scale bar
    len_scalebar = 30 * len_scaling
    ylim = ax.get_ylim()[0]
    xlim = ax.get_xlim()[1]
    dist = 0.078 / fig_size_whole_tissue[1]  # keep distance of 2 mm = 0.078 inch from image border
    ax.plot([dist * ylim, dist * ylim + len_scalebar], [ylim - dist * ylim, ylim - dist * ylim], lw=1.2, color='black')
    #ax.plot([dist * ylim, dist * ylim + len_scalebar], [ylim - dist * ylim, ylim - dist * ylim], lw=1.2, color='white')

    fig.tight_layout(pad=0)
    fig.set_dpi(600)
    fig_path = Path().absolute() / 'Fig1'
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / (condition + '_WholeTissueSnap.png'), dpi=600)
    plt.close('all')

    ############
    # PLOT SINGLE CELL SNAPSHOTS
    ############
    for frame in framenr_single_cell[index]:
        fig = plt.figure(figsize=fig_size_single_cell[index])
        fig.patch.set_facecolor('white')
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot()
        ax.axis('off')

        im_to_plot = gray_arr[ul_corner_single_cell[index][1]:ul_corner_single_cell[index][1] + int(
            fig_height_mum[index] * len_scaling),
                     ul_corner_single_cell[index][0]:ul_corner_single_cell[index][0] + int(
                         fig_width_mum_single_cell[index] * len_scaling),
                     frame]
        im_to_plot = 255 - im_to_plot
        im_to_plot = skimage.exposure.equalize_adapthist(im_to_plot, kernel_size=kernel_size, clip_limit=c_lim[index])
        im_to_plot = skimage.exposure.adjust_sigmoid(im_to_plot, gain=gain[index], cutoff=cut_off[index])
        im_to_plot = skimage.exposure.adjust_gamma(im_to_plot, gamma=gamma[index], gain=gamma_gain[index])

        ax.imshow(im_to_plot, cmap='gray')

        # plot scalebar only for last resp. first frame
        if (frame == framenr_single_cell[index][-1] and condition == 'BMDM') or (
                frame == framenr_single_cell[index][0] and condition == 'Ctrl'):
            len_scalebar = 30 * len_scaling
            ylim = ax.get_ylim()[0]
            xlim = ax.get_xlim()[1]
            dist = 0.078 / fig_size_single_cell[index][1]
            ax.plot([dist * ylim, dist * ylim + len_scalebar], [ylim - dist * ylim, ylim - dist * ylim], lw=1.2,
                   color='black')
            # ax.plot([dist * ylim, dist * ylim + len_scalebar], [ylim - dist * ylim, ylim - dist * ylim], lw=1.2,
              #       color='white')

        fig.tight_layout(pad=0)
        fig.set_dpi(600)
        fig_path = Path().absolute() / 'Fig1'
        Path(fig_path).mkdir(parents=True, exist_ok=True)
        time = time_scaling[frame]
        fig.savefig(fig_path / (condition + '_SingleCellSnap_T' + str(time) + '.png'), dpi=600)
        plt.close('all')

    return 0


# c
def Fig2_CellSnapshots(movie_path):
    """
    Store snapshots of an RTM at different time points with its convex hull, its perimeter and the outline of its
    fixed area marked with color
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie file is chosen in the fct)
    :return: 0, stores snapshots in subfolder in current working directory
    """

    movie_path = movie_path.parent.parent.parent / 'Ctrl' / 'TifData' / 'Ctrl.tif'  # use specifically this movie for the plot

    gray_arr, _ = Read_files.read_grayscale(movie_path)
    label_arr, _ = Read_files.read_labels(movie_path)
    _, mean_arr_bin, _, _ = Generate_files.generate_mean_shape(movie_path)

    len_scaling, _ = Read_files.read_scaling(movie_path)

    my_colors = ['orange', 'sandybrown', 'white', 'cornflowerblue']
    frames_to_plot = [0, 15, 36, 42]

    for frame in frames_to_plot:
        # ## CONVEX HULL
        convex_hull = skimage.morphology.convex_hull_object(label_arr[:, :, frame])
        convex_hull_border = skimage.segmentation.find_boundaries(convex_hull, mode='inner')
        for i in range(3):
            convex_hull_border = (skimage.morphology.binary_dilation(convex_hull_border))
        chull_mask = convex_hull - label_arr[:, :, frame]
        image_to_plot = skimage.color.label2rgb(chull_mask, gray_arr[:, :, frame], colors=my_colors)
        image_to_plot[convex_hull_border, :] = mcol.to_rgb(my_colors[1])

        # ## BORDERS OF FIXED AREA
        mean_arr_bin[~label_arr[:, :, frame]] = 0
        fixed_area_border = skimage.segmentation.find_boundaries(mean_arr_bin, mode='inner')
        for i in range(3):
            fixed_area_border = (skimage.morphology.binary_dilation(fixed_area_border))
        image_to_plot[fixed_area_border, :] = mcol.to_rgb(my_colors[2])

        # ## BORDERS OF WHOLE CELL
        area_border = skimage.segmentation.find_boundaries(label_arr[:, :, frame], mode='inner')
        for i in range(4):
            area_border = (skimage.morphology.binary_dilation(area_border))
        image_to_plot[area_border, :] = mcol.to_rgb(my_colors[3])

        # figure and image size
        fig_size = np.array([1.9, 1.2])
        width_mum = 60  # wanted width in µm
        fig_size_mum = fig_size * width_mum / fig_size[0]  # now in µm
        allowed_box_size = fig_size_mum * len_scaling

        # bounding box of cell
        reg_pro = skimage.measure.regionprops(label_arr[:, :, frame])
        my_bbox = np.array(reg_pro[0].bbox)
        current_box_size = np.array([my_bbox[3] - my_bbox[1], my_bbox[2] - my_bbox[0]])
        added_size = ((allowed_box_size - current_box_size) / 2).astype(int)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(image_to_plot[my_bbox[0] - added_size[1]:my_bbox[2] + added_size[1],
                  my_bbox[1] - added_size[0]:my_bbox[3] + added_size[0]])
        # add scalebar to last frame
        if frame == frames_to_plot[-1]:
            ax.plot([5 / fig_size_mum[0], 15 / fig_size_mum[0]], [0.9, 0.9], lw=0.9, transform=ax.transAxes, color='white')

        fig.tight_layout(pad=0)
        fig.set_dpi(400)

        fig_path = Path().absolute() / 'Fig2'
        Path(fig_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path / ('RTMSnapshot' + str(frame) + '.png'), dpi=600)
        plt.close(fig)

    return 0


# c
def Fig2_CellSize_Plot(movie_path):
    """
    Store plot showing cell area (whole, mobile and fixed) and perimeter for a specific cell of a specific movie
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie file is chosen in the fct)
    :return: 0, stores plot in subfolder in current working directory
    """

    movie_path = movie_path.parent.parent.parent / 'Ctrl' / 'TifData' / 'Ctrl.tif'

    areas, fixed_areas, mobile_areas, perimeters, _, _ = Read_files.read_cell_size_properties(movie_path)
    len_scaling, time_scaling = Read_files.read_scaling(movie_path)
    cell = 0
    mark_frames = [0, 15, 36, 42]

    color = 'cornflowerblue'
    plt.rcParams["font.family"] = "Arial"
    label_fontsize = 8
    fig_size = np.array([1.7, 2.25])

    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(3, 1, figure=fig)
    axs = [fig.add_subplot(gs[:2, 0]),  # area plot
           fig.add_subplot(gs[2, 0])]  # perimeter plot

    axs[0].tick_params(axis='y', size=2, width=1, labelsize=label_fontsize, pad=1)
    axs[0].xaxis.set_major_formatter(ticker.NullFormatter())
    axs[0].tick_params(axis='x', size=2, width=1, pad=1)

    axs[0].set_ylabel(r'Area $[\mu m ^2]$', fontsize=label_fontsize, fontfamily='Arial', labelpad=2)
    axs[0].plot(time_scaling, areas[cell, :], color=color, lw=1.25)
    axs[0].plot(time_scaling, mobile_areas[cell, :], ls=(0, (3, 1)), color=color, lw=1.25)
    axs[0].axhline(fixed_areas[cell], color=color, ls=':', lw=1.25)

    axs[1].set_ylabel(r'Perimeter $[\mu m]$', fontsize=label_fontsize, fontfamily='Arial', labelpad=2)
    axs[1].tick_params(axis='y', size=2, width=1, labelsize=label_fontsize, pad=1)
    axs[1].tick_params(axis='x', size=2, width=1, labelsize=label_fontsize, pad=1)
    axs[1].plot(time_scaling, perimeters[cell, :], color=color, lw=1.25)
    axs[1].set_xlabel(r'Time $[s]$', fontsize=label_fontsize, fontfamily='Arial', labelpad=2)

    # mark the frames of the movie snapshots in the plot
    for index in mark_frames:
        axs[0].plot(time_scaling[index], areas[cell, index], ls='', marker='.', color='orange')
        axs[0].plot(time_scaling[index], mobile_areas[cell, index], ls='', marker='.', color='orange')
        axs[1].plot(time_scaling[index], perimeters[cell, index], ls='', marker='.', color='orange')

    fig.set_dpi(500)
    fig.tight_layout(pad=0.12, h_pad=0.6)

    fig_path = Path().absolute() / 'Fig2'
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / 'Fig2_CellSizePlot.png', dpi=600)
    plt.close('all')
    return 0


# c
def Fig2_CellShape_Plot(movie_path):
    """
    Store plot showing cell shape (solidity, convexity, circularity and aspect ratio) for a cell of a specific movie
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie file is chosen in the fct)
    :return: 0, stores plot in subfolder in current working directory
    """
    movie_path = movie_path.parent.parent.parent / 'Ctrl' / 'TifData' / 'Ctrl.tif'

    solidity, convexity, _, major_axis, minor_axis, circularity = Read_files.read_cell_shape_properties(movie_path)
    aspect_ratio = minor_axis / major_axis
    len_scaling, time_scaling = Read_files.read_scaling(movie_path)

    cell = 0
    mark_frames = [0, 15, 36, 42]

    ################################
    # GENERATE FIRST FIGURE SHOWING TIME SERIES
    ################################
    labels = ['Soli.', 'Conv.', 'Circ.', 'Asp. ratio']
    color = 'cornflowerblue'

    label_fontsize = 8
    fig_size = np.array([1.65, 2.25])
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(4, 1, figure=fig)
    axs = [fig.add_subplot(gs[0, 0]),  # solidity plot
           fig.add_subplot(gs[1, 0]),  # convexity plot
           fig.add_subplot(gs[2, 0]),  # circularity plot
           fig.add_subplot(gs[3, 0])]  # aspect ratio plot
    for i, data in enumerate([solidity, convexity, circularity, aspect_ratio]):
        axs[i].set_ylabel(labels[i], fontsize=label_fontsize, fontfamily='Arial', labelpad=2)
        axs[i].tick_params(axis='y', size=2, width=1, labelsize=label_fontsize, pad=1)
        axs[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axs[i].yaxis.set_major_locator(ticker.MaxNLocator(2))
        if i < 3:
            axs[i].xaxis.set_major_formatter(ticker.NullFormatter())
            axs[i].tick_params(axis='x', size=2, width=1, pad=1)

        axs[i].plot(time_scaling, data[cell, :], color=color, lw=1.25)

        # mark the frames of the movie snapshots in the plot
        for index in mark_frames:
            axs[i].plot(time_scaling[index], data[cell, index], ls='', marker='.', color='orange')

    axs[-1].tick_params(axis='x', size=2, width=1, labelsize=label_fontsize, pad=1)
    axs[-1].set_xlabel(r'Time $[s]$', fontsize=label_fontsize, fontfamily='Arial', labelpad=2)
    fig.set_dpi(500)
    fig.tight_layout(pad=0.12)

    fig_path = Path().absolute() / 'Fig2'
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / 'Fig2_CellShapePlot.png', dpi=600)
    plt.close('all')

    return 0


# c
def Fig3_4_CellSnapshots(movie_path, condition):
    """
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie file is chosen in the fct)
    :param condition: str - 'Ctrl', 'LPS', 'YM201636', 'MCSF', 'Isoflurane', 'ExplantDirect', 'ExplantRest':
                whether to plot the snapshots for cells     in vivo treated with different chemicals,
                                                            or cells in explanted tissues
    :return: 0, stores snapshots in subfolder in current working directory
    """
    parent_path = movie_path.parent.parent.parent  # go to 'outermost' directory ('RTM_Data')

    available_conditions = ['Ctrl', 'LPS', 'YM201636', 'MCSF', 'Isoflurane', 'ExplantDirect', 'ExplantRest']
    outline_color = ['cornflowerblue', 'sandybrown', 'plum', 'darkseagreen', 'gold', 'lightseagreen', 'teal']
    movie_name = ['Ctrl_2021-05-21_4.tif', 'LPS_2021-05-28_150min.tif', 'YM201636.tif', 'MCSF100ng_2021-10-20.tif',
                  'Isoflurane_2021-11-08.tif', 'Ctrl_LPSE_2022-12-12.tif', 'Ctrl_LPSE_2022-12-14_IC.tif']
    cell = [3, 9, 13, 7, 7, 6, 11]  # which cell (label) to choose from the movie of the chosen condition
    frame = [17, 44, 0, 8, 249, 0, 30]  # which frame to use for the snapshot
    
    ctrl_movie_name = ['x', 'x', 'x', 'Ctrl_MCSF100ng_2021-10-20.tif', 'Ctrl_Isoflurane_2021-11-08.tif']
    ctrl_frame = [0, 0, 0, 20, 268]  # first 3 numbers are just placeholders
    ctrl_cell = [0, 0, 0, 8, 4]

    try:
        cond_index = available_conditions.index(condition)
    except ValueError:
        print('The only possible inputs for the summary_stat "condition" are ', available_conditions, '!')
        print('The input ', condition, 'was not accepted and no movie snapshote were stored.')
        return 0

    single_snapshot = True
    if condition == 'MCSF' or condition == 'Isoflurane':
        single_snapshot = False

    if single_snapshot:
        # READ IN RELEVANT FILES FOR PLOTTING
        movie_path = parent_path / condition / 'TifData' / movie_name[cond_index]
        gray_arr, _ = Read_files.read_grayscale(movie_path)
        label_arr, _ = Read_files.read_labels(movie_path)
        len_scaling, _ = Read_files.read_scaling(movie_path)
        _, mean_arr_bin, _, _ = Generate_files.generate_mean_shape(movie_path)

        image_to_plot = np.stack([gray_arr[:, :, frame[cond_index]], gray_arr[:, :, frame[cond_index]],
                                  gray_arr[:, :, frame[cond_index]]], axis=2)

        # MARK BORDERS OF FIXED AREA
        mean_arr_bin[label_arr[:, :, frame[cond_index]] != cell[cond_index] + 1] = 0
        fixed_area_border = skimage.segmentation.find_boundaries(mean_arr_bin, mode='inner')
        image_to_plot[fixed_area_border, :] = [255, 255, 255]

        # MARK BORDERS OF WHOLE CELL
        area_border = skimage.segmentation.find_boundaries(label_arr[:, :, frame[cond_index]] == cell[cond_index] + 1, mode='inner')
        image_to_plot[area_border, :] = [i * 255 for i in list(mcol.to_rgb(outline_color[cond_index]))]

        # MEASURES FOR FIG SIZE IN INCH, MICROMETER ETC
        fig_size = np.array([0.7, 1.3])
        width_mum = 50  # wanted width in µm
        fig_size_mum = fig_size * width_mum / fig_size[0]  # now in µm
        allowed_box_size = fig_size_mum * len_scaling

        reg_pro = skimage.measure.regionprops(label_arr[:, :, frame[cond_index]])
        my_bbox = np.array(reg_pro[cell[cond_index]].bbox)
        current_box_size = np.array([my_bbox[3] - my_bbox[1], my_bbox[2] - my_bbox[0]])
        added_size = ((allowed_box_size - current_box_size) / 2).astype(int)

        # CREATE FIGURE AND PLOT
        fig = plt.figure(figsize=fig_size)
        if condition in ['Ctrl', 'LPS', 'YM201636']:
            fig = plt.figure(figsize=fig_size[::-1])  # flip figure so that cells have right format for figure
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot()
        ax.axis('off')

        if condition in ['Ctrl', 'YM201636', 'LPS']:
            small_im_to_plot = np.rot90(image_to_plot[my_bbox[0] - added_size[1]:my_bbox[2] + added_size[1],
                               my_bbox[1] - added_size[0]:my_bbox[3] + added_size[0]])
        else:
            small_im_to_plot = image_to_plot[my_bbox[0] - added_size[1]:my_bbox[2] + added_size[1],
                                        my_bbox[1] - added_size[0]:my_bbox[3] + added_size[0]]
        scaling = small_im_to_plot.shape[0] / fig_size_mum[1]
        ax.imshow(small_im_to_plot)
        len_scalebar = 10 * scaling / fig_size_mum[1]
        ax.plot([5 / fig_size_mum[0], 5 / fig_size_mum[0] + len_scalebar / fig_size[1]],
                [5 / fig_size_mum[1], 5 / fig_size_mum[1]], lw=0.9, transform=ax.transAxes, color='white')
        fig.tight_layout(pad=0)
        fig_path = Path().absolute() / 'Fig3'
        Path(fig_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path / ('Snapshot_' + condition + '.png'), dpi=600)

    elif not single_snapshot:
        # READ IN RELEVANT FILES FOR PLOTTING
        movie_path = parent_path / condition / 'TifData' / movie_name[cond_index]
        gray_arr, _ = Read_files.read_grayscale(movie_path)
        label_arr, _ = Read_files.read_labels(movie_path)
        len_scaling, _ = Read_files.read_scaling(movie_path)
        _, mean_arr_bin, _, _ = Generate_files.generate_mean_shape(movie_path)

        movie_path_ctrl = parent_path / 'Ctrl' / 'TifData' / ctrl_movie_name[cond_index]
        gray_arr_ctrl, _ = Read_files.read_grayscale(movie_path_ctrl)
        label_arr_ctrl, _ = Read_files.read_labels(movie_path_ctrl)
        len_scaling_ctrl, _ = Read_files.read_scaling(movie_path_ctrl)
        _, mean_arr_bin_ctrl, _, _ = Generate_files.generate_mean_shape(movie_path_ctrl)

        # mark boundary for CONDITION cell
        image_to_plot = np.stack([gray_arr[:, :, frame[cond_index]], gray_arr[:, :, frame[cond_index]], gray_arr[:, :, frame[cond_index]]], axis=2)

        mean_arr_bin[label_arr[:, :, frame[cond_index]] != cell[cond_index] + 1] = 0
        fixed_area_border = skimage.segmentation.find_boundaries(mean_arr_bin, mode='inner')
        image_to_plot[fixed_area_border, :] = [255, 255, 255]

        area_border = skimage.segmentation.find_boundaries(label_arr[:, :, frame[cond_index]] == cell[cond_index] + 1, mode='inner')
        image_to_plot[area_border, :] = [i * 255 for i in list(mcol.to_rgb(outline_color[cond_index]))]

        # mark boundary for corresponding CTRL cell
        image_to_plot_ctrl = np.stack(
            [gray_arr_ctrl[:, :, ctrl_frame[cond_index]], gray_arr_ctrl[:, :, ctrl_frame[cond_index]], gray_arr_ctrl[:, :, ctrl_frame[cond_index]]], axis=2)

        mean_arr_bin_ctrl[label_arr_ctrl[:, :, ctrl_frame[cond_index]] != ctrl_cell[cond_index] + 1] = 0
        fixed_area_border = skimage.segmentation.find_boundaries(mean_arr_bin_ctrl, mode='inner')
        image_to_plot_ctrl[fixed_area_border, :] = [255, 255, 255]

        area_border = skimage.segmentation.find_boundaries(label_arr_ctrl[:, :, ctrl_frame[cond_index]] == ctrl_cell[cond_index] + 1, mode='inner')
        image_to_plot_ctrl[area_border, :] = [i * 255 for i in list(mcol.to_rgb(outline_color[0]))]

        fig_size = np.array([0.6, 1.3])
        width_mum = 40  # wanted width in µm
        fig_size_mum = fig_size * width_mum / (fig_size[0])  # now in µm

        allowed_box_size = fig_size_mum * len_scaling
        reg_pro = skimage.measure.regionprops(label_arr[:, :, frame[cond_index]])
        my_bbox = np.array(reg_pro[cell[cond_index]].bbox)
        current_box_size = np.array([my_bbox[3] - my_bbox[1], my_bbox[2] - my_bbox[0]])
        added_size = ((allowed_box_size - current_box_size) / 2).astype(int)

        allowed_box_size_ctrl = fig_size_mum * len_scaling_ctrl
        reg_pro = skimage.measure.regionprops(label_arr_ctrl[:, :, ctrl_frame[cond_index]])
        my_bbox_ctrl = np.array(reg_pro[ctrl_cell[cond_index]].bbox)
        current_box_size_ctrl = np.array([my_bbox_ctrl[3] - my_bbox_ctrl[1], my_bbox_ctrl[2] - my_bbox_ctrl[0]])
        added_size_ctrl = np.round(((allowed_box_size_ctrl - current_box_size_ctrl) / 2)).astype(int)

        fig = plt.figure(figsize=fig_size[::-1] * np.array([1, 2]))
        ax_ctrl = fig.add_subplot(211)
        ax = fig.add_subplot(212)
        fig.patch.set_facecolor('black')
        ax_ctrl.axis('off')

        rot_im = np.rot90(
            image_to_plot_ctrl[my_bbox_ctrl[0] - added_size_ctrl[1]:my_bbox_ctrl[2] + added_size_ctrl[1],
            my_bbox_ctrl[1] - added_size_ctrl[0]:my_bbox_ctrl[3] + added_size_ctrl[0]])
        ax_ctrl.imshow(rot_im)
        scaling = rot_im.shape[0] / fig_size_mum[1]
        len_scalebar = 10 * scaling / fig_size_mum[1]
        ax_ctrl.plot([5 / fig_size_mum[0], 5 / fig_size_mum[0] + len_scalebar / fig_size[1]],
                     [5 / fig_size_mum[1], 5 / fig_size_mum[1]], lw=0.9, transform=ax.transAxes, color='white')

        ax.axis('off')
        ax.imshow(np.rot90(image_to_plot[my_bbox[0] - added_size[1]:my_bbox[2] + added_size[1],
                           my_bbox[1] - added_size[0]:my_bbox[3] + added_size[0]]))
        ax.plot([5 / fig_size_mum[0], 5 / fig_size_mum[0] + len_scalebar / fig_size[1]],
                [5 / fig_size_mum[1], 5 / fig_size_mum[1]], lw=0.9, transform=ax.transAxes, color='white')
        # scale bar (length 20% of 50 µm => 10 µm)

        fig.tight_layout(pad=0)
        fig_path = Path().absolute() / 'Fig3'
        Path(fig_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path / ('Snapshot_' + condition + '.png'), dpi=600)

    return 0


def AuxilliaryFct_create_boxplot(data, x_labels, y_label='', colmap=None, population_comp=True):
    """
    IDEA: function takes features/ properties of cells of different conditions as iinput,
    plots boxplot and returns figure object
    :param data: list of numpy arrays. 1 numpy array stores cells of one condition.
    :param x_labels: list of strings. which labels to plot on the x-axis (i.e. names of the conditions)
    :param y_label: string. which cellular feature (e.g. mean area) is being plotted to put on y-axis
    :param colmap: colormap,
    :param population_comp: bool, whether whole cell populations (e.g. ctrl vs LPS) are compared, or
                    whether the cell feature of e.g. MCSF is subtracted from the resp Ctrl cell,
                    statistical significance is then calculated against 0
    :return: figure object
    """
    if colmap is None:
        colmap = plt.get_cmap('Accent')

    label_fontsize = 8
    plt.rcParams["font.family"] = "Arial"
    # plt.style.use('dark_background')
    fig_size = np.array([2.1, 1.7])  # for main figs
    fig_size = np.array([1.8, 1.5])  # for suppl figs
    if not population_comp:
        fig_size = np.array([2.1, 1.73])  # for main figs
        fig_size = np.array([1.8, 1.5])  # for suppl figs
    fig = plt.figure(figsize=fig_size)
    fig.set_dpi(600)

    h = [Size.Fixed(0.48), Size.Fixed(1.5), Size.Fixed(0.05)]  # for main figures
    h = [Size.Fixed(0.51), Size.Fixed(1.275), Size.Fixed(1)]  # for suppl figs

    if not population_comp:
        h = [Size.Fixed(0.48), Size.Fixed(1.5), Size.Fixed(0.05)]
        h = [Size.Fixed(0.51), Size.Fixed(1.275), Size.Fixed(1)]  # for suppl figs
    v = [Size.Fixed(0.3), Size.Fixed(1.2), Size.Fixed(0.14)]
    v = [Size.Fixed(0.31), Size.Fixed(1), Size.Fixed(0.16)]  # for suppl figs

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))
    # ax = fig.add_subplot()
    ax.set_ylabel(y_label, fontsize=label_fontsize, fontfamily='Arial', labelpad=1.4)
    ax.tick_params(axis='y', size=2, width=1, labelsize=label_fontsize, pad=1)
    cloud_extent = [1.0, 1.3]
    # VIOLIN PLOT
    vio = ax.violinplot(data, showextrema=False, widths=0.8)
    for pc, color in zip(vio['bodies'], [colmap(i) for i in range(len(x_labels))]):
        pc.set_facecolor(color)
    cloud_extent = [0.8, 1.2]
    # SCATTER POINT CLOUD
    for cond in range(0, len(x_labels)):
        # PLOT DATA POINTS AS CLOUD WITHIN THE BOX PLOT
        if population_comp:
            ax.plot(np.linspace(cond + cloud_extent[0], cond + cloud_extent[1], len(data[cond])), data[cond],
                    alpha=0.85, color=colmap(cond), ls='', marker='.', markersize=2)
        else:
            ax.plot(np.linspace(cond + cloud_extent[0], cond + cloud_extent[1], len(data[cond])), data[cond],
                    alpha=0.85, color=colmap(cond), ls='', marker='.', markersize=4)  # markersize=6

        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        print(y_label, x_labels[cond], 'mean:', np.mean(data[cond]), 'std:', np.std(data[cond]))

    bp = ax.boxplot(data, labels=x_labels, sym='', widths=0.4, showmeans=True, meanline=True,
                    meanprops=dict(ls='-', color='black'), medianprops=dict(ls=''))
    ax.tick_params(axis='x', size=2, width=1, labelsize=label_fontsize, pad=1.1, labelrotation=20)

    tick_dist = ax.get_yticks()[-1] - ax.get_yticks()[-2]
    if 1 > tick_dist > 0.01:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    elif tick_dist < 0.01:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax.yaxis.offsetText.set_fontsize(label_fontsize)
    # ax.yaxis.set_major_locator(ticker.LinearLocator(4))

    # CALCULATE STATISTICAL SIGNIFICANCE (always compare to data[0])
    if population_comp:
        significances = np.ones(len(x_labels) - 1)  # to store all p-values in!
        for cond in range(1, len(x_labels)):
            stats, pval = scipy.stats.ttest_ind(data[0], data[cond], equal_var=False, permutations=2000000)
            significances[cond - 1] = pval
    else:
        significances = np.ones(len(x_labels))  # to store all p-values in!
        for cond in range(0, len(x_labels)):
            # one-sided!
            if np.mean(data[cond]) < 0:
                res = scipy.stats.ttest_1samp(data[cond], 0, nan_policy='omit', alternative='less')
                significances[cond] = res.pvalue
            elif np.mean(data[cond]) > 0:
                res = scipy.stats.ttest_1samp(data[cond], 0, nan_policy='omit', alternative='greater')
                significances[cond] = res.pvalue
    amount_of_sign_results = len(np.where(significances < 0.05)[0])  # find out how many significant results we have
    print('significances:', significances)

    max_whisker = max([bp['caps'][i].get_ydata()[0] for i in range(len(x_labels) * 2)])
    usable_y_space = ax.get_ylim()[1] - max_whisker
    needed_y_space = (amount_of_sign_results * 0.082 + 0.04) * (ax.get_ylim()[1] - ax.get_ylim()[0])
    if usable_y_space < needed_y_space:
        ax.set_ylim(ax.get_ylim()[0], max_whisker + needed_y_space)

    for cond in range(0, len(x_labels)):
        # PLOT STATISTICAL SIGNIFICANCES!
        if population_comp:
            if cond > 0 and significances[cond - 1] < 0.05:
                x1, x2 = 1, cond + 1  # first column and resp condition
                # tells us 'die wievielte' significance line we draw
                number = np.where(np.where(significances < 0.05)[0] == cond - 1)[0][0]
                y = (1 - amount_of_sign_results * 0.095) + 0.095 * number
                col = 'black'
                ax.plot([x1, x2], [y, y], lw=0.75, c=col, transform=trans)
                if significances[cond - 1] < 0.001:
                    ax.text((x1 + x2) * .5, y - 0.06, '***', ha='center', va='bottom', color=col, transform=trans,
                            fontsize=label_fontsize + 2)
                elif significances[cond - 1] < 0.01:
                    ax.text((x1 + x2) * .5, y - 0.06, '**', ha='center', va='bottom', color=col, transform=trans,
                            fontsize=label_fontsize + 2)
                else:
                    ax.text((x1 + x2) * .5, y - 0.06, '*', ha='center', va='bottom', color=col, transform=trans,
                            fontsize=label_fontsize + 2)
                # text = 'p=' + str(np.round(significances[cond - 1], 4))
                # ax.text((x1 + x2) * .5, y, text, ha='center', va='bottom', color=col, transform=trans,
                #  fontsize=label_fontsize - 2)
        else:
            if significances[cond] < 0.05:
                y = 1 - 0.15
                col = 'black'
                if significances[cond - 1] < 0.001:
                    ax.text(cond + 1, y, '***', ha='center', va='bottom', color=col, transform=trans,
                            fontsize=label_fontsize + 2)
                elif significances[cond - 1] < 0.01:
                    ax.text(cond + 1, y, '**', ha='center', va='bottom', color=col, transform=trans,
                            fontsize=label_fontsize + 2)
                else:
                    ax.text(cond + 1, y, '*', ha='center', va='bottom', color=col, transform=trans,
                            fontsize=label_fontsize + 2)
                # text = 'p=' + str(np.round(significances[cond], 4))
                # ax.text(cond + 1, y, text, ha='center', va='bottom', color=col, transform=trans,
                #      fontsize=label_fontsize - 2)

    if not population_comp:
        ax.axhline(0, alpha=0.9, color='black', ls=':', lw=1)
    # ax.axhline(np.mean(data[0]), alpha=0.9, color='black', ls=':', lw=0.75)
    # fig.tight_layout(pad=0.1)

    return fig


# c
def Fig3_4_CellSize_boxplot(movie_path, summary_stat='mean', imaging_condition='in_vivo'):
    """
    Creates boxplot (with underlying violin plot) of cell size ([whole-, fixed-, mobile] area and perimeter) summarized
    by 'summary_stat' - compares the available cell populations subjected to chemical/ physical perturbations against
    the in vivo control population
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie files are chosen in the fct)
    :param summary_stat: str - 'mean', 'std' or 'trend': how to summarize the data over time to a single data point per cell
    :param imaging_condition: str - 'in_vivo' or 'Explant'
    :return: 0, stores plots in subfolder in current working directory
    """

    # depending on imaging condition, choose the correct (chemical) condition and corresponding colors
    if imaging_condition == 'in_vivo':
        conditions = ['Ctrl', 'LPS', 'MCSF', 'Isoflurane', 'YM201636']
        colors = ['lightskyblue', 'sandybrown', 'darkseagreen', 'gold', 'plum']
        colmap = mcol.LinearSegmentedColormap.from_list('my_map', colors, N=len(colors))
        x_labels = ['Ctrl', 'LPS', 'M-CSF', 'Iso', 'YM']
    elif imaging_condition == 'Explant':
        conditions = ['Ctrl', 'ExplantDirect', 'ExplantRest']
        colors = ['lightskyblue', 'lightseagreen', 'teal']
        colmap = mcol.LinearSegmentedColormap.from_list('my_map', colors, N=len(colors))
        x_labels = ['$\it{in}$ $\it{vivo}$', 'Expl direct', 'Expl rest']
    else:
        print('The only accepted input for "imaging_condition" are "in_vivo" or "Explant", but not ', imaging_condition)
        imaging_condition = 'in_vivo'
        conditions = ['Ctrl', 'LPS', 'MCSF', 'Isoflurane', 'YM201636']
        colors = ['lightskyblue', 'sandybrown', 'darkseagreen', 'gold', 'plum']
        colmap = mcol.LinearSegmentedColormap.from_list('my_map', colors, N=len(colors))
        x_labels = ['Ctrl', 'LPS', 'M-CSF', 'Iso', 'YM']
        print('Here, "in_vivo" is now used.')

    if summary_stat not in ['mean', 'std', 'trend']:
        print('The only accepted input for "summary_stat" are "mean", "std" or "trend", but not ', summary_stat)
        print('Here, "mean" is now used.')
        summary_stat = 'mean'

    all_data = [[None for x in range(len(conditions))] for y in range(4)]  # area, perimeter, mobile and fixed area
    parent_path = movie_path.parent.parent.parent  # go to 'outermost' directory ('RTM_Data')

    for cond_index, condition in enumerate(conditions):
        # all movie paths for one condition:
        p = (parent_path / condition / movie_path.parent.name).glob('*.tif')
        all_movie_paths = [x for x in p if x.is_file()]

        # for storing the data of ONE condition
        area_data, perimeter_data = [None] * len(all_movie_paths), [None] * len(all_movie_paths)
        mobile_area_data, fixed_area_data = [None] * len(all_movie_paths), [None] * len(all_movie_paths)

        for movie_index, movie_path in enumerate(all_movie_paths):
            # READ IN conditions, time scaling, handpicked cell excel file
            areas, fixed_area, mobile_areas, perimeters, pixel_area, intensity_areas = \
                Read_files.read_cell_size_properties(movie_path)
            _, time_scaling = Read_files.read_scaling(movie_path)

            # find cell index/label of handpicked cells
            handpicked_cells = Read_files.read_handpicked_excel_file(movie_path)
            hp_cells_movie = handpicked_cells[np.where(handpicked_cells[:, 0] == movie_path.name)[0], :]
            hp_cells_movie = hp_cells_movie[hp_cells_movie[:, 1].argsort()]  # sort by cell number
            hp_cell_indices = hp_cells_movie[hp_cells_movie[:, 2] == 'n'][:, 1].astype(int)
            if len(hp_cell_indices) == 0:
                continue  # no good cells found, I can skip this movie and do not/ cannot plot anything

            # storing cell size properties of the respective movie
            if summary_stat == 'mean':
                area_data[movie_index] = np.mean(areas[hp_cell_indices, :], axis=1)
                perimeter_data[movie_index] = np.mean(perimeters[hp_cell_indices, :], axis=1)
                mobile_area_data[movie_index] = np.mean(mobile_areas[hp_cell_indices, :], axis=1)
                fixed_area_data[movie_index] = fixed_area[hp_cell_indices]
            elif summary_stat == 'std':
                area_data[movie_index] = np.std(areas[hp_cell_indices, :], axis=1)
                perimeter_data[movie_index] = np.std(perimeters[hp_cell_indices, :], axis=1)
            elif summary_stat == 'trend':
                fits = np.zeros((2, len(hp_cell_indices)))
                for cell_index, cell in enumerate(hp_cell_indices):
                    # linear regression for area
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), areas[cell, :])
                    fits[0, cell_index] = model.coef_[0]

                    # linear regression for perimeter
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), perimeters[cell, :])
                    fits[1, cell_index] = model.coef_[0]

                area_data[movie_index] = fits[0, :]
                perimeter_data[movie_index] = fits[1, :]

        # remove last position(s) where no good cells were found in the movie
        if len(all_movie_paths) != movie_index:
            area_data = [i for i in area_data if i is not None]
            perimeter_data = [i for i in perimeter_data if i is not None]
            mobile_area_data = [i for i in mobile_area_data if i is not None]
            fixed_area_data = [i for i in fixed_area_data if i is not None]

            # area_data = area_data[:-(len(all_movie_paths) - movie_index)]
            # perimeter_data = perimeter_data[:-(len(all_movie_paths) - movie_index)]
            # mobile_area_data = mobile_area_data[:-(len(all_movie_paths) - movie_index)]
            # fixed_area_data = fixed_area_data[:-(len(all_movie_paths) - movie_index)]

        # store property data from movie of resp. condition in correct place in all_data
        all_data[0][cond_index] = np.concatenate(area_data)
        all_data[1][cond_index] = np.concatenate(perimeter_data)
        if summary_stat == 'mean':
            all_data[2][cond_index] = np.concatenate(mobile_area_data)
            all_data[3][cond_index] = np.concatenate(fixed_area_data)

    # for saving plots
    fig_path = Path().absolute() / 'Fig3-4_CellSizeBoxplots'
    Path(fig_path).mkdir(parents=True, exist_ok=True)

    if summary_stat == 'mean':
        # MAKE AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labels, y_label=r'Mean area $[\mu m^2]$', colmap=colmap)
        fig.savefig(fig_path / ('CellSize_mean_Area_' + imaging_condition + '.png'), dpi=600)
        plt.close(fig)
        # MAKE PERIMETER BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labels, y_label=r'Mean perim. $[\mu m]$', colmap=colmap)
        fig.savefig(fig_path / ('CellSize_mean_Perimeter_' + imaging_condition + '.png'), dpi=600)
        plt.close(fig)
        # MAKE MOBILE AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[2], x_labels=x_labels, y_label=r'Mean mob. area $[\mu m^2]$', colmap=colmap)
        fig.savefig(fig_path / ('CellSize_mean_MobileArea_' + imaging_condition + '.png'), dpi=600)
        plt.close(fig)
        # MAKE FIXED AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[3], x_labels=x_labels, y_label=r'Fixed area $[\mu m^2]$', colmap=colmap)
        fig.savefig(fig_path / ('CellSize_mean_FixedArea_' + imaging_condition + '.png'), dpi=600)
        plt.close(fig)
    elif summary_stat == 'std':
        # MAKE AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labels, y_label=r'Std of area $[\mu m^2]$', colmap=colmap)
        fig.savefig(fig_path / ('CellSize_std_Area_' + imaging_condition + '.png'), dpi=600)
        plt.close(fig)
        # MAKE PERIMETER BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labels, y_label=r'Std of perim. $[\mu m]$', colmap=colmap)
        fig.savefig(fig_path / ('CellSize_std_Perimeter_' + imaging_condition + '.png'), dpi=600)
        plt.close(fig)
    elif summary_stat == 'trend':
        # MAKE AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labels, y_label=r'Trend of area $[\mu m^2/s]$', colmap=colmap)
        fig.savefig(fig_path / ('CellSize_trend_Area_' + imaging_condition + '.png'), dpi=600)
        plt.close(fig)
        # MAKE PERIMETER BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labels, y_label=r'Trend of perim. $[\mu m/s]$', colmap=colmap)
        fig.savefig(fig_path / ('CellSize_trend_Perimeter_' + imaging_condition + '.png'), dpi=600)
        plt.close(fig)

    return 0


# c
def Fig3_CellSize_direct_boxplot(movie_path, summary_stat='mean'):
    """
    Creates boxplot (with underlying violin plot) of cell size ([whole-, fixed-, mobile] area and perimeter) summarized
    by 'summary_stat' - compares cells under M-CSF and Isoflurane against the same cells of the control population
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie files are chosen in the fct)
    :param summary_stat: str - 'mean', 'std' or 'trend': how to summarize the data over time to a single data point per cell
    :return: 0, stores plots in subfolder in current working directory
    """

    conditions = ['MCSF', 'Isoflurane']
    colors = ['darkseagreen', 'gold']
    colmap = mcol.LinearSegmentedColormap.from_list('my_map', colors, N=len(colors))

    if summary_stat not in ['mean', 'std', 'trend']:
        print('The only accepted input for "summary_stat" are "mean", "std" or "trend", but not ', summary_stat)
        print('Here, "mean" is now used.')
        summary_stat = 'mean'

    all_data = [[None for x in range(len(conditions))] for y in range(4)]  # area, perimeter, mobile and fixed area
    parent_path = movie_path.parent.parent.parent  # go to 'outermost' directory ('RTM_Data')

    for cond_index, condition in enumerate(conditions):
        # all movie paths for one condition:
        p = (parent_path / condition / movie_path.parent.name).glob('*.tif')
        all_movie_paths = [x for x in p if x.is_file()]

        # for storing the data of ONE condition
        area_data, perimeter_data = [[] for i in range(len(all_movie_paths))], [[] for i in range(len(all_movie_paths))]
        mobile_area_data, fixed_area_data = [[] for i in range(len(all_movie_paths))], [[] for i in
                                                                                        range(len(all_movie_paths))]
        movie_index = 0
        for movie_path in all_movie_paths:

            # first check if the corresponding control movie exists!
            ctrl_movie_path = 'Ctrl_' + movie_path.name
            ctrl_movie_path = parent_path / 'Ctrl' / movie_path.parent.name / ctrl_movie_path
            if not ctrl_movie_path.is_file():
                continue  # there is no corresponding ctrl movie, so we skip this condition movie!

            # find cell index/label of handpicked cells FOR CTRL
            handpicked_cells = Read_files.read_handpicked_excel_file(ctrl_movie_path)
            hp_cells_movie = handpicked_cells[np.where(handpicked_cells[:, 0] == ctrl_movie_path.name)[0], :]
            hp_cells_movie = hp_cells_movie[hp_cells_movie[:, 1].argsort()]  # sort by cell number
            ctrl_hp_cell_indices = hp_cells_movie[hp_cells_movie[:, 2] == 'n'][:, 1].astype(int)
            if len(ctrl_hp_cell_indices) == 0:
                continue  # no good cells found, I can skip this movie and do not/ cannot plot anything

            # find cell index/label of handpicked cells FOR CONDITION
            handpicked_cells = Read_files.read_handpicked_excel_file(movie_path)
            hp_cells_movie = handpicked_cells[np.where(handpicked_cells[:, 0] == movie_path.name)[0], :]
            hp_cells_movie = hp_cells_movie[hp_cells_movie[:, 1].argsort()]  # sort by cell number
            hp_cell_indices = hp_cells_movie[hp_cells_movie[:, 2] == 'n'][:, 1].astype(int)
            if len(hp_cell_indices) == 0:
                continue  # no good cells found, I can skip this movie and do not/ cannot plot anything

            # read in properties of CONDITION FILE
            areas, fixed_area, mobile_areas, perimeters, _, _ = Read_files.read_cell_size_properties(movie_path)
            _, time_scaling = Read_files.read_scaling(movie_path)
            amount_of_cells = areas.shape[0]

            # read in properties of CTRL FILE
            areas_c, fixed_area_c, mobile_areas_c, perimeters_c, _, _ = \
                Read_files.read_cell_size_properties(ctrl_movie_path)
            _, time_scaling_c = Read_files.read_scaling(ctrl_movie_path)

            # SORT CELLS TOGETHER AND SUBTRACT VALUES
            for cell in range(amount_of_cells):
                if cell in hp_cell_indices:  # condition cell is considered normal
                    ctrl_cell_nr = int(hp_cells_movie[cell, 4])  # corresponding ctrl cell
                    if ctrl_cell_nr in ctrl_hp_cell_indices:  # corresponding ctrl cell is considered normal
                        # print(movie_path.name, ctrl_movie_path.name)
                        # print('condition cell', cell, 'is normal, and ctrl cell', ctrl_cell_nr, 'is also normal')
                        if summary_stat == 'mean':
                            area_data[movie_index].append(np.mean(areas[cell, :]) - np.mean(areas_c[ctrl_cell_nr, :]))
                            perimeter_data[movie_index].append(np.mean(perimeters[cell, :]) - \
                                                               np.mean(perimeters_c[ctrl_cell_nr, :]))
                            mobile_area_data[movie_index].append(np.mean(mobile_areas[cell, :]) - \
                                                                 np.mean(mobile_areas_c[ctrl_cell_nr, :]))
                            fixed_area_data[movie_index].append(fixed_area[cell] - fixed_area_c[ctrl_cell_nr])
                        elif summary_stat == 'std':
                            area_data[movie_index].append(np.std(areas[cell, :]) - np.std(areas_c[ctrl_cell_nr, :]))
                            perimeter_data[movie_index].append(np.std(perimeters[cell, :]) - \
                                                               np.std(perimeters_c[ctrl_cell_nr, :]))
                        elif summary_stat == 'trend':
                            # linear regression for area
                            model = LinearRegression().fit(time_scaling.reshape((-1, 1)), areas[cell, :])
                            model_c = LinearRegression().fit(time_scaling_c.reshape((-1, 1)), areas_c[ctrl_cell_nr, :])
                            area_data[movie_index].append(model.coef_[0] - model_c.coef_[0])

                            # linear regression for perimeter
                            model = LinearRegression().fit(time_scaling.reshape((-1, 1)), perimeters[cell, :])
                            model_c = LinearRegression().fit(time_scaling_c.reshape((-1, 1)),
                                                             perimeters_c[ctrl_cell_nr, :])
                            perimeter_data[movie_index].append(model.coef_[0] - model_c.coef_[0])

            movie_index += 1

        # remove last position(s) where no good cells were found in the movie
        if len(all_movie_paths) != movie_index:
            area_data = [i for i in area_data if i is not None]
            perimeter_data = [i for i in perimeter_data if i is not None]
            mobile_area_data = [i for i in mobile_area_data if i is not None]
            fixed_area_data = [i for i in fixed_area_data if i is not None]

            # area_data = area_data[:-(len(all_movie_paths) - movie_index)]
            # perimeter_data = perimeter_data[:-(len(all_movie_paths) - movie_index)]
            # mobile_area_data = mobile_area_data[:-(len(all_movie_paths) - movie_index)]
            # fixed_area_data = fixed_area_data[:-(len(all_movie_paths) - movie_index)]

        # store property data from movie of resp. condition in correct place in all_data
        all_data[0][cond_index] = np.concatenate(area_data)
        all_data[1][cond_index] = np.concatenate(perimeter_data)
        if summary_stat == 'mean':
            all_data[2][cond_index] = np.concatenate(mobile_area_data)
            all_data[3][cond_index] = np.concatenate(fixed_area_data)

    x_labs = ['M-CSF', 'Iso']
    my_dpi = 500
    # for saving plots
    fig_path = Path().absolute() / 'Fig3-4_CellSizeBoxplots'
    Path(fig_path).mkdir(parents=True, exist_ok=True)

    if summary_stat == 'mean':
        # MAKE AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labs, y_label=r'$\Delta$(Mean area) $[\mu m^2]$', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellSize_mean_direct_Area.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE PERIMETER BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labs, y_label=r'$\Delta$(Mean perim.) $[\mu m]$', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellSize_mean_direct_Perimeter.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE MOBILE AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[2], x_labels=x_labs, y_label=r'$\Delta$(Mean mob. area) $[\mu m^2]$',
                               colmap=colmap, population_comp=False)
        fig.savefig(fig_path / 'CellSize_mean_direct_MobileArea.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE FIXED AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[3], x_labels=x_labs, y_label=r'$\Delta$(Fixed area) $[\mu m^2]$', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellSize_mean_direct_FixedArea.png', dpi=my_dpi)
        plt.close(fig)
    elif summary_stat == 'std':
        # MAKE AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labs, y_label=r'$\Delta$(Std of area) $[\mu m^2]$',
                               colmap=colmap, population_comp=False)
        fig.savefig(fig_path / 'CellSize_std_direct_Area.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE PERIMETER BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labs, y_label=r'$\Delta$(Std of perim.) $[\mu m]$',
                               colmap=colmap, population_comp=False)
        fig.savefig(fig_path / 'CellSize_std_direct_Perimeter.png', dpi=my_dpi)
        plt.close(fig)
    elif summary_stat == 'trend':
        # MAKE AREA BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labs, y_label=r'$\Delta$(Trend of area) $[\mu m^2/s]$',
                               colmap=colmap, population_comp=False)
        fig.savefig(fig_path / 'CellSize_trend_direct_Area.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE PERIMETER BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labs, y_label=r'$\Delta$(Trend of perim.) $[\mu m/s]$',
                               colmap=colmap, population_comp=False)
        fig.savefig(fig_path / 'CellSize_trend_direct_Perimeter.png', dpi=my_dpi)
        plt.close(fig)

    return 0


# c
def Fig3_4_CellShape_boxplot(movie_path, summary_stat='mean', imaging_condition='in_vivo'):
    """
    Creates boxplot (with underlying violin plot) of cell shape (solidity, convexity, aspect rati0, circularity)
    summarized by 'summary_stat' - compares the available cell populations subjected to chemical/ physical
    perturbations against the in vivo control population
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie files are chosen in the fct)
    :param summary_stat: str - 'mean', 'std' or 'trend': how to summarize the data over time to a single data point / cell
    :param imaging_condition: str - 'in_vivo' or 'Explant'
    :return: 0, stores plot in subfolder in current working directory
    """
    if imaging_condition == 'in_vivo':
        conditions = ['Ctrl', 'LPS', 'MCSF', 'Isoflurane', 'YM201636']
        colors = ['lightskyblue', 'sandybrown', 'darkseagreen', 'gold', 'plum']
        colmap = mcol.LinearSegmentedColormap.from_list('my_map', colors, N=len(colors))
        x_labs = ['Ctrl', 'LPS', 'M-CSF', 'Iso', 'YM']

    elif imaging_condition == 'Explant':
        conditions = ['Ctrl', 'ExplantDirect', 'ExplantRest']
        colors = ['lightskyblue', 'lightseagreen', 'teal']
        colmap = mcol.LinearSegmentedColormap.from_list('my_map', colors, N=len(colors))
        x_labs = ['$\it{in}$ $\it{vivo}$', 'Expl direct', 'Expl rest']

    if summary_stat not in ['mean', 'std', 'trend']:
        print('The only accepted input for summary_stat are "mean", "std" or "trend", but not ', summary_stat)
        print('Here, "mean" is now used.')
        summary_stat = 'mean'

    all_data = [[None for x in range(len(conditions))] for y in range(4)]  # area, perimeter, mobile and fixed area
    parent_path = movie_path.parent.parent.parent  # go to 'outermost' directory ('RTM_Data')

    for cond_index, condition in enumerate(conditions):
        # all movie paths for one condition:
        p = (parent_path / condition / movie_path.parent.name).glob('*.tif')
        all_movie_paths = [x for x in p if x.is_file()]

        # for storing the data of ONE condition
        solidity_data, convexity_data = [None] * len(all_movie_paths), [None] * len(all_movie_paths)
        aspect_ratio_data, circularity_data = [None] * len(all_movie_paths), [None] * len(all_movie_paths)

        movie_index = 0
        for movie_path in all_movie_paths:
            # read in conditions, time scaling, handpicked cell excel file
            solidity, convexity, _, major_axis, minor_axis, circularity = \
                Read_files.read_cell_shape_properties(movie_path)
            aspect_ratio = minor_axis / major_axis
            _, time_scaling = Read_files.read_scaling(movie_path)

            # find cell index/label of handpicked cells
            handpicked_cells = Read_files.read_handpicked_excel_file(movie_path)
            hp_cells_movie = handpicked_cells[np.where(handpicked_cells[:, 0] == movie_path.name)[0], :]
            hp_cells_movie = hp_cells_movie[hp_cells_movie[:, 1].argsort()]  # sort by cell number
            hp_cell_indices = hp_cells_movie[hp_cells_movie[:, 2] == 'n'][:, 1].astype(int)
            if len(hp_cell_indices) == 0:
                continue  # no good cells found, I can skip this movie and do not/ cannot plot anything

            # storing cell size properties of the respective movie
            if summary_stat == 'mean':
                solidity_data[movie_index] = np.mean(solidity[hp_cell_indices, :], axis=1)
                convexity_data[movie_index] = np.mean(convexity[hp_cell_indices, :], axis=1)
                aspect_ratio_data[movie_index] = np.mean(aspect_ratio[hp_cell_indices, :], axis=1)
                circularity_data[movie_index] = np.mean(circularity[hp_cell_indices, :], axis=1)
            elif summary_stat == 'std':
                solidity_data[movie_index] = np.std(solidity[hp_cell_indices, :], axis=1)
                convexity_data[movie_index] = np.std(convexity[hp_cell_indices, :], axis=1)
                aspect_ratio_data[movie_index] = np.std(aspect_ratio[hp_cell_indices, :], axis=1)
                circularity_data[movie_index] = np.std(circularity[hp_cell_indices, :], axis=1)
            elif summary_stat == 'trend':
                fits = np.zeros((4, len(hp_cell_indices)))
                for cell_index, cell in enumerate(hp_cell_indices):  # linear regressions for all parameters
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), solidity[cell, :])
                    fits[0, cell_index] = model.coef_[0]
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), convexity[cell, :])
                    fits[1, cell_index] = model.coef_[0]
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), aspect_ratio[cell, :])
                    fits[2, cell_index] = model.coef_[0]
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), circularity[cell, :])
                    fits[3, cell_index] = model.coef_[0]

                solidity_data[movie_index] = fits[0, :]
                convexity_data[movie_index] = fits[1, :]
                aspect_ratio_data[movie_index] = fits[2, :]
                circularity_data[movie_index] = fits[3, :]
            movie_index += 1

        # remove last position(s) where no good cells were found in the movie
        if len(all_movie_paths) != movie_index:
            solidity_data = [i for i in solidity_data if i is not None]
            convexity_data = [i for i in convexity_data if i is not None]
            aspect_ratio_data = [i for i in aspect_ratio_data if i is not None]
            circularity_data = [i for i in circularity_data if i is not None]
            # solidity_data = solidity_data[:-(len(all_movie_paths) - movie_index)]
            # convexity_data = convexity_data[:-(len(all_movie_paths) - movie_index)]
            # aspect_ratio_data = aspect_ratio_data[:-(len(all_movie_paths) - movie_index)]
            # circularity_data = circularity_data[:-(len(all_movie_paths) - movie_index)]

        # store property data from movie of resp. condition in correct place in all_data
        all_data[0][cond_index] = np.concatenate(solidity_data)
        all_data[1][cond_index] = np.concatenate(convexity_data)
        all_data[2][cond_index] = np.concatenate(aspect_ratio_data)
        all_data[3][cond_index] = np.concatenate(circularity_data)

    my_dpi = 500
    # for storing plots
    fig_path = Path().absolute() / 'Fig3-4_CellShapeBoxplots'
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    if summary_stat == 'mean':
        # MAKE SOLIDITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labs, y_label='Mean soli.', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_mean_Solidity_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
        # MAKE CONVEXITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labs, y_label='Mean conv.', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_mean_Convexity_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
        # MAKE ASPECT RATIO BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[2], x_labels=x_labs, y_label='Mean asp. ratio', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_mean_AspectRatio_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
        # MAKE CIRCULARITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[3], x_labels=x_labs, y_label='Mean circ.', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_mean_Circularity_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
    elif summary_stat == 'std':
        # MAKE SOLIDITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labs, y_label='Std of soli.', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_std_Solidity_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
        # MAKE CONVEXITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labs, y_label='Std of conv.', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_std_Convexity_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
        # MAKE ASPECT RATIO BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[2], x_labels=x_labs, y_label='Std of asp. ratio', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_std_AspectRatio_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
        # MAKE CIRCULARITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[3], x_labels=x_labs, y_label='Std of circ.', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_std_Circularity_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
    elif summary_stat == 'trend':
        # MAKE SOLIDITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labs, y_label=r'Trend of soli. $[1/s]$', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_trend_Solidity_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
        # MAKE CONVEXITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labs, y_label=r'Trend of conv. $[1/s]$', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_trend_Convexity_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
        # MAKE ASPECT RATIO BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[2], x_labels=x_labs, y_label=r'Trend of asp. ratio $[1/s]$', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_trend_AspectRatio_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)
        # MAKE CIRCULARITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[3], x_labels=x_labs, y_label=r'Trend of circ. $[1/s]$', colmap=colmap)
        fig.savefig(fig_path / ('CellShape_trend_Circularity_' + imaging_condition + '.png'), dpi=my_dpi)
        plt.close(fig)

    return 0


# c
def Fig3_CellShape_direct_boxplot(movie_path, summary_stat='mean'):
    """
       Creates boxplot (with underlying violin plot) of cell shape (solidity, convexity, aspect ratio, circularity)
       summarized by 'summary_stat' - compares cells under M-CSF and Isoflurane against the same cells of the control
       population
       :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie files are chosen in the fct)
       :param summary_stat: str - 'mean', 'std' or 'trend': how to summarize the data over time to a single data point per cell
       :return: 0, stores plots in subfolder of current working directory
       """

    conditions = ['MCSF', 'Isoflurane']

    colors = ['darkseagreen', 'gold']
    colmap = mcol.LinearSegmentedColormap.from_list('my_map', colors, N=len(colors))

    if summary_stat not in ['mean', 'std', 'trend']:
        print('The only accepted input for summary_stat are "mean", "std" or "trend", but not', summary_stat)
        print('Instead of invalid inputs, " mean"is used.')
        summary_stat = 'mean'

    all_data = [[None for x in range(len(conditions))] for y in range(4)]  # area, perimeter, mobile and fixed area
    parent_path = movie_path.parent.parent.parent  # go to 'outermost' directory ('RTM_Data')

    for cond_index, condition in enumerate(conditions):
        # all movie paths for one condition:
        p = (parent_path / condition / movie_path.parent.name).glob('*.tif')
        all_movie_paths = [x for x in p if x.is_file()]

        # for storing the data of ONE condition
        solidity_data, convexity_data = [[] for i in range(len(all_movie_paths))], [[] for i in
                                                                                    range(len(all_movie_paths))]
        aspect_ratio_data, circularity_data = [[] for i in range(len(all_movie_paths))], [[] for i in
                                                                                          range(len(all_movie_paths))]

        movie_index = 0
        for movie_path in all_movie_paths:

            # first check if the corresponding control movie exists!
            ctrl_movie_path = 'Ctrl_' + movie_path.name
            ctrl_movie_path = parent_path / 'Ctrl' / movie_path.parent.name / ctrl_movie_path
            if not ctrl_movie_path.is_file():
                continue  # there is no corresponding ctrl movie, so we skip this condition movie!

            # find cell index/label of handpicked cells FOR CTRL
            handpicked_cells = Read_files.read_handpicked_excel_file(ctrl_movie_path)
            hp_cells_movie = handpicked_cells[np.where(handpicked_cells[:, 0] == ctrl_movie_path.name)[0], :]
            hp_cells_movie = hp_cells_movie[hp_cells_movie[:, 1].argsort()]  # sort by cell number
            ctrl_hp_cell_indices = hp_cells_movie[hp_cells_movie[:, 2] == 'n'][:, 1].astype(int)
            if len(ctrl_hp_cell_indices) == 0:
                continue  # no good cells found, I can skip this movie and do not/ cannot plot anything

            # find cell index/label of handpicked cells FOR CONDITION
            handpicked_cells = Read_files.read_handpicked_excel_file(movie_path)
            hp_cells_movie = handpicked_cells[np.where(handpicked_cells[:, 0] == movie_path.name)[0], :]
            hp_cells_movie = hp_cells_movie[hp_cells_movie[:, 1].argsort()]  # sort by cell number
            hp_cell_indices = hp_cells_movie[hp_cells_movie[:, 2] == 'n'][:, 1].astype(int)
            if len(hp_cell_indices) == 0:
                continue  # no good cells found, I can skip this movie and do not/ cannot plot anything

            # read in properties of CONDITION FILE
            solidity, convexity, _, major_axis, minor_axis, circularity = Read_files.read_cell_shape_properties(
                movie_path)
            aspect_ratio = minor_axis / major_axis
            amount_of_cells = solidity.shape[0]
            _, time_scaling = Read_files.read_scaling(movie_path)
            # read in properties of CTRL FILE
            solidity_c, convexity_c, _, major_axis, minor_axis, circularity_c = Read_files.read_cell_shape_properties(
                ctrl_movie_path)
            aspect_ratio_c = minor_axis / major_axis
            _, time_scaling_c = Read_files.read_scaling(ctrl_movie_path)

            # SORT CELLS TOGETHER AND SUBTRACT VALUES
            for cell in range(amount_of_cells):
                if cell in hp_cell_indices:  # condition cell is considered normal
                    ctrl_cell_nr = int(hp_cells_movie[cell, 4])  # corresponding ctrl cell
                    if ctrl_cell_nr in ctrl_hp_cell_indices:  # corresponding ctrl cell is considered normal
                        # print(movie_path.name, ctrl_movie_path.name)
                        # print('condition cell', cell, 'is normal, and ctrl cell', ctrl_cell_nr, 'is also normal')
                        if summary_stat == 'mean':
                            solidity_data[movie_index].append(np.mean(solidity[cell, :]) -
                                                              np.mean(solidity_c[ctrl_cell_nr, :]))
                            convexity_data[movie_index].append(np.mean(convexity[cell, :]) -
                                                               np.mean(convexity_c[ctrl_cell_nr, :]))
                            aspect_ratio_data[movie_index].append(np.mean(aspect_ratio[cell, :]) -
                                                                  np.mean(aspect_ratio_c[ctrl_cell_nr, :]))
                            circularity_data[movie_index].append(np.mean(circularity[cell, :]) -
                                                                 np.mean(circularity_c[ctrl_cell_nr, :]))
                        elif summary_stat == 'std':
                            solidity_data[movie_index].append(np.std(solidity[cell, :]) -
                                                              np.std(solidity_c[ctrl_cell_nr, :]))
                            convexity_data[movie_index].append(np.std(convexity[cell, :]) -
                                                               np.std(convexity_c[ctrl_cell_nr, :]))
                            aspect_ratio_data[movie_index].append(np.std(aspect_ratio[cell, :]) -
                                                                  np.std(aspect_ratio_c[ctrl_cell_nr, :]))
                            circularity_data[movie_index].append(np.std(circularity[cell, :]) -
                                                                 np.std(circularity_c[ctrl_cell_nr, :]))
                        elif summary_stat == 'trend':
                            # linear regression for solidity
                            model = LinearRegression().fit(time_scaling.reshape((-1, 1)), solidity[cell, :])
                            model_c = LinearRegression().fit(time_scaling_c.reshape((-1, 1)),
                                                             solidity_c[ctrl_cell_nr, :])
                            solidity_data[movie_index].append(model.coef_[0] - model_c.coef_[0])

                            # linear regression for convexity
                            model = LinearRegression().fit(time_scaling.reshape((-1, 1)), convexity[cell, :])
                            model_c = LinearRegression().fit(time_scaling_c.reshape((-1, 1)),
                                                             convexity_c[ctrl_cell_nr, :])
                            convexity_data[movie_index].append(model.coef_[0] - model_c.coef_[0])

                            # linear regression for aspect ratio
                            model = LinearRegression().fit(time_scaling.reshape((-1, 1)), aspect_ratio[cell, :])
                            model_c = LinearRegression().fit(time_scaling_c.reshape((-1, 1)),
                                                             aspect_ratio_c[ctrl_cell_nr, :])
                            aspect_ratio_data[movie_index].append(model.coef_[0] - model_c.coef_[0])

                            # linear regression for circularity
                            model = LinearRegression().fit(time_scaling.reshape((-1, 1)), circularity[cell, :])
                            model_c = LinearRegression().fit(time_scaling_c.reshape((-1, 1)),
                                                             circularity_c[ctrl_cell_nr, :])
                            circularity_data[movie_index].append(model.coef_[0] - model_c.coef_[0])

            movie_index += 1

        # remove last position(s) where no good cells were found in the movie
        if len(all_movie_paths) != movie_index:
            solidity_data = [i for i in solidity_data if i is not None]
            convexity_data = [i for i in convexity_data if i is not None]
            aspect_ratio_data = [i for i in aspect_ratio_data if i is not None]
            circularity_data = [i for i in circularity_data if i is not None]
            # solidity_data = solidity_data[:-(len(all_movie_paths) - movie_index)]
            # convexity_data = convexity_data[:-(len(all_movie_paths) - movie_index)]
            # aspect_ratio_data = aspect_ratio_data[:-(len(all_movie_paths) - movie_index)]
            # circularity_data = circularity_data[:-(len(all_movie_paths) - movie_index)]

        # store property data from movie of resp. condition in correct place in all_data
        all_data[0][cond_index] = np.concatenate(solidity_data)
        all_data[1][cond_index] = np.concatenate(convexity_data)
        all_data[2][cond_index] = np.concatenate(aspect_ratio_data)
        all_data[3][cond_index] = np.concatenate(circularity_data)

    x_labs = ['M-CSF', 'Iso']
    my_dpi = 500
    # for storing plots
    fig_path = Path().absolute() / 'Fig3-4_CellSizeBoxplots'
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    if summary_stat == 'mean':
        # MAKE SOLIDITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labs, y_label=r'$\Delta$(Mean soli.)', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_mean_direct_Solidity.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE CONVEXITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labs, y_label=r'$\Delta$(Mean conv.)', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_mean_direct_Convexity.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE ASPECT RATIO BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[2], x_labels=x_labs, y_label=r'$\Delta$(Mean asp. ratio)', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_mean_direct_AspectRatio.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE CIRCULARITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[3], x_labels=x_labs, y_label=r'$\Delta$(Mean circ.)', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_mean_direct_Circularity.png', dpi=my_dpi)
        plt.close(fig)
    elif summary_stat == 'std':
        # MAKE SOLIDITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labs, y_label=r'$\Delta$(Std of soli.)', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_std_direct_Solidity.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE CONVEXITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labs, y_label=r'$\Delta$(Std of conv.)', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_std_direct_Convexity.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE ASPECT RATIO BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[2], x_labels=x_labs, y_label=r'$\Delta$(Std of asp. ratio)', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_std_direct_AspectRatio.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE CIRCULARITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[3], x_labels=x_labs, y_label=r'$\Delta$(Std of circ.)', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_std_direct_Circularity.png', dpi=my_dpi)
        plt.close(fig)
    elif summary_stat == 'trend':
        # MAKE SOLIDITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[0], x_labels=x_labs, y_label=r'$\Delta$(Trend of soli.) $[1/s]$', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_trend_direct_Solidity.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE CONVEXITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[1], x_labels=x_labs, y_label=r'$\Delta$(Trend of conv.) $[1/s]$', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_trend_direct_Convexity.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE ASPECT RATIO BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[2], x_labels=x_labs, y_label=r'$\Delta$(Trend of asp. ratio) $[1/s]$',
                               colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_trend_direct_AspectRatio.png', dpi=my_dpi)
        plt.close(fig)
        # MAKE CIRCULARITY BOXPLOT
        fig = AuxilliaryFct_create_boxplot(all_data[3], x_labels=x_labs, y_label=r'$\Delta$(Trend of circ.) $[1/s]$', colmap=colmap,
                               population_comp=False)
        fig.savefig(fig_path / 'CellShape_trend_direct_Circularity.png', dpi=my_dpi)
        plt.close(fig)

    return 0


# c
def AuxiliaryFct_read_features(parent_path, conditions, use_identical=False):
    """
    :param parent_path: pathlib path to outermost directory of stored data (folder called 'MacrophageData')
    :param conditions: which (chemical or physical) conditions to read in
    :param use_identical: whether to read in cell data of M-CSF or Isoflurane where the same cell is compared in Ctrl and
     conditon state
    :return: pandas data frame: features (like mean area etc) for the single cells of the different conditions
            list: with name of features extracted (see my_columns)
    """
    my_columns = ['Mean area', 'Mean perimeter', 'Mean mobile area', 'Fixed area',
                  'Std area', 'Std perimeter', 'Trend area', 'Trend perimeter',
                  'Mean soli.', 'Mean conv.', 'Mean circ.', 'Mean asp. ratio',
                  'Std soli.', 'Std conv.', 'Std circ.', 'Std asp. ratio',
                  'Trend soli.', 'Trend conv.', 'Trend circ.', 'Trend asp. ratio', 'Label', 'Movie name', 'Cell Nr']
    amount_of_columns = len(my_columns)

    features = my_columns[:20]
    df = pd.DataFrame(columns=my_columns)

    # READ-IN and STORING as pandas DATAFRAME
    if not use_identical:
        for cond_index, condition in enumerate(conditions):
            # all movie paths for one condition:
            p = (parent_path / condition / 'TifData').glob('*.tif')
            all_movie_paths = [x for x in p if x.is_file()]

            movie_index = 0
            for movie_path in all_movie_paths:
                # read in conditions, time scaling, handpicked cell excel file
                areas, fixed_area, mobile_areas, perimeters, pixel_area, intensity_areas = \
                    Read_files.read_cell_size_properties(movie_path)
                solidity, convexity, _, major_axis, minor_axis, circularity = \
                    Read_files.read_cell_shape_properties(movie_path)
                aspect_ratio = minor_axis / major_axis
                _, time_scaling = Read_files.read_scaling(movie_path)

                # find cell index/label of handpicked cells
                handpicked_cells = Read_files.read_handpicked_excel_file(movie_path)
                hp_cells_movie = handpicked_cells[np.where(handpicked_cells[:, 0] == movie_path.name)[0], :]
                hp_cells_movie = hp_cells_movie[hp_cells_movie[:, 1].argsort()]  # sort by cell number
                hp_cell_indices = hp_cells_movie[hp_cells_movie[:, 2] == 'n'][:, 1].astype(int)
                if len(hp_cell_indices) == 0:
                    continue  # no good cells found, I can skip this movie and do not/ cannot plot anything

                movie_feature_data = np.zeros((len(hp_cell_indices), amount_of_columns), dtype=object)

                # storing cell size properties of the respective movie
                # MEAN CELL SIZE
                movie_feature_data[:, 0] = np.mean(areas[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 1] = np.mean(perimeters[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 2] = np.mean(mobile_areas[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 3] = fixed_area[hp_cell_indices]
                # STD CELL SIZE
                movie_feature_data[:, 4] = np.std(areas[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 5] = np.std(perimeters[hp_cell_indices, :], axis=1)
                # TREND CELL SIZE
                for cell_index, cell in enumerate(hp_cell_indices):
                    # linear regression for area
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), areas[cell, :])
                    movie_feature_data[cell_index, 6] = model.coef_[0]
                    # linear regression for perimeter
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), perimeters[cell, :])
                    movie_feature_data[cell_index, 7] = model.coef_[0]

                # MEAN CELL SHAPE
                movie_feature_data[:, 8] = np.mean(solidity[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 9] = np.mean(convexity[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 10] = np.mean(circularity[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 11] = np.mean(aspect_ratio[hp_cell_indices, :], axis=1)
                # STD CELL SHAPE
                movie_feature_data[:, 12] = np.std(solidity[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 13] = np.std(convexity[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 14] = np.std(circularity[hp_cell_indices, :], axis=1)
                movie_feature_data[:, 15] = np.std(aspect_ratio[hp_cell_indices, :], axis=1)
                # TREND CELL SHAPE
                for cell_index, cell in enumerate(hp_cell_indices):
                    # linear regression for solidity
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), solidity[cell, :])
                    movie_feature_data[cell_index, 16] = model.coef_[0]
                    # linear regression for convexity
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), convexity[cell, :])
                    movie_feature_data[cell_index, 17] = model.coef_[0]
                    # linear regression for circularity
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), circularity[cell, :])
                    movie_feature_data[cell_index, 18] = model.coef_[0]
                    # linear regression for aspect ratio
                    model = LinearRegression().fit(time_scaling.reshape((-1, 1)), aspect_ratio[cell, :])
                    movie_feature_data[cell_index, 19] = model.coef_[0]
                    movie_feature_data[cell_index, 22] = cell

                # STORE (information of all cells of a single movie) IN PANDAS DATAFRAME
                movie_feature_data[:, 20] = condition
                movie_feature_data[:, 21] = movie_path.name
                temp_df = pd.DataFrame(movie_feature_data, columns=my_columns)
                df = pd.concat([df, temp_df], ignore_index=True)

                movie_index += 1

    # READ-IN and STORING as pandas DATAFRAME for IDENTICAL CELLS
    elif use_identical:
        for cond_index, condition in enumerate(conditions):
            # all movie paths for one condition:
            p = (parent_path / condition / 'Originals').glob('*.tif')
            all_movie_paths = [x for x in p if x.is_file()]

            movie_index = 0
            for movie_path in all_movie_paths:
                # first check if the corresponding control movie exists!
                ctrl_movie_path = 'Ctrl_' + movie_path.name
                ctrl_movie_path = parent_path / 'Ctrl' / 'Originals' / ctrl_movie_path
                if not ctrl_movie_path.is_file():
                    continue  # there is no corresponding ctrl movie, so we skip this condition movie!

                # find cell index/label of handpicked cells FOR CTRL
                handpicked_cells = Read_files.read_handpicked_excel_file(ctrl_movie_path)
                hp_cells_movie = handpicked_cells[np.where(handpicked_cells[:, 0] == ctrl_movie_path.name)[0], :]
                hp_cells_movie = hp_cells_movie[hp_cells_movie[:, 1].argsort()]  # sort by cell number
                ctrl_hp_cell_indices = hp_cells_movie[hp_cells_movie[:, 2] == 'n'][:, 1].astype(int)
                if len(ctrl_hp_cell_indices) == 0:
                    continue  # no good cells found, I can skip this movie and do not/ cannot plot anything

                # find cell index/label of handpicked cells
                handpicked_cells = Read_files.read_handpicked_excel_file(movie_path)
                hp_cells_movie = handpicked_cells[np.where(handpicked_cells[:, 0] == movie_path.name)[0], :]
                hp_cells_movie = hp_cells_movie[hp_cells_movie[:, 1].argsort()]  # sort by cell number
                hp_cell_indices = hp_cells_movie[hp_cells_movie[:, 2] == 'n'][:, 1].astype(int)
                if len(hp_cell_indices) == 0:
                    continue  # no good cells found, I can skip this movie and do not/ cannot plot anything

                # find out indices of ctrl - condition cell pairs
                ctrl_hp_cell_indices_new = np.full(hp_cell_indices.shape, -1)
                for i, cell_id in enumerate(hp_cell_indices):
                    ctrl_cell_nr = int(hp_cells_movie[cell_id, 4])
                    if ctrl_cell_nr in ctrl_hp_cell_indices:
                        ctrl_hp_cell_indices_new[i] = ctrl_cell_nr
                    else:
                        hp_cell_indices[i] = -1
                ctrl_hp_cell_indices = ctrl_hp_cell_indices_new[ctrl_hp_cell_indices_new != -1]
                hp_cell_indices = hp_cell_indices[hp_cell_indices != -1]

                both_conds = [condition, 'Ctrl']
                both_cell_indices = [hp_cell_indices, ctrl_hp_cell_indices]
                both_m_paths = [movie_path, ctrl_movie_path]
                for cell_indices, cond, m_path in zip(both_cell_indices, both_conds, both_m_paths):
                    # read in conditions, time scaling
                    areas, fixed_area, mobile_areas, perimeters, pixel_area, intensity_areas = \
                        Read_files.read_cell_size_properties(m_path)
                    solidity, convexity, _, major_axis, minor_axis, circularity = \
                        Read_files.read_cell_shape_properties(m_path)
                    aspect_ratio = minor_axis / major_axis
                    _, time_scaling = Read_files.read_scaling(m_path)

                    movie_feature_data = np.zeros((len(hp_cell_indices), amount_of_columns), dtype=object)
                    # storing cell size properties of the respective movie
                    # MEAN CELL SIZE
                    movie_feature_data[:, 0] = np.mean(areas[cell_indices, :], axis=1)
                    movie_feature_data[:, 1] = np.mean(perimeters[cell_indices, :], axis=1)
                    movie_feature_data[:, 2] = np.mean(mobile_areas[cell_indices, :], axis=1)
                    movie_feature_data[:, 3] = fixed_area[cell_indices]
                    # STD CELL SIZE
                    movie_feature_data[:, 4] = np.std(areas[cell_indices, :], axis=1)
                    movie_feature_data[:, 5] = np.std(perimeters[cell_indices, :], axis=1)
                    # TREND CELL SIZE
                    for cell_index, cell in enumerate(cell_indices):
                        # linear regression for area
                        model = LinearRegression().fit(time_scaling.reshape((-1, 1)), areas[cell, :])
                        movie_feature_data[cell_index, 6] = model.coef_[0]
                        # linear regression for perimeter
                        model = LinearRegression().fit(time_scaling.reshape((-1, 1)), perimeters[cell, :])
                        movie_feature_data[cell_index, 7] = model.coef_[0]

                    # MEAN CELL SHAPE
                    movie_feature_data[:, 8] = np.mean(solidity[cell_indices, :], axis=1)
                    movie_feature_data[:, 9] = np.mean(convexity[cell_indices, :], axis=1)
                    movie_feature_data[:, 10] = np.mean(circularity[cell_indices, :], axis=1)
                    movie_feature_data[:, 11] = np.mean(aspect_ratio[cell_indices, :], axis=1)
                    # STD CELL SHAPE
                    movie_feature_data[:, 12] = np.std(solidity[cell_indices, :], axis=1)
                    movie_feature_data[:, 13] = np.std(convexity[cell_indices, :], axis=1)
                    movie_feature_data[:, 14] = np.std(circularity[cell_indices, :], axis=1)
                    movie_feature_data[:, 15] = np.std(aspect_ratio[cell_indices, :], axis=1)
                    # TREND CELL SHAPE
                    for cell_index, cell in enumerate(cell_indices):
                        # linear regression for solidity
                        model = LinearRegression().fit(time_scaling.reshape((-1, 1)), solidity[cell, :])
                        movie_feature_data[cell_index, 16] = model.coef_[0]
                        # linear regression for convexity
                        model = LinearRegression().fit(time_scaling.reshape((-1, 1)), convexity[cell, :])
                        movie_feature_data[cell_index, 17] = model.coef_[0]
                        # linear regression for circularity
                        model = LinearRegression().fit(time_scaling.reshape((-1, 1)), circularity[cell, :])
                        movie_feature_data[cell_index, 18] = model.coef_[0]
                        # linear regression for aspect ratio
                        model = LinearRegression().fit(time_scaling.reshape((-1, 1)), aspect_ratio[cell, :])
                        movie_feature_data[cell_index, 19] = model.coef_[0]

                    # STORE (information of all cells of a single movie) IN PANDAS DATAFRAME
                    movie_feature_data[:, 20] = cond
                    temp_df = pd.DataFrame(movie_feature_data, columns=my_columns)
                    df = pd.concat([df, temp_df])

                movie_index += 1

    return df, features


def AuxiliaryFct_perform_dim_red(orig_dataf, trans_dataf, labels, method='PCA', ndim=2):
    """
    :param orig_dataf: pandas data frame, original data as extracted by fct AuxiliaryFct_read_features
    :param trans_dataf: pandas data frame, data normalized to be used in dimensionality reduction algorithms
    :param labels: labels/ names of the features of the cells in the dataframe
    :param method: string, which method to use for the dimensionality reduction (e.g. PCA, UMAP, T-SNE, ...)
    :param ndim: int (2 or 3), whether to reduce data to 2 or 3 dimensions
    :return: results - array-like of shape (n_samples, n_features). Direct output of dimensionaltiy reduction method
            fit_results - array-like of shape (n_samples, n_features). Transformed input data according to dim. red. method
            axes_name - string of abbreviation of dim. red. method to be used as axes labels for plotting later
            title - string: Whole name of dimensionality reduction method, can be used as plot title.
    """
    available_methods = ['PCA', 'KernelPCA', 'TSNE', 'Isomap', 'LLE', 'SpecEmb', 'MDS', 'LDA', 'T-SVD', 'FA', 'ICA',
                         'NMF', 'LaDA', 'NCA', 'UMAP']
    axes_names = ['PC ', 'K-PCA ', 'T-SNE ', 'Iso ', 'LLE ', 'SE ', 'MDS ', 'LDA ', 'T-SVD ', 'FA ', 'ICA ', 'NMF ',
                  'LaDA ', 'NCA ', 'UMAP ']
    titles = ['Principal Component Analysis', 'Kernel Principal Component Analysis',
              'T-Distributed Stochastic Neighbor Embedding', 'Isometric Mapping', 'Locally Linear Embedding',
              'Spectral Embedding', 'Multi-Dimensional Scaling', 'Linear Discriminant Analysis',
              'Truncated Singular Value Decomposition', 'Factor Analysis', 'Fast Independent Component Analysis',
              'Non-Negative Matrix Factorization', 'Latent Dirichlet Allocation', 'Neighborhood Components Analysis',
              'Uniform Manifold Approximation and Projection']
    if method not in available_methods:
        print('you are using an unknown method. instead PCA is used!')
        method = 'PCA'

    if ndim == 3:
        dim_reduc_meths = [PCA(n_components=3), KernelPCA(n_components=3, kernel='poly', degree=3),
                           TSNE(n_components=3),
                           Isomap(n_components=3), LocallyLinearEmbedding(n_components=3),
                           SpectralEmbedding(n_components=3),
                           MDS(n_components=3, metric=False), LinearDiscriminantAnalysis(n_components=3),
                           TruncatedSVD(n_components=3), FactorAnalysis(n_components=3),
                           FastICA(n_components=3, whiten='unit-variance'), NMF(n_components=3),
                           LatentDirichletAllocation(n_components=3), NeighborhoodComponentsAnalysis(n_components=3),
                           umap.UMAP(n_components=3)]
    elif ndim == 2:
        dim_reduc_meths = [PCA(n_components=2), KernelPCA(n_components=2, kernel='poly', degree=3),
                           TSNE(n_components=2),
                           Isomap(n_components=2), LocallyLinearEmbedding(n_components=2),
                           SpectralEmbedding(n_components=2),
                           MDS(n_components=2, metric=False), LinearDiscriminantAnalysis(n_components=2),
                           TruncatedSVD(n_components=2), FactorAnalysis(n_components=2),
                           FastICA(n_components=2, whiten='unit-variance'), NMF(n_components=2),
                           LatentDirichletAllocation(n_components=2), NeighborhoodComponentsAnalysis(n_components=3),
                           umap.UMAP(n_components=2, min_dist=0.05, n_neighbors=15)]

    method_index = np.where(np.array(available_methods) == method)[0][0]
    print(method)

    if method == 'LDA' or method == 'NCA':
        if ndim == 3 and len(labels) > 3:
            results = dim_reduc_meths[method_index]
            fit_results = results.fit_transform(trans_dataf, labels)
        elif ndim == 2 and len(labels) > 2:
            results = dim_reduc_meths[method_index]
            fit_results = results.fit_transform(trans_dataf, labels)
    elif method == 'NMF' or method == 'LaDA':  # no negative values
        # negative values in trends: now consider strength of trend, but not 'direction' (increasing vs decreasing)
        results = dim_reduc_meths[method_index]
        fit_results = results.fit_transform(np.abs(orig_dataf))
    else:
        results = dim_reduc_meths[method_index]
        fit_results = results.fit_transform(trans_dataf)

    return results, fit_results, axes_names[method_index], titles[method_index]


def Fig3_4_DimensReduction(movie_path, imaging_condition='in_vivo', method='PCA', use_condition=['Ctrl', 'LPS'],
               identical_cells=False, ndim=2, use_legend=True, plot_acc_to_movie=False):
    """
    IDEA: read in feature data of all cells for the given condition(s), perform a dimensionality reduction of the given
    features (see method), and then plot 'cells' in the reduced space (2D or 3D), colored according to their condition
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie files are chosen in the fct)
    :param imaging_condition: string 'in_vivo' or 'Explant'
    :param method: string, which dimensionality reduction to use. Available parameters are 'PCA', 'KernelPCA', 'TSNE',
                    'Isomap', 'LLE', 'SpecEmb', 'MDS', 'LDA', 'T-SVD', 'FA', 'ICA', 'NMF', 'LaDA', 'NCA', 'UMAP'
    :param use_condition: list of strings, which conditions to use in the dimensionality reduction.
                            For in vivo: 'Ctrl', 'LPS', 'Isoflurane', 'YM201636'
                            For Explant: 'Ctrl', 'ExplantDirect', ExplantRest'
    :param identical_cells: bool, For in vivo conditions MCSF or Isoflurane, whether to only consider cells that are
                            available both in control and in perturbed condition
    :param ndim: int 2 or 3: whether to reduce the dimensionality to 2 or 3 dimensions
    :param use_legend: bool, whether to add the legend to the scatter plot
    :param plot_acc_to_movie: bool, whether to additionally store plot, where cells of one condition are colored
    according to the experiment/ movie they stem from
    :return: 0, stores plots in subfolder of current working directory
    """
    if imaging_condition == 'in_vivo':
        conditions = ['Ctrl', 'LPS', 'MCSF', 'Isoflurane', 'YM201636']
        conditions_legend = ['Ctrl', 'LPS', 'M-CSF', 'Iso', 'YM']
        colors = ['lightskyblue', 'sandybrown', 'darkseagreen', 'gold', 'plum']
    elif imaging_condition == 'Explant':
        conditions = ['Ctrl', 'ExplantDirect', 'ExplantRest']
        conditions_legend = ['$\it{in}$ $\it{vivo}$', 'Expl direct', 'Expl rest']
        colors = ['lightskyblue', 'lightseagreen', 'teal']

    # available_conditions = ['LPS', 'MCSF',]
    if use_condition == 'all':
        if imaging_condition == 'in_vivo':
            use_condition = ['LPS', 'MCSF', 'Isoflurane', 'YM201636']
        elif imaging_condition == 'Explant':
            use_condition = ['ExplantDirect', 'ExplantRest']

    # READ IN all cells with all features (pd frame)
    parent_path = movie_path.parent.parent.parent  # go to 'outermost' directory ('MacrophageData')
    rtm_df, features = AuxiliaryFct_read_features(parent_path, use_condition, use_identical=identical_cells)

    # TRANSFORM DATA (so that all features are distributed normally!!) actually transform data/ reduce dimensions
    x = rtm_df.loc[:, features].values
    x = StandardScaler().fit_transform(x)  # normalizing the features
    y = rtm_df.loc[:, 'Label'].values

    res, fit_res, ax_name, titl = AuxiliaryFct_perform_dim_red(rtm_df.loc[:, features].values, x, y, method=method, ndim=2)

    use_colors, use_legends = [], []
    condi_string = ''
    for i, condi in enumerate(use_condition):
        # get correct color and legend name
        cond_index = conditions.index(condi)
        use_colors.append(colors[cond_index])
        use_legends.append(conditions_legend[cond_index])
        condi_string += condi

    # PLOT:
    figsize = (2.3, 1.5)  # for in vivo (Fig 3)
    figsize = (2.2, 1.6)  # for Explant (Fig 4)
    label_fontsize = 8
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=figsize)
    if ndim == 3:
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel(ax_name + str(1), fontsize=label_fontsize)
        ax.set_ylabel(ax_name + str(2), fontsize=label_fontsize)
        ax.set_zlabel(ax_name + str(3), fontsize=label_fontsize)
        for target, color in zip(use_condition, use_colors):
            indicesToKeep = np.array((rtm_df['Label'] == target))
            ax.scatter(fit_res[indicesToKeep, 0], fit_res[indicesToKeep, 1], fit_res[indicesToKeep, 2], c=color, s=15,
                       label=target)

    elif ndim == 2:
        ax = fig.add_subplot()
        ax.set_xlabel(ax_name + str(1), fontsize=label_fontsize)
        ax.set_ylabel(ax_name + str(2), fontsize=label_fontsize, labelpad=1.3)
        ax.tick_params(axis='y', size=2, width=1, labelsize=label_fontsize, pad=1)
        ax.tick_params(axis='x', size=2, width=1, labelsize=label_fontsize, pad=1)

        for target, color, label in zip(use_condition, use_colors, use_legends):
            indicesToKeep = np.array((rtm_df['Label'] == target))
            ax.scatter(fit_res[indicesToKeep, 0], fit_res[indicesToKeep, 1], c=color, s=15, label=label)

    if use_legend:
        ax.legend(fontsize=label_fontsize, handletextpad=0.15, markerscale=0.75, labelspacing=0.25)
    fig.tight_layout(pad=0.0)
    fig_path = Path().absolute() / 'Fig3-4_DimensionReduction'
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    fig_name = method + '_' + condi_string + '_' + str(ndim) + 'D.png'
    if identical_cells:
        fig_name = fig_name[:-4] + '_direct.png'
    fig.savefig(fig_path / fig_name, dpi=500, bbox_inches='tight')
    plt.close('all')

    # FOR SCATTERING ACCORDING TO MOVIE NAMES
    if plot_acc_to_movie:
        if imaging_condition == 'in_vivo':
            my_cond = 'Ctrl'
        elif imaging_condition == 'Explant':
            my_cond = 'ExplantDirect'
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        ax.set_xlabel(ax_name + str(1), fontsize=label_fontsize)
        ax.set_ylabel(ax_name + str(2), fontsize=label_fontsize, labelpad=1.3)
        ax.tick_params(axis='x', size=2, width=1, labelsize=label_fontsize, pad=1)
        ax.tick_params(axis='y', size=2, width=1, labelsize=label_fontsize, pad=1)
        helper_arr = rtm_df.loc[(rtm_df['Label'] == my_cond)]
        use_movies = helper_arr['Movie name'].unique()  # gives amount of unique movies for the given condition!
        colors = plt.get_cmap('terrain', len(use_movies) + 2)

        # Plot all other conditions
        indicesToKeep = np.array((rtm_df['Label'] != my_cond))
        if np.any(indicesToKeep):
            ax.scatter(fit_res[indicesToKeep, 0], fit_res[indicesToKeep, 1], c='gray', marker='*', s=3, label='Others')

        for index, target in enumerate(use_movies):
            indicesToKeep = np.array(((rtm_df['Movie name'] == target) & (rtm_df['Label'] == my_cond)))
            if np.any(indicesToKeep):
                ax.scatter(fit_res[indicesToKeep, 0], fit_res[indicesToKeep, 1], color=colors(index), s=15,
                           label=target[0])

        if use_legend:
            use_movies = np.insert(use_movies, 0, 'other conditions')
            ax.legend(use_movies, fontsize=2, handletextpad=0.15, markerscale=0.5)
        fig.tight_layout(pad=0.0)
        # for storing plots
        fig_path = Path().absolute() / 'Fig3-4_DimensionReduction'
        Path(fig_path).mkdir(parents=True, exist_ok=True)
        fig_name = method + '_MovieNames_' + my_cond + '_' + condi_string
        if identical_cells:
            fig_name = fig_name + '_direct_2D_' + imaging_condition + '.png'
        else:
            fig_name = fig_name + '_2D_' + imaging_condition + '.png'
        fig.savefig(fig_path / fig_name, dpi=500, bbox_inches='tight')

    plt.close('all')

    return 0


def Fig5_LabellingSnapshots(movie_path):
    """
    IDEA: function to store snapshots of the labelling process. For more details on the labelling itself, please refer
    to the function Generate_files.store_label_array()
    :param movie_path: pathlib Path to .tif file of an arbitrary movie (the correct movie files are chosen in the fct)
    :return: 0, stores snapshots
    """

    frame = 0

    movie_path = movie_path.parent.parent.parent / 'Ctrl' / 'TifData' / 'Ctrl_2021-04-29_2.tif'

    #################
    # READ-IN
    #################
    gray_arr, _ = Read_files.read_grayscale(movie_path)
    bin_arr, amount_of_frames = Read_files.read_binary(movie_path)
    len_scaling, _ = Read_files.read_scaling(movie_path)
    mean_arr, mean_arr_bin, _, mean_arr_labels = Generate_files.generate_mean_shape(movie_path)
    amount_of_cells = np.unique(mean_arr_labels)[-1]

    colors = Generate_files.generate_color_map(amount_of_cells)

    # STORE FIRST FEW SNAPSHOTS
    fig_path = Path().absolute() / 'Fig5'
    Path(fig_path).mkdir(parents=True, exist_ok=True)

    figsize = np.array([6, 6 * bin_arr.shape[0] / bin_arr.shape[1]])
    fig = plt.figure(figsize=figsize) # same fig will be reused multiple times
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot()
    ax.axis('off')

    # snapshot 1 - GRAYSCALE
    ax.imshow(gray_arr[:, :, frame], cmap='gray')
    fig.tight_layout(pad=0)
    fig.savefig(fig_path / 'Snapshot1.png', dpi=500)

    # snapshot 2 - BINARY UNPROCESSED
    ax.imshow(bin_arr[:, :, frame], cmap='gray')
    fig.tight_layout(pad=0)
    fig.savefig(fig_path / 'Snapshot2.png', dpi=500)

    # snapshot 4 - MEAN OF FRAMES
    ax.imshow(mean_arr, cmap='gray')
    fig.tight_layout(pad=0)
    fig.savefig(fig_path / 'Snapshot4.png', dpi=500)

    # snapshot 5 - MEAN (OF) FRAMES THRESHOLDED
    ax.imshow(mean_arr_bin, cmap='gray')
    fig.savefig(fig_path / 'Snapshot5.png', dpi=500)

    # snapshot 6 - MEAN (OF) FRAMES THRESHOLDED AND LABELLED
    ax.imshow(skimage.color.label2rgb(mean_arr_labels, image=mean_arr_bin, colors=colors.colors, kind='overlay',
                                        alpha=1))
    fig.savefig(fig_path / 'Snapshot6.png', dpi=500)

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
    # labelling the 'ORIGINAL' binary image
    rw_labels = np.copy(mean_arr_labels)
    plot_arr = skimage.color.label2rgb(rw_labels, bg_label=0, colors=colors.colors)
    rw_labels[np.where(bin_arr[:, :, frame] == 0)] = -1  # background (always dark) should not be labelled
    tempor_labelled_arr = skimage.segmentation.watershed(image=bin_arr[:, :, frame], markers=rw_labels,
                                                         mask=bin_arr[:, :, frame])

    # snapshot 7
    plot_arr[np.where(rw_labels == 0)[0], np.where(rw_labels == 0)[1], :] = [1, 1, 1]
    ax.imshow(plot_arr)
    fig.savefig(fig_path / 'Snapshot7.png', dpi=500)

    # snapshot 8
    ax.imshow(skimage.color.label2rgb(tempor_labelled_arr, bg_label=0, colors=colors.colors))
    fig.savefig(fig_path / 'Snapshot8.png', dpi=500)

    # labelling the DILATED binary image (finding cutoff protrusions)
    dilated_binary = skimage.morphology.binary_dilation(bin_arr[:, :, frame], footprint=se)
    rw_labels_2 = np.zeros(dilated_binary.shape, dtype=np.int8)
    rw_labels_2[dilated_binary] = tempor_labelled_arr[dilated_binary]
    plot_arr = skimage.color.label2rgb(rw_labels_2, bg_label=0, colors=colors.colors)
    rw_labels_2[np.where(dilated_binary == 0)] = -1
    tempor_labelled_arr_dil = skimage.segmentation.watershed(image=dilated_binary, markers=rw_labels_2,
                                                             mask=dilated_binary)

    # snapshot 9
    plot_arr[np.where(rw_labels_2 == 0)[0], np.where(rw_labels_2 == 0)[1], :] = [1, 1, 1]
    ax.imshow(plot_arr)
    fig.savefig(fig_path / 'Snapshot9.png', dpi=500)

    # snapshot 10
    ax.imshow(skimage.color.label2rgb(tempor_labelled_arr_dil, bg_label=0, colors=colors.colors))
    fig.savefig(fig_path / 'Snapshot10.png', dpi=500)

    # having CLEANED binary image with cutoff protrusions labelled correctly
    bin_arr_clea = Generate_files.generate_binary_preprocessed(bin_arr, len_scaling)
    tempor_labelled_arr_final = bin_arr_clea[:, :, frame] * tempor_labelled_arr_dil

    # snapshot 3
    ax.imshow(bin_arr_clea[:, :, frame], cmap='gray')
    fig.savefig(fig_path / 'Snapshot3.png', dpi=500)

    # snapshot 11
    plot_arr = skimage.color.label2rgb(tempor_labelled_arr_dil, bg_label=0, bg_color=None, colors=colors.colors)
    plot_arr[bin_arr_clea[:, :, frame]] *= 0.5
    ax.imshow(plot_arr)
    fig.savefig(fig_path / 'Snapshot11.png', dpi=500)

    # snapshot 12
    ax.imshow(skimage.color.label2rgb(tempor_labelled_arr_final, bg_label=0, colors=colors.colors))
    fig.savefig(fig_path / 'Snapshot12.png', dpi=500)

    plt.close('all')

    return 0


# c
def Supp_Fig1_image_intensity(movie_path):
    """
    :param movie_path: pathlib Path to an arbitrary .tif file, correct movie name will be chosen in the fct
    :return: 0, stores plot iin current working directory
    """

    movie_name = 'Ctrl.tif'
    movie_path = movie_path.parent / movie_name

    gray_arr, _ = Read_files.read_grayscale(movie_path)
    mean_int_per_frame = np.mean(gray_arr, axis=(0, 1))
    _, time_scaling = Read_files.read_scaling(movie_path)

    # PLOTTING
    fig_size = [3, 2]  # in inches
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot()
    label_fontsize = 8
    ax.tick_params(axis='y', size=2, width=1, labelsize=label_fontsize, pad=1)
    ax.tick_params(axis='x', size=2, width=1, labelsize=label_fontsize, pad=1)
    ax.set_ylabel('Image intensity', fontsize=label_fontsize, fontfamily='Arial', labelpad=2)
    ax.set_xlabel(r'Time $[s]$', fontsize=label_fontsize, fontfamily='Arial', labelpad=2)
    ax.plot(time_scaling, mean_int_per_frame / 255, color='black', lw=1.25)
    fig.tight_layout(pad=0.3)

    supp_path = Path().absolute() / 'Supplement'
    Path(supp_path).mkdir(parents=True, exist_ok=True)
    fig.savefig(supp_path / 'SF_ImageIntensity.png', dpi=600)
    plt.close('all')

    return 0


# c
def Supp_Movie_1_2(movie_path, condition='Ctrl'):
    """
    :param movie_path: pathlib Path to an arbitrary .tif file, correct movie name will be chosen in the fct
    :param condition: "BMDM" or "Ctrl"
    :return: 0, stores movie in current working directory
    """

    def animate(frame):

        if frame % 50 == 0:
            print('Frame ', frame, 'from ', amount_of_frames)

        im_to_plot = np.copy(gray_arr[:, :, frame])

        my_text = str(time_scaling[frame] // 60) + ':' + str(time_scaling[frame] % 60)
        if (time_scaling[frame] % 60) < 10:
            my_text = str(time_scaling[frame] // 60) + ':0' + str(time_scaling[frame] % 60)

        image_artist.set_data(im_to_plot)
        text_artist.set_text(my_text)

        artists = [image_artist, text_artist]
        return artists

    movie_name_input = ['BMDM_2023-07-06.tif', 'Ctrl_2021-05-21_2.tif']
    supp_path = Path().absolute() / 'Supplement'
    Path(supp_path).mkdir(parents=True, exist_ok=True)
    movie_name_output = ['BMDM_invitro.mp4', 'RTM_invivo.mp4']

    # coordinates (in pixels) where text etc should be printed on the movie
    # first entry is for BMDM movie, second entry is for Ctrl movie
    timetext_pos = [[330, 1920], [80, 470]]
    scalebar_pos = [[90, 1995], [20, 490]]
    scaletext_pos = [[147, 1975], [45, 485]]
    scalebar_length = [100, 50]
    titles = ['BMDMs $\it{in}$ $\it{vitro}$', 'RTMs $\it{in}$ $\it{vivo}$']
    horiz_alignment = ['right', 'right']
    if condition == 'BMDM':
        index = 0
    elif condition == 'Ctrl':
        index = 1
    else:
        print('Available inputs for the parameter "condition" are only "BMDM" or "Ctrl"!\n'
              'No movie was saved for the condition ', condition)
        return 0

    parent_path = movie_path.parent.parent.parent  # go to 'outermost' directory ('RTM_Data')
    # READ-IN #
    movie_path = parent_path / condition / 'TifData' / movie_name_input[index]
    gray_arr, amount_of_frames = Read_files.read_grayscale(movie_path)
    len_scaling, time_scaling = Read_files.read_scaling(movie_path)
    im_shape = gray_arr.shape

    time_betw_frames = np.mean(np.diff(time_scaling))
    my_fps = 180 / time_betw_frames  # 1s in movie should correspond to 180s in real time

    # generate figure
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=np.array([6, (6 * im_shape[0] / im_shape[1]) + 0.4]))
    ax = fig.add_subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    image_artist = ax.imshow(gray_arr[:, :, 0], cmap='gray', animated=True)
    ax.set_title(titles[index], fontweight='bold', fontsize=16)

    # plot first instance of time scaling
    text_artist = ax.text(timetext_pos[index][0], timetext_pos[index][1], str(time_scaling[0]), color='white',
                          ha=horiz_alignment[index], fontweight='bold', fontsize=12, animated=True)

    # plot length scaling
    length_of_bar_pxl = scalebar_length[index] * len_scaling  # like above, but in pxl 'units'
    ax.axhline(scalebar_pos[index][1], (scalebar_pos[index][0]) / gray_arr.shape[1],
               (scalebar_pos[index][0] + length_of_bar_pxl) / gray_arr.shape[1], lw=2.5, color='white')
    ax.text(scaletext_pos[index][0], scaletext_pos[index][1], str(scalebar_length[index]) + ' µm', color='white',
            fontweight='bold', fontsize=12)

    fig.set_dpi(600)
    fig.tight_layout(pad=0.2)

    ###############
    # MOVIE WRITER
    ###############
    writer = anim.writers['ffmpeg']
    write = writer(fps=my_fps, metadata=dict(artist='Miriam Schnitzerlein'), bitrate=1800)
    ani = anim.FuncAnimation(fig, animate, frames=amount_of_frames, repeat=False, blit=True)

    ani.save(supp_path / movie_name_output[index], writer=write)

    return 0


# c
def Supp_Movie_3(movie_path):
    """
    :param movie_path: pathlib Path to an arbitrary .tif file, correct movie name will be chosen in the fct
    :return: 0, stores movie in current working directory
    """

    def animate(frame):

        for i, data in enumerate([areas, solidity, convexity, circularity, aspect_ratio, perimeters]):
            dot_artists[i].set_data(time_scaling[frame], data[cell, frame])
        dot_artists[-1].set_data(time_scaling[frame], mobile_areas[cell, frame])

        # PREPARE MOVIE FRAME
        # ## CONVEX HULL
        convex_hull = skimage.morphology.convex_hull_object(label_arr[:, :, frame])
        convex_hull_border = skimage.segmentation.find_boundaries(convex_hull, mode='inner')
        convex_hull_border = skimage.morphology.binary_dilation(convex_hull_border, footprint=[(my_footprint, 3)])
        chull_mask = convex_hull - label_arr[:, :, frame]
        image_to_plot = skimage.color.label2rgb(chull_mask, gray_arr[:, :, frame], colors=my_colors)
        image_to_plot[convex_hull_border, :] = mcol.to_rgb(my_colors[1])

        # ## BOUNDARY OF FIXED AREA
        mean_arr_bin[~label_arr[:, :, frame]] = 0
        fixed_area_border = skimage.segmentation.find_boundaries(mean_arr_bin, mode='inner')
        fixed_area_border = skimage.morphology.binary_dilation(fixed_area_border, footprint=[(my_footprint, 3)])
        image_to_plot[fixed_area_border, :] = mcol.to_rgb(my_colors[2])

        # ## BOUNDARY OF WHOLE CELL
        area_border = skimage.segmentation.find_boundaries(label_arr[:, :, frame], mode='inner')
        area_border = skimage.morphology.binary_dilation(area_border, footprint=[(my_footprint, 4)])
        image_to_plot[area_border, :] = mcol.to_rgb(my_colors[3])
        movie_artist.set_data(image_to_plot)

        artists = np.copy(dot_artists)
        np.append(artists, movie_artist)
        return artists

    # READ-IN
    movie_path = movie_path.parent.parent / 'TifData' / 'Ctrl.tif'

    gray_arr, amount_of_frames = Read_files.read_grayscale(movie_path)
    label_arr, _ = Read_files.read_labels(movie_path)
    _, mean_arr_bin, _, _ = Generate_files.generate_mean_shape(movie_path)
    image_size = gray_arr.shape

    areas, fixed_areas, mobile_areas, perimeters, _, _ = Read_files.read_cell_size_properties(movie_path)
    solidity, convexity, _, major_axis, minor_axis, circularity = Read_files.read_cell_shape_properties(movie_path)
    aspect_ratio = minor_axis / major_axis
    len_scaling, time_scaling = Read_files.read_scaling(movie_path)
    cell = 0

    labels = [r'Area $[\mu m^2]$', 'Soli.', 'Conv.', 'Circ.', 'Asp. ratio', r'Perimeter $[\mu m]$']
    color = 'cornflowerblue'

    label_fontsize = 8
    plt.rcParams["font.family"] = "Arial"

    # DEFINE FIGURE
    fig_size = np.array([5, 5 * image_size[0] / image_size[1] * 5/3 ])
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(10, 2, figure=fig)
    axs = [fig.add_subplot(gs[:6, :]),  # movie
           fig.add_subplot(gs[6:9, 0]),  # area plot
           fig.add_subplot(gs[6, 1]),  # solidity plot
           fig.add_subplot(gs[7, 1]),  # convexity
           fig.add_subplot(gs[8, 1]),  # circularity
           fig.add_subplot(gs[9, 1]),  # aspect ratio
           fig.add_subplot(gs[9, 0])]  # perimeter plot

    # PLOT LINE PLOTS
    first_frame = 0
    dot_artists = [None] * 7
    for i, data in enumerate([areas, solidity, convexity, circularity, aspect_ratio, perimeters]):
        axs[i+1].set_ylabel(labels[i], fontsize=label_fontsize, fontfamily='Arial', labelpad=2)
        axs[i+1].tick_params(axis='y', size=2, width=1, labelsize=label_fontsize, pad=1)
        axs[i+1].yaxis.set_major_locator(ticker.MaxNLocator(2))
        if i < 4:
            axs[i+1].xaxis.set_major_formatter(ticker.NullFormatter())
            axs[i+1].tick_params(axis='x', size=2, width=1, pad=1)
        if 0 < i < 5:
            axs[i + 1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  # for shape quantifiers
        if i > 3:
            axs[i+1].tick_params(axis='x', size=2, width=1, labelsize=label_fontsize, pad=1)
            axs[i+1].set_xlabel(r'Time $[s]$', fontsize=label_fontsize, fontfamily='Arial', labelpad=2)

        axs[i+1].plot(time_scaling, data[cell, :], color=color, lw=1.6)
        if i == 0:
            area_line = axs[i+1].plot(time_scaling, data[cell, :], color=color, lw=1.6)[0]
        dot_artists[i] = axs[i+1].plot(time_scaling[first_frame], data[cell, first_frame], ls='', marker='.', ms=8, color='orange', animated=True)[0]
    # plot mobile and fixed area
    marea_line = axs[1].plot(time_scaling, mobile_areas[cell, :], ls=(0, (3, 1)), color=color, lw=1.6)[0]
    dot_artists[-1] = axs[1].plot(time_scaling[first_frame], mobile_areas[cell, first_frame], ls='', marker='.', ms=8, color='orange', animated=True)[0]

    farea_line = axs[1].axhline(fixed_areas[cell], color=color, ls=':', lw=1.6)

    axs[1].legend([area_line, marea_line, farea_line], ['Whole area', 'Mobile area', 'Fixed area'], loc=[0.5, 0.4], prop={'family':'Arial', 'size': label_fontsize})

    # PLOT FIRST INSTANCE OF MOVIE
    my_footprint = skimage.morphology.diamond(1)  # structuring element for dilation
    my_colors = ['orange', 'sandybrown', 'white', 'cornflowerblue']
    convex_hull = skimage.morphology.convex_hull_object(label_arr[:, :, first_frame])
    convex_hull_border = skimage.segmentation.find_boundaries(convex_hull, mode='inner')
    convex_hull_border = skimage.morphology.binary_dilation(convex_hull_border, footprint=[(my_footprint, 3)])
    chull_mask = convex_hull - label_arr[:, :, first_frame]
    image_to_plot = skimage.color.label2rgb(chull_mask, gray_arr[:, :, first_frame], colors=my_colors)
    image_to_plot[convex_hull_border, :] = mcol.to_rgb(my_colors[1])

    # ## BORDERS OF FIXED AREA
    mean_arr_bin[~label_arr[:, :, first_frame]] = 0
    fixed_area_border = skimage.segmentation.find_boundaries(mean_arr_bin, mode='inner')
    fixed_area_border = skimage.morphology.binary_dilation(fixed_area_border, footprint=[(my_footprint, 3)])
    image_to_plot[fixed_area_border, :] = mcol.to_rgb(my_colors[2])

    # ## BORDERS OF WHOLE CELL
    area_border = skimage.segmentation.find_boundaries(label_arr[:, :, first_frame], mode='inner')
    area_border = skimage.morphology.binary_dilation(area_border, footprint=[(my_footprint, 4)])
    image_to_plot[area_border, :] = mcol.to_rgb(my_colors[3])

    axs[0].axis('off')

    movie_artist = axs[0].imshow(image_to_plot)
    # add scalebar
    scalebar_length = 10 * len_scaling
    axs[0].plot([70, 70+scalebar_length], [image_size[0]-70, image_size[0]-70], lw=2, color='white')
    axs[0].text(105, image_size[0]-90, '10 µm', color='white', fontsize=label_fontsize+2, fontfamily='Arial', fontweight='bold')

    # add legend to movie
    axs[0].text(image_size[1]-400, image_size[0]-190, 'Convex outline', color=my_colors[0], fontsize=label_fontsize+2, fontfamily='Arial', fontweight='bold')
    axs[0].text(image_size[1]-400, image_size[0]-140, 'Cell outline', color=my_colors[3], fontsize=label_fontsize+2, fontfamily='Arial', fontweight='semibold')
    axs[0].text(image_size[1]-400, image_size[0]-90, 'Fixed area outline', color=my_colors[2], fontsize=label_fontsize+2, fontfamily='Arial', fontweight='black')

    fig.set_dpi(500)
    fig.tight_layout(pad=0.15)

    writer = anim.writers['ffmpeg']
    write = writer(fps=10, metadata=dict(artist='Miriam Schnitzerlein'), bitrate=1800)
    ani = anim.FuncAnimation(fig, animate, frames=amount_of_frames, repeat=False, blit=True)

    supp_path = Path().absolute() / 'Supplement'
    Path(supp_path).mkdir(parents=True, exist_ok=True)
    ani.save(supp_path / 'S3_movie.mp4', writer=write)

    return 0

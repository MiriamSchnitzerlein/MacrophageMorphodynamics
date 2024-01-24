from pathlib import Path  # v 1.0.1

import Generate_files
import GeneratePlotsForPaper

if __name__ == '__main__':

    AVAILABLE_CONDITIONS = ['Ctrl', 'LPS', 'Isoflurane', 'MCSF', 'YM201636',
                            'ExplantDirect', 'ExplantRest', 'BMDM']
    my_condition = AVAILABLE_CONDITIONS[0]

    # MOVIE_PATH
    DIRECTORY_PATH = Path.home() / 'Docs' / 'RTM_Data' / 'MacrophageData'
    MOVIE_PATH = DIRECTORY_PATH / my_condition / 'TifData' / 'Ctrl.tif'

    # 1. PROCESS AND ANALYSE TIF DATA AND GENERATE ALL NECESSARY (INTERMEDIATE) FILES FOR ALL MOVIES
    for condition in AVAILABLE_CONDITIONS[:-1]:
        print(condition)
        movie_directory = DIRECTORY_PATH / condition / 'TifData'

        for MOVIE_PATH in movie_directory.glob('*.tif'):
            Generate_files.store_scaling(MOVIE_PATH)
            Generate_files.store_grayscale(MOVIE_PATH)
            Generate_files.remove_frames_restore_data(MOVIE_PATH)
            Generate_files.store_binary(MOVIE_PATH)
            Generate_files.store_label_array(MOVIE_PATH)
            Generate_files.store_cell_properties(MOVIE_PATH)

    # 2. GENERATE PLOTS SHOWING ANALYSED DATA OF MACROPHAGES
    # FIGURE 2 plots
    GeneratePlotsForPaper.Fig2_CellSnapshots(MOVIE_PATH)
    GeneratePlotsForPaper.Fig2_CellSize_Plot(MOVIE_PATH)
    GeneratePlotsForPaper.Fig2_CellShape_Plot(MOVIE_PATH)

    GeneratePlotsForPaper.Fig3_4_CellSize_boxplot(MOVIE_PATH, 'trend', 'in_vivo')

    # FIGURE 3 and 4 plots
    for condition in AVAILABLE_CONDITIONS[:-1]:
        GeneratePlotsForPaper.Fig3_4_CellSnapshots(MOVIE_PATH, condition)
    for stat in ['mean', 'std', 'trend']:
        for imaging_condition in ['in_vivo', 'Explant']:
            GeneratePlotsForPaper.Fig3_4_CellSize_boxplot(MOVIE_PATH, stat, imaging_condition)
            GeneratePlotsForPaper.Fig3_4_CellShape_boxplot(MOVIE_PATH, stat, imaging_condition)
        GeneratePlotsForPaper.Fig3_CellSize_direct_boxplot(MOVIE_PATH, stat)
        GeneratePlotsForPaper.Fig3_CellShape_direct_boxplot(MOVIE_PATH, stat)

    GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'in_vivo', 'PCA', AVAILABLE_CONDITIONS[0:5], plot_acc_to_movie=True)
    GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'in_vivo', 'PCA', ['MCSF', 'YM201636'])
    GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'Explant', 'PCA', ['Ctrl', 'ExplantDirect', 'ExplantRest'], plot_acc_to_movie=True)

    GeneratePlotsForPaper.Fig5_LabellingSnapshots(MOVIE_PATH)

    # SUPPLEMENTARY DATA (plots or movies)
    GeneratePlotsForPaper.Supp_Fig1_image_intensity(MOVIE_PATH)
    for condition in ['BMDM', 'Ctrl']:
        GeneratePlotsForPaper.Supp_Movie_1_2(MOVIE_PATH, condition)
    GeneratePlotsForPaper.Supp_Movie_3(MOVIE_PATH)
    # to generate scatter plots with other dimensionality reduction methods (as in supplemental figures), choose methods accordingly
    GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'in_vivo', 'TSNE', AVAILABLE_CONDITIONS[0:5], )
    GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'Explant', 'TSNE', ['Ctrl', 'ExplantDirect', 'ExplantRest'])



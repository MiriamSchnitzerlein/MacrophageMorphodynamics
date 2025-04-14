from pathlib import Path  # v 1.0.1

import Generate_files
import GeneratePlotsForPaper

if __name__ == '__main__':

    AVAILABLE_CONDITIONS = ['Ctrl', 'MCSF', 'TGFbeta', 'LPS', 'YM201636', 'IFN10', 'Old', 'OldMCSF',
                            'Explant 1', 'Explant 2']
    my_condition = AVAILABLE_CONDITIONS[0]

    # MOVIE_PATH
    # TODO: insert the correct path here to run the script!!
    DIRECTORY_PATH = Path.home() / 'your' / 'path' / 'MacrophageData_V2'
    DIRECTORY_PATH = Path.home() / 'Docs' / 'RTM_Data' / 'MacrophageData_New'
    MOVIE_PATH = DIRECTORY_PATH / my_condition / 'TifData' / 'Ctrl_2021-04-29_1.tif'

    # 1. PROCESS AND ANALYSE TIF DATA AND GENERATE ALL NECESSARY (INTERMEDIATE) FILES FOR ALL MOVIES
    for condition in AVAILABLE_CONDITIONS:
        print(condition)
        movie_directory = DIRECTORY_PATH / condition / 'TifData'

        for MOVIE_PATH in movie_directory.glob('*.tif'):
            MOVIE_PATH = DIRECTORY_PATH / condition / 'TifData' / MOVIE_PATH

            Generate_files.store_scaling(MOVIE_PATH)
            Generate_files.store_grayscale(MOVIE_PATH)
            Generate_files.remove_frames_restore_data(MOVIE_PATH)
            Generate_files.store_binary(MOVIE_PATH)
            Generate_files.store_label_array(MOVIE_PATH)
            Generate_files.store_cell_properties(MOVIE_PATH)

    # 2. GENERATE PLOTS SHOWING ANALYSED DATA OF MACROPHAGES
    # FIGURE 2 plots
    GeneratePlotsForPaper.Fig2_CellSnapshots(MOVIE_PATH, v=2)
    GeneratePlotsForPaper.Fig2_CellSize_Plot(MOVIE_PATH, v=2)
    GeneratePlotsForPaper.Fig2_CellShape_Plot(MOVIE_PATH, v=2)

    # FIGURE 3 and 4 plots
    for condition in AVAILABLE_CONDITIONS:
        GeneratePlotsForPaper.Fig3_4_CellSnapshots_V2(MOVIE_PATH, condition)

    for imaging_condition in ['in_vivo', 'Old', 'Explant']:
        for stat in ['mean', 'std', 'trend']:
            GeneratePlotsForPaper.Fig3_4_CellSize_boxplot(MOVIE_PATH, stat, imaging_condition, v=2)
            GeneratePlotsForPaper.Fig3_4_CellShape_boxplot(MOVIE_PATH, stat, imaging_condition, v=2)
        GeneratePlotsForPaper.Fig3_4_CellSize_boxplot(MOVIE_PATH, 'dyn', imaging_condition, v=2)

    all_chem_conds = ['Ctrl', 'MCSF', 'TGFbeta', 'LPS', 'IFN10', 'YM201636']
    inflamm_conds = ['Ctrl', 'LPS', 'IFN10', 'YM201636']
    resolv_conds = ['Ctrl', 'MCSF', 'TGFbeta']
    old_conds = ['Ctrl', 'Old', 'OldMCSF']
    explant_conds = ['Ctrl', 'Explant 1', 'Explant 2']
    for condit in [all_chem_conds, inflamm_conds, resolv_conds]:
        GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'in_vivo', 'PCA', 'all', plot_acc_to_movie=True,
                                                     use_legend=True, use_features='all', v=2)
    GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'Old', 'PCA', old_conds, plot_acc_to_movie=True, v=2)
    GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'Explant', 'PCA', explant_conds, plot_acc_to_movie=True,
                                                 use_features='all', v=2)
    GeneratePlotsForPaper.perform_Rosenbaum_tests(MOVIE_PATH)

    GeneratePlotsForPaper.Fig5_LabellingSnapshots(MOVIE_PATH, v=2)

    # SUPPLEMENTARY DATA (plots or movies)
    GeneratePlotsForPaper.Supp_Fig1_image_intensity(MOVIE_PATH, v=2)

    GeneratePlotsForPaper.Supp_Movie_3(MOVIE_PATH, v=2)
    for method in ['TSNE', 'UMAP', 'LDA', 'NCA']:
        GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'in_vivo', method, all_chem_conds, v=2)
        GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'Old', method, old_conds, use_legend=False, v=2)
        GeneratePlotsForPaper.Fig3_4_DimensReduction(MOVIE_PATH, 'Explant', method, explant_conds, use_legend=False,
                                                     v=2)


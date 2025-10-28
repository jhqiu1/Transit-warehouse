import sys
from aad_catl_search_operators.utils.path_util import get_path
from aad_catl_search_operators.utils.plot_util import read_experiment, plot_algorithm_boxplots, plot_algorithm_bars, \
    color_list, marker_list, plot_front_dict, plot_time_lines, metri_cal
from aad_catl_search_operators.CATL_LLM.MyProblem import MyProblem
from _evaluate import MyEvaluation, generate_ref_point

if __name__ == '__main__':

    ########################################
    # todo: 读对比实验结果
    # experiment_name = str(get_path('results/device_level'))
    experiment_name = 'six_obj'
    experiment_path = str(get_path(f'aad_catl_search_operators/results/{experiment_name}'))
    print(experiment_path)

    obj_list = ['obj_1', 'obj_2', 'obj_3', 'obj_4', 'obj_5', 'obj_6']
    obj_pori_list = [5, 4, 3, 2, 1, 0]
    bat_no = 'PS_20250812141441'
    # problem = MyProblem(obj_list, bat_no)
    # [ref_point] = generate_ref_point(problem, 10)
    ref_point = [1.80000000e+01, 3.32049228e+04, 2.00000000e+01, 8.26656010e+08,
                 1.08000000e+02, 3.32181123e+03]
    train_F, train_data_dict, train_time_dict, searching_time = read_experiment(experiment_path, ref_point=ref_point,
                                                                                num=2)
    # test_F, test_data_dict, test_time_dict = read_experiment(experiment_path, ref_point=[7e8, 120.0], num=5)
    # test_20_F, test_20_data_dict, test_20_time_dict = read_experiment(experiment_path, ref_point=[7e9, 120.0], num=5)

    ########################################
    # todo: 不同算子定义不同颜色、形状
    color_map = {
        'NSGA': color_list[0],
        'NSGA+assignment_mutation': color_list[1],
        'NSGA+assignment_crossover': color_list[2],
        # 'NSGA+permutation_mutation': color_list[3],
        # 'NSGA+permutation_crossover': color_list[3],
        'NSGA+permutation_mutation': color_list[4]
        #
        # 'PLS': color_list[5],
        # 'PLS+assign_mut+test': color_list[6],
        # 'PLS+perm_mut+test': color_list[7]

        # 'NSGA+perm': color_list[0],  # 所有要对比算子的名称、颜色、记号
        # 'NSGA+perm+test': color_list[1],
    }

    marker_map = {
        'NSGA': marker_list[0],
        'NSGA+assignment_mutation': marker_list[1],
        'NSGA+assignment_crossover': marker_list[2],
        # 'NSGA+permutation_mutation': marker_list[3],
        # 'NSGA+permutation_crossover': marker_list[3],
        'NSGA+permutation_mutation': marker_list[4]
        # 'PLS': marker_list[6],
        # 'PLS+assign_mut+test': marker_list[4],
        # 'PLS+perm_mut+test': marker_list[5]
    }

    # 按照目标优先级筛选计算目标函数指标
    metri_cal(train_F, train_data_dict, searching_time, obj_pori_list)

    hv_combinations = [
        (train_data_dict, 'six_obj_0815'),
        # (test_data_dict, 'test_10'),
        # (test_20_data_dict, 'test_20'),

    ]

    PF_combinations = [
        (train_F, 'six_obj_0815'),
        # (test_F, 'test_10'),
        # (test_20_F, 'test_20')

    ]

    ########################################
    # todo: HV直方图

    for data_dict, data_selected in hv_combinations:
        plot_algorithm_bars(
            color_map=color_map,
            y_limit=(0, 0.2),
            data_dict=data_dict,
            bar_width=0.15,
            selected_algorithms=list(color_map.keys()),
            selected_data=[data_selected],
            save_path=f'{experiment_path}/hv_{data_selected}.png',
            title=f'Comparison of Algorithms',
            ylabel="Normalized HV",
            normalize_by_baseline=False,
            # normalized_baseline_name='NSGA+perm'  # 用于normalization的算子
        )

    ########################################
    # todo: Pareto Front
    # obj_list = ['obj_1', 'obj_2']
    #
    # for F, data_selected in PF_combinations:
    #     plot_front_dict(  # Pareto Front
    #         Front=F,
    #         selected_data=[data_selected],
    #         selected_algorithms=list(color_map.keys()),
    #         xlabel=obj_list[0],
    #         ylabel=obj_list[1],
    #         color_map=color_map,
    #         marker_map=marker_map,
    #         save_dir=experiment_path
    #     )

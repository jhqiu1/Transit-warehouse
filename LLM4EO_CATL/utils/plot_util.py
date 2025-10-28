import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import geatpy as ea

color_list = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]

marker_list = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "s",
    "*",
    "h",
    "x",
    "+",
]


def read_baseline(path, num, ref_point=[16000.0, 280.0]):
    ind_hv = HV(ref_point=ref_point)

    F = []
    F_joint = []
    time_list = []
    searching_time_list = []
    for i in range(num):
        filename = os.path.join(path, "run_{}.json".format(i))
        with open(filename) as fp:
            _data = json.load(fp)
            F.append(np.array(_data["F"][0]))
            time_list.append(os.path.getmtime(filename))
            searching_time_list.append(_data["time"])
            # F_joint.extend(np.array(_data['F'][0]))
    try:
        F_joint = np.concatenate(F, axis=0)
    except Exception as e:
        print(F)

    # hv = [ea.indicator.HV(np.array(_F).astype('float'), PF=np.array([ref_point]).astype('float')) for _F in F]
    # HV计算
    hv = []
    # 遍历 F 中的每个元素 _F
    for _F in F:
        # 将当前元素 _F 转为 float 类型的 NumPy 数组
        _F_float = np.array(_F).astype("float")
        # 将参考点 ref_point 转为 float 类型的 NumPy 数组（注意保持 PF 参数的维度）
        pf_array = np.array([ref_point]).astype("float")
        # 调用 HV 函数计算结果
        result = ea.indicator.HV(_F_float, PF=pf_array)
        # 将结果添加到列表 hv
        hv.append(result)

    return F_joint, hv, time_list, searching_time_list


def read_experiment(folder_name, ref_point=[16000.0, 280.0], num=11):
    F_data_dict = {}
    hv_data_dict = {}
    time_data_dict = {}
    searching_time = {}
    print(folder_name)
    for data_name in [
        f
        for f in os.listdir(folder_name)
        if os.path.isdir(os.path.join(folder_name, f))
    ]:
        for algo_name in [
            f
            for f in os.listdir(os.path.join(folder_name, data_name))
            if os.path.isdir(os.path.join(folder_name, data_name, f))
        ]:
            # algorithm_name = path_name.split('@')[1].split('+')[-1]
            # if 'baseline' not in algorithm_name:
            #     algorithm_name = algorithm_name[6:]

            path = os.path.join(folder_name, data_name, algo_name)
            print(path)

            F, hv, time_list, searching_time_list = read_baseline(path, num, ref_point)
            hv_data_dict.setdefault(data_name, {}).setdefault(algo_name, hv)
            F_data_dict.setdefault(data_name, {}).setdefault(algo_name, F)
            time_data_dict.setdefault(data_name, {}).setdefault(algo_name, time_list)
            searching_time.setdefault(data_name, {}).setdefault(
                algo_name, searching_time_list
            )
    return F_data_dict, hv_data_dict, time_data_dict, searching_time


def plot_algorithm_boxplots(
    data_dict,
    title="Comparison of Algorithms",
    ylabel="Normalized HV",
    figsize=(10, 6),
    selected_data=None,
    selected_algorithms=None,
    normalize_by_baseline=True,
    group_spacing=1.0,
    intra_spacing=0.15,
    color_map=None,
    save_path=None,
):
    """
    绘制嵌套字典格式的数据的 box plot，支持筛选和按 baseline 最小值归一化。绘制箱线图对比不同算法在各数据集上的HV分布，支持基线归一化（normalize_by_baseline）和自定义颜色/间距。

    Parameters:
    ◦ data_dict: Dict[str, Dict[str, List[float]]]

    ◦ selected_data: Optional[List[str]]

    ◦ selected_algorithms: Optional[List[str]]

    ◦ normalize_by_baseline: 是否以 baseline 的最小值归一化

    ◦ group_spacing: 每组（任务）之间的间距

    ◦ intra_spacing: 同一组内不同算法 box 的间距

    """

    all_data = list(data_dict.keys())
    data_x = selected_data if selected_data is not None else all_data

    all_algorithms = sorted({algo for task in data_dict.values() for algo in task})
    algorithms = (
        selected_algorithms if selected_algorithms is not None else all_algorithms
    )

    num_algorithms = len(algorithms)
    num_tasks = len(data_x)

    plt.figure(figsize=figsize)

    for i, algo in enumerate(algorithms):
        data_for_algo = []
        positions = []

        for j, task in enumerate(data_x):
            base_pos = j * group_spacing
            pos = base_pos + (i - (num_algorithms - 1) / 2) * intra_spacing
            positions.append(pos)

            data_values = data_dict.get(task, {})
            for key, _values in data_values.items():
                if algo in key:
                    values = _values
                    break

            if normalize_by_baseline:
                baseline_values = data_dict.get(task, {}).get("baseline", [])
                if baseline_values:
                    min_baseline = min(baseline_values)
                    if min_baseline != 0:
                        values = [v / min_baseline for v in values]

            data_for_algo.append(values)

        color = (
            color_map.get(algo, f"C{i}")  # 指定颜色或使用默认颜色序列
            if algo != "baseline"
            else "#1f77b4"  # baseline 固定为灰色
        )

        plt.boxplot(
            data_for_algo,
            positions=positions,
            widths=intra_spacing * 0.8,
            patch_artist=True,
            boxprops=dict(facecolor=color),
            medianprops=dict(color="black"),
            labels=[None] * len(positions),
        )

    # 设置 X 轴标签
    xtick_positions = [i * group_spacing for i in range(num_tasks)]
    plt.xticks(ticks=xtick_positions, labels=data_x)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 自定义图例
    legend_handles = [
        # Patch(facecolor=f"C{i}", label=algo) for i, algo in enumerate(algorithms)
        Patch(
            facecolor=color_map.get(algo, f"C{i}") if color_map else f"C{i}", label=algo
        )
        for i, algo in enumerate(algorithms)
    ]
    plt.legend(handles=legend_handles, title="Algorithms")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_algorithm_bars(
    data_dict,
    title="Comparison of Algorithms",
    ylabel="Avg Normalized HV",
    figsize=(10, 6),
    selected_data=None,
    selected_algorithms=None,
    normalize_by_baseline=True,
    group_spacing=1.0,
    bar_width=0.15,
    color_map=None,  # 新增参数：指定算法颜色，dict[str, str]
    save_path=None,
    y_limit=None,
    normalized_baseline_name=None,
):
    """
    绘制带误差的柱状图，支持筛选任务和算法、颜色自定义及baseline归一化。

    Parameters:
    ◦ data_dict: Dict[str, Dict[str, List[float]]]

    ◦ selected_data: Optional[List[str]]

    ◦ selected_algorithms: Optional[List[str]]

    ◦ normalize_by_baseline: 是否以 baseline 的最小值归一化

    ◦ group_spacing: 每组任务之间的间距

    ◦ bar_width: 每个柱子的宽度

    ◦ algorithm_colors: 可选，指定每个算法的颜色（dict[str, str]）

    """

    baseline_name = (
        "baseline" if normalized_baseline_name is None else normalized_baseline_name
    )

    all_data = list(data_dict.keys())
    data_x = selected_data if selected_data is not None else all_data

    all_algorithms = sorted({algo for task in data_dict.values() for algo in task})
    algorithms = (
        selected_algorithms if selected_algorithms is not None else all_algorithms
    )

    num_algorithms = len(algorithms)
    num_tasks = len(data_x)

    plt.figure(figsize=figsize)
    plt.rcParams.update(
        {
            "font.sans-serif": ["SimHei"],
            "axes.unicode_minus": False,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "figure.titlesize": 16,
        }
    )

    for i, algo in enumerate(algorithms):
        means = []
        stds = []
        positions = []

        for j, task in enumerate(data_x):
            base_pos = j * group_spacing
            pos = base_pos + (i - (num_algorithms - 1) / 2) * bar_width
            positions.append(pos)

            data_values = data_dict.get(task, {})
            for key, _values in data_values.items():
                if algo == key:
                    values = _values
                    break

            if normalize_by_baseline:
                baseline_values = data_dict.get(task, {}).get(baseline_name, [])
                if baseline_values:
                    min_baseline = np.mean(baseline_values)

                    # min_baseline = min(baseline_values)
                    if min_baseline != 0:
                        values = [v / min_baseline for v in values]
                    # print(f"{task}--{baseline_name}: {min_baseline}")
                    # print(f"{task}--{algo}: {np.mean(values)}\n")
                else:
                    raise ValueError("`baseline_values` is []")

            means.append(np.mean(values) if values else 0)
            stds.append(np.std(values) if values else 0)

        color = (
            color_map.get(algo, f"C{i}")  # 指定颜色或使用默认颜色序列
            if algo != "baseline"
            else "#1f77b4"  # baseline 固定为灰色
        )

        plt.bar(
            positions,
            means,
            yerr=stds,
            width=bar_width * 0.9,
            label=algo,
            color=color,
            capsize=4,
            edgecolor="black",
        )

        # 在柱子顶端添加 HV 均值
        for pos, mean in zip(positions, means):
            plt.text(
                pos, mean + 0.005, f"{mean:.5f}", ha="center", va="bottom", fontsize=12
            )

    xtick_positions = [i * group_spacing for i in range(num_tasks)]
    plt.xticks(ticks=xtick_positions, labels=data_x)
    plt.ylim(y_limit[0], y_limit[1])

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    # 添加图例
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_time_lines(
    data_dict,
    title="Comparison of Algorithms",
    ylabel="Avg Normalized HV",
    figsize=(10, 6),
    selected_data=None,
    selected_algorithms=None,
    color_map=None,  # 新增参数：指定算法颜色，dict[str, str]
    marker_map=None,
    save_path=None,
):
    """
    绘制带误差的柱状图，支持筛选任务和算法、颜色自定义及baseline归一化。

    Parameters:
    ◦ data_dict: Dict[str, Dict[str, List[float]]]

    ◦ selected_data: Optional[List[str]]

    ◦ selected_algorithms: Optional[List[str]]

    ◦ normalize_by_baseline: 是否以 baseline 的最小值归一化

    ◦ group_spacing: 每组任务之间的间距

    ◦ bar_width: 每个柱子的宽度

    ◦ algorithm_colors: 可选，指定每个算法的颜色（dict[str, str]）

    """

    group_spacing = 1.0

    all_data = list(data_dict.keys())
    data_x = selected_data if selected_data is not None else all_data

    all_algorithms = sorted({algo for task in data_dict.values() for algo in task})
    algorithms = (
        selected_algorithms if selected_algorithms is not None else all_algorithms
    )

    num_algorithms = len(algorithms)
    num_tasks = len(data_x)

    plt.figure(figsize=figsize)
    plt.rcParams.update(
        {
            "font.sans-serif": ["SimHei"],
            "axes.unicode_minus": False,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "figure.titlesize": 16,
        }
    )

    x_ticks = list(range(len(data_x)))

    for i, algo in enumerate(algorithms):
        means = []
        stds = []
        positions = []

        for j, task in enumerate(data_x):

            data_values = data_dict.get(task, {})
            for key, _values in data_values.items():
                if algo == key:
                    values = _values
                    break

            mean_value = (max(values) - min(values)) / (len(values) - 1)
            means.append(mean_value)

        color = (
            color_map.get(algo, f"C{i}")  # 指定颜色或使用默认颜色序列
            if algo != "baseline"
            else "#1f77b4"  # baseline 固定为灰色
        )

        marker = marker_map.get(algo)

        plt.plot(
            x_ticks,
            means,
            label=algo,
            color=color,
            marker=marker,
        )

    xtick_positions = [i * group_spacing for i in range(num_tasks)]
    plt.xticks(ticks=xtick_positions, labels=data_x)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_front_dict(
    Front,
    selected_data=None,
    selected_algorithms=None,
    xlabel="Objective 1",
    ylabel="Objective 2",
    save_dir=None,
    color_map=None,
    marker_map=None,
):
    # 确保 save_dir 存在
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for data_name, algo_dict in Front.items():
        if selected_data is not None and data_name not in selected_data:
            continue  # 跳过未选择的数据

        plt.figure(figsize=(6, 5))
        for algo_name, points in algo_dict.items():
            if selected_algorithms is not None and algo_name not in selected_algorithms:
                continue  # 跳过未选择的算法

            color = color_map.get(algo_name, None) if color_map else None
            marker = marker_map.get(algo_name, None) if marker_map else "o"
            # plt.scatter(points[:, 0], points[:, 1], label=algo_name, alpha=1, edgecolors=color, marker=marker, facecolors='none', s=50)
            plt.scatter(
                points[:, 0],
                points[:, 1],
                label=algo_name,
                alpha=0.5,
                color=color,
                marker=marker,
                s=50,
            )

        plt.title(f"Scatter Plot for {data_name}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f"{data_name}_scatter.png")
            print(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def metri_cal(
    train_F,
    train_data_dict,
    train_time_dict,
    obj_pori_list,
    operators_path,
    hv_ref_point=None,
):
    obj_best_results = {}
    time_avg_results = {}
    hv_results = {}  # 存储各算法的HV值

    for exp in train_F:
        for alg in train_F[exp]:
            # 获取每个算法的解集
            alg_result = train_F[exp][alg]

            # 计算HV值（如果提供了参考点）
            if hv_ref_point is not None:
                try:
                    # 使用geatpy的HV计算方式
                    pf_array = np.array([hv_ref_point]).astype("float")
                    hv_value = ea.indicator.HV(
                        np.array(alg_result).astype("float"), PF=pf_array
                    )
                    hv_results[alg] = hv_value
                except Exception as e:
                    print(f"计算HV时出错（算法{alg}）：{str(e)}")
                    hv_results[alg] = np.nan

            # 获取最优解（按优先级排序）
            sort_keys = [-alg_result[:, i] for i in obj_pori_list]
            sorted_indices = np.lexsort(sort_keys)
            sorting_objs = alg_result[sorted_indices]
            obj_best_results[alg] = sorting_objs[0]

            # 计算平均时间
            time_data = train_time_dict[exp][alg]
            time_avg_results[alg] = np.mean(time_data)

        # 创建DataFrame
        data = []
        for method in obj_best_results.keys():
            row = {
                "Method": method,
                "HV": hv_results.get(method, np.nan),
            }

            # 添加各目标值
            for i, obj_value in enumerate(obj_best_results[method]):
                row[f"Obj{i+1}"] = obj_value

            # 添加时间
            row["Time"] = time_avg_results[method]

            data.append(row)

        df = pd.DataFrame(data)
        columns = (
            ["Method", "HV"]
            + [f"Obj{i+1}" for i in range(len(obj_pori_list))]
            + ["Time"]
        )
        df = df[columns]

        # 导出到Excel
        df.to_excel(
            os.path.join(operators_path, f"{exp}_optimization_results.xlsx"),
            index=False,
            float_format="%.4f",
        )

    print("Excel文件已成功导出")


def non_dominated_sort(solutions):
    """
    非支配排序算法
    :param solutions: 二维数组，每行是一个解
    :return: 第一层非支配解（帕累托前沿）
    """
    n = len(solutions)
    # 存储每个解被支配的次数
    domination_count = np.zeros(n, dtype=int)
    # 存储每个解支配的其他解
    dominated_solutions = [[] for _ in range(n)]
    # 存储第一层非支配解
    pareto_front = []

    # 第一遍遍历：计算支配关系
    for i in range(n):
        for j in range(i + 1, n):
            # 比较解i和解j
            dominates = True
            dominated = True

            # 检查i是否支配j
            for k in range(solutions.shape[1]):
                if solutions[i, k] > solutions[j, k]:
                    dominates = False
                if solutions[j, k] > solutions[i, k]:
                    dominated = False

            if dominates and not dominated:
                domination_count[j] += 1
                dominated_solutions[i].append(j)
            elif dominated and not dominates:
                domination_count[i] += 1
                dominated_solutions[j].append(i)

    # 找出第一层非支配解（支配计数为0）
    for i in range(n):
        if domination_count[i] == 0:
            pareto_front.append(solutions[i])

    return np.array(pareto_front)


if __name__ == "__main__":
    from utils.path_util import get_path

    experiment_name = "device_level"
    experiment_path = str(get_path(f"results/{experiment_name}"))
    print(experiment_path)
    F, data_dict, time_dict = read_experiment(
        experiment_path, ref_point=[10000.0, 10000.0], num=3
    )
    print(F)

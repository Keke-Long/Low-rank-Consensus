import os
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

# 结果存储的 CSV 文件路径
SPEARMAN_CSV_PATH = "./Nuscenes/results/1 spearman/spearman_results.csv"

# **所有模型名称**
MODEL_NAMES = ["Qwen2-VL-7B-Instruct", "llava-1.5-7b-hf", "deepseek-vl2-tiny"]

def calculate_spearman_for_model(model_name, df):
    """
    计算单个模型的 hallucination_check 与 ResidualError 之间的 Spearman 相关性
    """
    # 筛选当前模型数据
    df_model = df[df["Model"] == model_name].copy()

    # **去除 NaN 值**
    df_model = df_model.dropna(subset=["hallucination_check", "ResidualError"])
    if df_model.empty:
        return None  # 如果去掉 NaN 后没数据，返回 None

    hallucination_ratios = []
    residual_errors = df_model["ResidualError"].tolist()

    for hallucination_str in df_model["hallucination_check"]:
        if isinstance(hallucination_str, float) and np.isnan(hallucination_str):
            continue  # 跳过 NaN 值
        if not isinstance(hallucination_str, str):
            continue  # 跳过非字符串值

        hallucination_list = hallucination_str.split(", ")
        total_sentences = len(hallucination_list)
        yes_count = sum(1 for entry in hallucination_list if "Yes" in entry)
        hallucination_ratio = yes_count / total_sentences if total_sentences > 0 else 0
        hallucination_ratios.append(hallucination_ratio)

    # **检查数据是否是常数数组**
    if len(set(hallucination_ratios)) == 1 or len(set(residual_errors)) == 1:
        return None  # 相关性无法计算

    spearman_corr, _ = spearmanr(hallucination_ratios, residual_errors)
    return spearman_corr


def calculate_spearman_multi3(df):
    """
    计算所有 3 个模型的数据合并后的 Spearman 相关性
    """
    # 筛选所有 3 个模型的数据
    df_multi = df[df["Model"].isin(MODEL_NAMES)].copy()
    
    # **去除 NaN 值**
    df_multi = df_multi.dropna(subset=["hallucination_check", "ResidualError_multi_agent"])
    if df_multi.empty:
        return None  # 如果去掉 NaN 后没数据，返回 None

    hallucination_ratios = []
    residual_errors = df_multi["ResidualError_multi_agent"].tolist()

    for hallucination_str in df_multi["hallucination_check"]:
        if isinstance(hallucination_str, float) and np.isnan(hallucination_str):
            continue  # 跳过 NaN 值
        if not isinstance(hallucination_str, str):
            continue  # 跳过非字符串值

        hallucination_list = hallucination_str.split(", ")
        total_sentences = len(hallucination_list)
        yes_count = sum(1 for entry in hallucination_list if "Yes" in entry)
        hallucination_ratio = yes_count / total_sentences if total_sentences > 0 else 0
        hallucination_ratios.append(hallucination_ratio)

    # **检查数据是否是常数数组**
    if len(set(hallucination_ratios)) == 1 or len(set(residual_errors)) == 1:
        return None  # 相关性无法计算

    spearman_corr, _ = spearmanr(hallucination_ratios, residual_errors)
    return spearman_corr


def save_spearman_results(model_results_dict, multi3_results):
    """
    将多个模型和 Spearman_multi3 结果存储到 CSV 文件中
    """
    # 构建基础 DataFrame
    rows = []
    for scene_id, model_corrs in model_results_dict.items():
        row = {"Scene_ID": scene_id}
        for model_name in MODEL_NAMES:
            spearman_val = model_corrs.get(model_name, 0)
            if spearman_val is None:
                spearman_val = 0
            row[f"Spearman_{model_name}"] = spearman_val
        rows.append(row)
    
    df_model = pd.DataFrame(rows)

    # 添加 Spearman_multi3 数据
    df_multi3 = pd.DataFrame(multi3_results, columns=["Scene_ID", "Spearman_multi3"])

    # 合并两部分数据
    df_final = pd.merge(df_model, df_multi3, on="Scene_ID", how="outer")

    # 如果 CSV 已经存在，进行合并处理
    if os.path.exists(SPEARMAN_CSV_PATH):
        df_existing = pd.read_csv(SPEARMAN_CSV_PATH)
        df_final = pd.merge(df_existing, df_final, on="Scene_ID", how="outer", suffixes=("_old", ""))
        
        # 只保留最新的数据列
        expected_columns = ["Scene_ID"] + [f"Spearman_{model}" for model in MODEL_NAMES] + ["Spearman_multi3"]
        df_final = df_final[expected_columns]

    # 替换 NaN 为 0
    df_final.fillna(0, inplace=True)

    df_final.to_csv(SPEARMAN_CSV_PATH, index=False)
    print(f"Spearman results saved to {SPEARMAN_CSV_PATH}")



# **示例调用**
if __name__ == "__main__":
    results_caption_dir = "./Nuscenes/results_caption/"
    output_dir = "./Nuscenes/results/1 spearman/"

    model_results = {}
    multi3_results = []
    required_columns = ["hallucination_check", "ResidualError", "ResidualError_multi_agent", "Model"]

    for filename in sorted(os.listdir(results_caption_dir)):
        if filename.endswith(".csv"):
            scene_id = os.path.splitext(filename)[0]
            csv_path = os.path.join(results_caption_dir, filename)
            df = pd.read_csv(csv_path)

            # 检查必要的列是否齐全
            if not all(col in df.columns for col in required_columns):
                print(f"[Warning] Skipped {filename} due to missing required columns.")
                continue

            # 计算单个模型的 Spearman 相关性
            model_corrs = {}
            for model in MODEL_NAMES:
                corr = calculate_spearman_for_model(model, df)
                model_corrs[model] = corr if corr is not None else 0.0

            model_results[scene_id] = model_corrs

            # 计算 Spearman_multi3
            multi3_corr = calculate_spearman_multi3(df)
            multi3_corr = multi3_corr if multi3_corr is not None else 0.0
            multi3_results.append((scene_id, multi3_corr))

    save_spearman_results(model_results, multi3_results)



def plot_spearman_histograms(output_path):
    """
    读取 spearman_results.csv，并绘制所有 Spearman 相关性的直方图，标注均值、方差等统计信息
    """
    df_spearman = pd.read_csv(SPEARMAN_CSV_PATH)

    # 需要绘制的列（排除 Scene_ID）
    spearman_columns = [col for col in df_spearman.columns if col != "Scene_ID"]

    # 创建子图网格
    num_columns = len(spearman_columns)
    rows = (num_columns + 2) // 3  # 计算需要多少行
    cols = min(num_columns, 3)  # 最多 3 列

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = axes.flatten()  # 把子图对象变成列表，方便索引

    # 遍历每一列数据，绘制直方图
    for i, col in enumerate(spearman_columns):
        ax = axes[i]
        data = df_spearman[col].dropna()  # 去掉 NaN 值

        if data.empty:
            ax.text(0.5, 0.5, "No Data", fontsize=12, ha="center", va="center")
            ax.set_title(col)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # 计算统计量
        mean_val = data.mean()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()
        median_val = data.median()

        # 绘制直方图
        ax.hist(data, bins=20, color="blue", edgecolor="black", alpha=0.7)

        # 标注统计信息
        stats_text = (
            f"Mean: {mean_val:.3f}\n"
            f"Std: {std_val:.3f}\n"
            f"Min: {min_val:.3f}\n"
            f"Max: {max_val:.3f}\n"
            f"Median: {median_val:.3f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", horizontalalignment="right",
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        ax.set_title(col)
        ax.set_xlabel("Spearman Correlation")
        ax.set_ylabel("Frequency")

    # 移除多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Spearman 直方图已保存到 {output_path}")


# 运行绘图函数
output_path = "./Nuscenes/results/1 spearman/spearman_results.png"
plot_spearman_histograms(output_path)
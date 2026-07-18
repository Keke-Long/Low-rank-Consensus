import os
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

# 所有模型名称
MODEL_NAMES = ["Qwen2-VL-7B-Instruct", "llava-1.5-7b-hf", "deepseek-vl2-tiny"]

# 要分析的rank值
RANKS_TO_ANALYZE = [3, 5, 7]

def calculate_spearman_for_model_and_rank(model_name, df, rank):
    """
    计算单个模型在特定rank下的 hallucination_check 与 ResidualError 之间的 Spearman 相关性
    """
    # 筛选当前模型数据
    df_model = df[df["Model"] == model_name].copy()
    
    # 构建对应rank的列名
    residual_col = f"ResidualError_r={rank}"
    
    # 检查列是否存在
    if residual_col not in df_model.columns:
        return None
    
    # 去除 NaN 值
    df_model = df_model.dropna(subset=["hallucination_check", residual_col])
    if df_model.empty:
        return None

    hallucination_ratios = []
    residual_errors = df_model[residual_col].tolist()

    for hallucination_str in df_model["hallucination_check"]:
        if isinstance(hallucination_str, float) and np.isnan(hallucination_str):
            continue
        if not isinstance(hallucination_str, str):
            continue

        hallucination_list = hallucination_str.split(", ")
        total_sentences = len(hallucination_list)
        yes_count = sum(1 for entry in hallucination_list if "Yes" in entry)
        hallucination_ratio = yes_count / total_sentences if total_sentences > 0 else 0
        hallucination_ratios.append(hallucination_ratio)

    # 检查数据是否是常数数组
    if len(set(hallucination_ratios)) == 1 or len(set(residual_errors)) == 1:
        return 0.0  # 常数数组时返回0

    spearman_corr, _ = spearmanr(hallucination_ratios, residual_errors)
    return spearman_corr

def calculate_spearman_multi_agent_and_rank(df, rank):
    """
    计算多模型在特定rank下的 Spearman 相关性
    """
    # 筛选所有模型的数据
    df_multi = df[df["Model"].isin(MODEL_NAMES)].copy()
    
    # 构建对应rank的列名
    residual_col = f"ResidualError_multi_agent_r={rank}"
    
    # 检查列是否存在
    if residual_col not in df_multi.columns:
        return None
    
    # 去除 NaN 值
    df_multi = df_multi.dropna(subset=["hallucination_check", residual_col])
    if df_multi.empty:
        return None

    hallucination_ratios = []
    residual_errors = df_multi[residual_col].tolist()

    for hallucination_str in df_multi["hallucination_check"]:
        if isinstance(hallucination_str, float) and np.isnan(hallucination_str):
            continue
        if not isinstance(hallucination_str, str):
            continue

        hallucination_list = hallucination_str.split(", ")
        total_sentences = len(hallucination_list)
        yes_count = sum(1 for entry in hallucination_list if "Yes" in entry)
        hallucination_ratio = yes_count / total_sentences if total_sentences > 0 else 0
        hallucination_ratios.append(hallucination_ratio)

    # 检查数据是否是常数数组
    if len(set(hallucination_ratios)) == 1 or len(set(residual_errors)) == 1:
        return 0.0

    spearman_corr, _ = spearmanr(hallucination_ratios, residual_errors)
    return spearman_corr

def plot_single_model_correlation(model_name, correlations, output_dir):
    """
    绘制单个模型的最优相关性系数分布（排除=0的情况）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建单个图，尺寸稍小
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    
    if not correlations or len(correlations) == 0:
        ax.text(0.5, 0.5, "No Data", fontsize=12, ha="center", va="center")
        ax.set_title(f"Optimal_Spearman_{model_name}")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # 排除相关性系数=0的情况
        data = np.array([x for x in correlations if x != 0.0])
        
        if len(data) == 0:
            ax.text(0.5, 0.5, "No Non-Zero Data", fontsize=12, ha="center", va="center")
            ax.set_title(f"Optimal_Spearman_{model_name}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # 计算统计量
            mean_val = data.mean()
            std_val = data.std()
            min_val = data.min()
            max_val = data.max()
            median_val = np.median(data)
            
            # 绘制直方图
            ax.hist(data, bins=20, color="mediumaquamarine", edgecolor="black", alpha=0.7)
            
            # 标注统计信息
            stats_text = (
                f"Mean: {mean_val:.3f}\n"
                f"Std: {std_val:.3f}"
            )
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment="top", horizontalalignment="left",
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            
            ax.set_xlabel("Spearman Correlation")
            ax.set_ylabel("Frequency")
    
    # 保存图片
    output_path = os.path.join(output_dir, f"optimal_spearman_{model_name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存 {model_name} 的相关性分布图到 {output_path}")

def plot_multi_model_correlation(correlations, output_dir):
    """
    绘制多模型的最优相关性系数分布（排除=0的情况）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建单个图，尺寸稍小
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    
    if not correlations or len(correlations) == 0:
        ax.text(0.5, 0.5, "No Data", fontsize=12, ha="center", va="center")
        ax.set_title("Optimal_Spearman_multi3")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # 排除相关性系数=0的情况
        data = np.array([x for x in correlations if x != 0.0])
        
        if len(data) == 0:
            ax.text(0.5, 0.5, "No Non-Zero Data", fontsize=12, ha="center", va="center")
            ax.set_title("Optimal_Spearman_multi3")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # 计算统计量
            mean_val = data.mean()
            std_val = data.std()
            min_val = data.min()
            max_val = data.max()
            median_val = np.median(data)
            
            # 绘制直方图
            ax.hist(data, bins=20, color="mediumaquamarine", edgecolor="black", alpha=0.7)
            
            # 标注统计信息
            stats_text = (
                f"Mean: {mean_val:.3f}\n"
                f"Std: {std_val:.3f}"
            )
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment="top", horizontalalignment="left",
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            
            ax.set_xlabel("Spearman Correlation")
            ax.set_ylabel("Frequency")
    
    # 保存图片
    output_path = os.path.join(output_dir, "optimal_spearman_multi3.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存多模型的相关性分布图到 {output_path}")

def plot_all_correlation_distributions(csv_best_correlations_by_model, csv_best_multi_correlations, output_dir):
    """
    分别绘制所有模型的最优相关性系数分布
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个单模型绘制图表
    for model_name in MODEL_NAMES:
        correlations = csv_best_correlations_by_model.get(model_name, [])
        plot_single_model_correlation(model_name, correlations, output_dir)
    
    # 绘制多模型图表
    plot_multi_model_correlation(csv_best_multi_correlations, output_dir)

def save_detailed_results(detailed_results, output_path):
    """
    保存详细的结果到CSV文件，小数保留4位
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame(detailed_results)
    df = df.sort_values('Scene_ID')  # 按Scene_ID排序
    
    # 将所有数值列格式化为4位小数
    for col in df.columns:
        if col != 'Scene_ID' and not col.endswith('_rank'):
            df[col] = df[col].round(4)
    
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"详细结果已保存到 {output_path}")

if __name__ == "__main__":
    print("开始分析不同rank的最优Spearman相关性...")
    print(f"分析的rank值: {RANKS_TO_ANALYZE}")
    
    results_caption_dir = "./Nuscenes/results_caption/"
    
    # 存储每个模型的最佳相关性
    csv_best_correlations_by_model = {model: [] for model in MODEL_NAMES}
    csv_best_multi_correlations = []
    
    # 存储详细结果
    detailed_results = []
    
    processed_files = 0
    
    csv_files = sorted([f for f in os.listdir(results_caption_dir) if f.endswith(".csv")])
    
    for filename in csv_files:
        scene_id = os.path.splitext(filename)[0]
        csv_path = os.path.join(results_caption_dir, filename)
        df = pd.read_csv(csv_path)
        
        # 检查是否包含hallucination_check列
        if "hallucination_check" not in df.columns:
            print(f"[Warning] Skipped {filename} - missing hallucination_check column")
            continue
        
        # 初始化结果行
        result_row = {'Scene_ID': scene_id}
        
        # 为每个模型计算所有rank的相关性
        for model in MODEL_NAMES:
            model_correlations = {}
            
            # 计算每个rank的相关性
            for rank in RANKS_TO_ANALYZE:
                corr = calculate_spearman_for_model_and_rank(model, df, rank)
                if corr is not None:
                    model_correlations[rank] = corr
                else:
                    model_correlations[rank] = 0.0
                
                # 添加到结果行
                result_row[f'Spearman_{model}_r={rank}'] = model_correlations[rank]
            
            # 选择相关性最大的rank
            if model_correlations:
                best_rank = max(model_correlations.keys(), key=lambda r: model_correlations[r])
                best_corr = model_correlations[best_rank]
                csv_best_correlations_by_model[model].append(best_corr)
                
                # 添加最佳结果
                result_row[f'Spearman_{model}_best'] = best_corr
                result_row[f'Spearman_{model}_best_rank'] = best_rank
        
        # 为multi-model计算所有rank的相关性
        multi_correlations = {}
        for rank in RANKS_TO_ANALYZE:
            multi_corr = calculate_spearman_multi_agent_and_rank(df, rank)
            if multi_corr is not None:
                multi_correlations[rank] = multi_corr
            else:
                multi_correlations[rank] = 0.0
            
            # 添加到结果行
            result_row[f'Spearman_multi3_r={rank}'] = multi_correlations[rank]
        
        # 选择multi-model的最佳rank
        if multi_correlations:
            best_rank = max(multi_correlations.keys(), key=lambda r: multi_correlations[r])
            best_corr = multi_correlations[best_rank]
            csv_best_multi_correlations.append(best_corr)
            
            # 添加最佳结果
            result_row[f'Spearman_multi3_best'] = best_corr
            result_row[f'Spearman_multi3_best_rank'] = best_rank
        
        detailed_results.append(result_row)
        processed_files += 1
    
    print(f"\n处理了 {processed_files} 个CSV文件")
    
    # 保存详细结果到CSV
    output_csv_path = "./Nuscenes/results/1 spearman/detailed_rank_results.csv"
    save_detailed_results(detailed_results, output_csv_path)
    
    # 统计每个模型的结果（包括排除0值后的统计）
    print(f"\n=== 结果统计 ===")
    for model in MODEL_NAMES:
        correlations = csv_best_correlations_by_model[model]
        if correlations:
            non_zero_correlations = [x for x in correlations if x != 0.0]
            print(f"{model}: {len(correlations)} 个场景")
            print(f"  全部 - 均值: {np.mean(correlations):.4f}, 最大值: {np.max(correlations):.4f}, 最小值: {np.min(correlations):.4f}")
            if non_zero_correlations:
                print(f"  非零 - 数量: {len(non_zero_correlations)}, 均值: {np.mean(non_zero_correlations):.4f}, 最大值: {np.max(non_zero_correlations):.4f}, 最小值: {np.min(non_zero_correlations):.4f}")
    
    if csv_best_multi_correlations:
        non_zero_multi = [x for x in csv_best_multi_correlations if x != 0.0]
        print(f"Multi-model: {len(csv_best_multi_correlations)} 个场景")
        print(f"  全部 - 均值: {np.mean(csv_best_multi_correlations):.4f}, 最大值: {np.max(csv_best_multi_correlations):.4f}, 最小值: {np.min(csv_best_multi_correlations):.4f}")
        if non_zero_multi:
            print(f"  非零 - 数量: {len(non_zero_multi)}, 均值: {np.mean(non_zero_multi):.4f}, 最大值: {np.max(non_zero_multi):.4f}, 最小值: {np.min(non_zero_multi):.4f}")
    
    # 绘制所有图表（分别保存）
    output_plots_dir = "./Nuscenes/results/1 spearman/"
    plot_all_correlation_distributions(csv_best_correlations_by_model, csv_best_multi_correlations, output_plots_dir)
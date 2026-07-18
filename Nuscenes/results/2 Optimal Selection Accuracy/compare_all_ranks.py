import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_NAMES = ["deepseek-vl2-tiny", "Qwen2-VL-7B-Instruct", "llava-1.5-7b-hf"]
RANKS = [3, 5, 7]

def evaluate_single_file_with_rank(csv_path, model_name, rank):
    """
    计算单个模型特定rank的分类指标（Accuracy, Precision, Recall, F1 Score）。
    """
    df = pd.read_csv(csv_path)

    # **筛选当前模型数据**
    df_model = df[df["Model"] == model_name]

    # **检查必要列**
    residual_col = f"ResidualError_r={rank}"
    if "hallucination_check" not in df_model.columns or residual_col not in df_model.columns:
        print(f"⚠️ Missing necessary columns for model {model_name}, rank {rank} in file {csv_path}.")
        return None

    # **去除 NaN 值**
    df_model = df_model.dropna(subset=["hallucination_check", residual_col])
    if df_model.empty:
        return None

    # **根据对应rank的ResidualError排序，选择最小ResidualError对应的Caption**
    df_sorted = df_model.sort_values(by=residual_col)
    best_caption_row = df_sorted.iloc[0]

    # **处理 hallucination_check，防止 NaN**
    hallucination_str = str(best_caption_row["hallucination_check"])  # 转换为字符串
    hallucination_list = hallucination_str.split(", ")

    # **获取真实标签和预测标签**
    true_label = 1 if any("Yes" in entry for entry in hallucination_list) else 0
    pred_label = 0  # 预测 ResidualError 最小的 Caption 没有幻觉

    # 计算分类指标
    accuracy = accuracy_score([true_label], [pred_label])
    precision = precision_score([true_label], [pred_label], zero_division=0)
    recall = recall_score([true_label], [pred_label], zero_division=0)
    f1 = f1_score([true_label], [pred_label], zero_division=0)

    return {
        "File": os.path.basename(csv_path),
        f"Accuracy_{model_name}_r={rank}": accuracy,
        f"Precision_{model_name}_r={rank}": precision,
        f"Recall_{model_name}_r={rank}": recall,
        f"F1_Score_{model_name}_r={rank}": f1
    }

def evaluate_multi_agent_with_rank(csv_path, rank):
    """
    计算 multi-agent 特定rank的结果（所有模型取前 5 行，但只选其中对应rank的ResidualError最小的句子）。
    """
    df = pd.read_csv(csv_path)

    # **检查必要列**
    residual_col = f"ResidualError_multi_agent_r={rank}"
    if "hallucination_check" not in df.columns or residual_col not in df.columns:
        print(f"⚠️ Missing necessary columns for rank {rank} in file {csv_path}.")
        return None

    # **去除 NaN**
    df = df.dropna(subset=["hallucination_check", residual_col])
    if df.empty:
        return None

    # **取所有模型的前 5 行数据**
    df_selected = df.sort_values(by=residual_col).groupby("Model").head(5)

    # **从选出的 Caption 中，选择对应rank的ResidualError最小的一句**
    best_caption_row = df_selected.sort_values(by=residual_col).iloc[0]

    # **处理 hallucination_check，防止 NaN**
    hallucination_str = str(best_caption_row["hallucination_check"])
    hallucination_list = hallucination_str.split(", ")

    # **真实标签**
    true_label = 1 if any("Yes" in entry for entry in hallucination_list) else 0

    # **预测标签（默认 ResidualError 最小的 Caption 无幻觉）**
    pred_label = 0

    # **计算 Accuracy**
    accuracy = accuracy_score([true_label], [pred_label])
    precision = precision_score([true_label], [pred_label], zero_division=0)
    recall = recall_score([true_label], [pred_label], zero_division=0)
    f1 = f1_score([true_label], [pred_label], zero_division=0)

    return {
        "File": os.path.basename(csv_path),
        f"Accuracy_multi_agent_r={rank}": accuracy,
        f"Precision_multi_agent_r={rank}": precision,
        f"Recall_multi_agent_r={rank}": recall,
        f"F1_Score_multi_agent_r={rank}": f1
    }


def evaluate_all_files(results_dir, output_dir):
    """
    遍历所有 CSV 文件，计算所有模型在不同rank下的分类指标 + multi-agent 结果，并保存到 CSV。
    """
    sorted_files = sorted([f for f in os.listdir(results_dir) if f.endswith(".csv")])

    all_results = []

    for filename in sorted_files:
        csv_path = os.path.join(results_dir, filename)
        result = {"File": filename}

        # **计算单个模型在不同rank下的指标**
        for model in MODEL_NAMES:
            for rank in RANKS:
                model_result = evaluate_single_file_with_rank(csv_path, model, rank)
                if model_result:
                    result.update(model_result)

        # **计算 multi-agent 在不同rank下的结果**
        for rank in RANKS:
            multi_result = evaluate_multi_agent_with_rank(csv_path, rank)
            if multi_result:
                result.update(multi_result)

        all_results.append(result)

    # **转换为 DataFrame**
    df_results = pd.DataFrame(all_results)

    # **计算平均指标**
    avg_metrics = df_results.drop(columns=["File"]).mean().to_frame().T
    avg_metrics.insert(0, "File", "Average")

    # **合并平均值**
    df_results = pd.concat([df_results, avg_metrics], ignore_index=True)

    # **保存结果**
    output_path = os.path.join(output_dir, "classification_metrics_all_models_all_ranks.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")

    # **打印结果**
    print("\nClassification Metrics for All Ranks:")
    print(df_results)

    # **生成分rank的汇总统计**
    generate_rank_summary(df_results, output_dir)


def generate_rank_summary(df_results, output_dir):
    """
    生成按rank分组的汇总统计
    """
    avg_row = df_results[df_results["File"] == "Average"]
    
    summary_data = []
    
    for rank in RANKS:
        rank_summary = {"Rank": rank}
        
        # 单模型指标汇总
        for model in MODEL_NAMES:
            rank_summary[f"{model}_Accuracy"] = avg_row[f"Accuracy_{model}_r={rank}"].iloc[0] if f"Accuracy_{model}_r={rank}" in avg_row.columns else None
        
        # Multi-agent指标汇总
        rank_summary["MultiAgent_Accuracy"] = avg_row[f"Accuracy_multi_agent_r={rank}"].iloc[0] if f"Accuracy_multi_agent_r={rank}" in avg_row.columns else None
        
        summary_data.append(rank_summary)
    
    # **保存rank汇总**
    summary_df = pd.DataFrame(summary_data)
    summary_output_path = os.path.join(output_dir, "classification_metrics_summary.csv")
    summary_df.to_csv(summary_output_path, index=False)
    print(f"\n✅ Rank summary saved to {summary_output_path}")
    
    print("\nRank Summary:")
    print(summary_df)


# **示例调用**
if __name__ == "__main__":
    results_caption_dir = "./Nuscenes/results_caption/"
    output_dir = "./Nuscenes/results/2 Optimal Selection Accuracy/"
    os.makedirs(output_dir, exist_ok=True)

    # **计算所有模型 + multi-agent 在所有rank下的分类指标**
    evaluate_all_files(results_caption_dir, output_dir)
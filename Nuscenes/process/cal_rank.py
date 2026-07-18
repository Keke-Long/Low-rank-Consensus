'''
Calculate residual using SVD with multiple rank values (完全不改变原有顺序)
'''

import numpy as np
import os
import csv
import torch
import gc
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

# 计算答案的句子嵌入
def compute_embeddings(answers):
    """
    计算句子的嵌入，并在计算后释放显存
    """
    model = SentenceTransformer('intfloat/e5-large-v2', device="cpu")
    model.to("cuda")  # 计算时切换到 GPU
    embeddings = model.encode(answers, convert_to_numpy=True)
    return embeddings

# 对指定rank进行低秩近似
def low_rank_approximation_with_rank(embeddings, rank):
    """
    使用指定rank进行低秩近似
    """
    svd = TruncatedSVD(n_components=rank, random_state=42)
    low_rank_matrix = svd.fit_transform(embeddings)
    reconstructed = svd.inverse_transform(low_rank_matrix)
    
    # 解释方差比例
    explained_variance = svd.explained_variance_ratio_.sum()
    
    return reconstructed, explained_variance

# 计算残差（误差部分）
def calculate_residuals(original, reconstructed):
    residuals = original - reconstructed
    residual_norms = np.linalg.norm(residuals, axis=1)  # 计算每个答案的 L2 范数
    return residuals, residual_norms

# 清空所有ResidualError开头的列
def clean_residual_columns(rows, fieldnames):
    """
    删除所有以ResidualError开头的列（包括列名和数据）
    
    Parameters:
    - rows: CSV数据行
    - fieldnames: 字段名列表
    
    Returns:
    - cleaned_fieldnames: 清理后的字段名列表（不包含ResidualError相关列）
    """
    # 找出所有以ResidualError开头的列
    residual_columns = [col for col in fieldnames if col.startswith('ResidualError')]
    
    # 从fieldnames中移除这些列
    cleaned_fieldnames = [col for col in fieldnames if not col.startswith('ResidualError')]
    
    # 从每行数据中删除这些列的数据
    for row in rows:
        for col in residual_columns:
            if col in row:
                del row[col]
    
    if residual_columns:
        print(f"  Deleted historical columns: {residual_columns}")
    
    return cleaned_fieldnames

# 处理单个模型（不包含文件读写，不改变顺序）
def process_single_model(rows, model_name, candidate_ranks):
    """
    处理单个模型的残差计算，计算所有rank的结果，完全不改变顺序
    
    Returns:
    - all_residuals: 字典，key为rank，value为另一个字典（行索引->残差值）
    """
    # 提取模型结果及其原始索引
    model_data = [(i, row) for i, row in enumerate(rows) if row['Model'] == model_name]
    
    if not model_data:
        print(f"    No data found for model {model_name}")
        return {}
    
    indices, model_rows = zip(*model_data)
    model_results = [row['Caption'] for row in model_rows]
    
    # 计算嵌入
    embeddings = compute_embeddings(model_results)
    
    # 对每个候选rank计算残差
    all_residuals = {}
    max_possible_rank = min(embeddings.shape[0] - 1, embeddings.shape[1])
    
    for rank in candidate_ranks:
        if rank > max_possible_rank:
            print(f"      Rank {rank} exceeds maximum possible rank {max_possible_rank}, skipping")
            continue
            
        # 低秩近似
        reconstructed, explained_variance = low_rank_approximation_with_rank(embeddings, rank)
        
        # 计算残差
        _, residual_norms = calculate_residuals(embeddings, reconstructed)
        
        print(f"      Rank {rank}: mean_residual={residual_norms.mean():.4f}, "
              f"std={residual_norms.std():.4f}, explained_var={explained_variance:.3f}")
        
        # 创建原始索引到残差的映射
        residual_dict = {indices[i]: residual_norms[i] for i in range(len(indices))}
        all_residuals[rank] = residual_dict
    
    # 释放 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()
    
    return all_residuals

# 处理多模型（不包含文件读写，不改变顺序）
def process_multi_model(rows, candidate_ranks):
    """
    处理多模型的残差计算，计算所有rank的结果，按原有顺序选择样本
    
    Returns:
    - all_residuals: 字典，key为rank，value为另一个字典（行索引->残差值）
    """
    # 定义每个模型要选择的样本数
    model_sample_counts = {
        'deepseek-vl2-tiny': 4,  # deepseek选4个
        'Qwen2-VL-7B-Instruct': 3,  # Qwen选3个
        'llava-1.5-7b-hf': 3  # llava选3个
    }
    
    # 收集各模型指定数量的数据及其原始索引（按原有顺序）
    selected_data = []
    for model_name, sample_count in model_sample_counts.items():
        model_data = [(i, row) for i, row in enumerate(rows) if row['Model'] == model_name][:sample_count]
        selected_data.extend(model_data)
        print(f"    Selected {len(model_data)} samples from {model_name} (by original order)")
    
    if not selected_data:
        print(f"  No data found for multi-agent processing")
        return {}
    
    indices, selected_rows = zip(*selected_data)
    selected_captions = [row['Caption'] for row in selected_rows]
    print(f"  Multi-agent processing with {len(selected_captions)} samples (4+3+3=10)")
    
    # 计算嵌入
    embeddings = compute_embeddings(selected_captions)
    
    # 对每个候选rank计算残差
    all_residuals = {}
    max_possible_rank = min(embeddings.shape[0] - 1, embeddings.shape[1])
    
    for rank in candidate_ranks:
        if rank > max_possible_rank:
            print(f"      Rank {rank} exceeds maximum possible rank {max_possible_rank}, skipping")
            continue
            
        # 低秩近似
        reconstructed, explained_variance = low_rank_approximation_with_rank(embeddings, rank)
        
        # 计算残差
        _, residual_norms = calculate_residuals(embeddings, reconstructed)
        
        print(f"      Rank {rank}: mean_residual={residual_norms.mean():.4f}, "
              f"std={residual_norms.std():.4f}, explained_var={explained_variance:.3f}")
        
        # 创建原始索引到残差的映射
        residual_dict = {indices[i]: residual_norms[i] for i in range(len(indices))}
        all_residuals[rank] = residual_dict
    
    # 释放 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()
    
    return all_residuals

# 主处理函数：处理整个CSV文件（完全不改变顺序）
def process_entire_csv(csv_file, input_folder, candidate_ranks):
    """
    处理整个CSV文件：计算所有模型所有rank的残差，完全不改变原有顺序
    """
    csv_path = os.path.join(input_folder, csv_file)
    
    # 读取 CSV 文件
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    
    # 如果需要清空历史列，取消下面这行的注释
    # fieldnames = clean_residual_columns(rows, fieldnames)
    
    # 处理各个单独的模型
    # for model_name in ['deepseek-vl2-tiny', 'Qwen2-VL-7B-Instruct', 'llava-1.5-7b-hf']:
    #     try:
    #         print(f"  Processing model: {model_name}")
    #         all_residuals = process_single_model(rows, model_name, candidate_ranks)
            
    #         if all_residuals:
    #             # 对每个rank创建对应的列并填充数据
    #             for rank, residual_dict in all_residuals.items():
    #                 col_name = f'ResidualError_r={rank}'
    #                 if col_name not in fieldnames:
    #                     fieldnames.append(col_name)
                    
    #                 # 更新rows中的ResidualError值（不改变行顺序）
    #                 for idx, residual_value in residual_dict.items():
    #                     rows[idx][col_name] = f"{residual_value:.4f}"
                
    #             print(f"    ✅ Processed {model_name} with ranks {sorted(all_residuals.keys())} (order preserved)")
                
    #     except Exception as e:
    #         print(f"    Error processing {model_name}: {e}")
    
    # 处理多模型
    try:
        print(f"  Processing multi-agent")
        all_residuals = process_multi_model(rows, candidate_ranks)
        
        if all_residuals:
            # 对每个rank创建对应的列并填充数据
            for rank, residual_dict in all_residuals.items():
                col_name = f'ResidualError_multi_agent_r={rank}'
                if col_name not in fieldnames:
                    fieldnames.append(col_name)
                
                # 更新rows中的ResidualError_multi_agent值（不改变行顺序）
                for idx, residual_value in residual_dict.items():
                    rows[idx][col_name] = f"{residual_value:.4f}"
            
            print(f"    ✅ Processed multi-agent with ranks {sorted(all_residuals.keys())} (order preserved)")
            
    except Exception as e:
        print(f"    Error in multi-agent processing: {e}")
    
    # 写回 CSV 文件
    with open(csv_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Completed processing {csv_file}")

if __name__ == "__main__":
    # 参数设置
    CANDIDATE_RANKS = [1, 2, 3]  # 候选rank列表
    
    print(f"SVD residual calculation with multiple ranks (Order Completely Preserved)")
    print(f"Candidate ranks: {CANDIDATE_RANKS}")
    print(f"Multi-agent sampling: deepseek-vl2-tiny(4) + Qwen2-VL-7B-Instruct(3) + llava-1.5-7b-hf(3) = 10 samples")
    print("Important: All rows maintain their exact original order throughout processing")
    print("="*80)
    
    input_folder = "./Nuscenes/results_caption/"
    csv_files = sorted(f for f in os.listdir(input_folder) if f.endswith('.csv'))
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}")
        process_entire_csv(csv_file, input_folder, CANDIDATE_RANKS)
    
    print(f"\n{'='*80}")
    print("All files processed successfully!")
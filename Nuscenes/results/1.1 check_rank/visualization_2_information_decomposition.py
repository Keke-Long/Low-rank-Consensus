import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import os
import csv

def compute_embeddings(answers):
    model = SentenceTransformer('intfloat/e5-large-v2')  
    embeddings = model.encode(answers)
    return embeddings

def analyze_information_decomposition(input_folder):
    """
    可视化低秩部分vs残差部分的信息量分布
    """
    # 数据读取部分 (与之前相同)
    csv_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.csv')])
    first_file = csv_files[0]
    csv_path = os.path.join(input_folder, first_file)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    models = ['deepseek-vl2-tiny', 'Qwen2-VL-7B-Instruct', 'llava-1.5-7b-hf']
    all_captions = []
    caption_labels = []
    
    for model in models:
        model_rows = [row for row in rows if row['Model'] == model]
        if model_rows:
            sample_size = 4 if model == 'deepseek-vl2-tiny' else 3
            # sample_size = 10
            captions = [row['Caption'] for row in model_rows[:sample_size]]
            all_captions.extend(captions)
            caption_labels.extend([f"{model.split('-')[0]}_{i+1}" for i in range(len(captions))])
    
    # 计算embedding和SVD
    embeddings = compute_embeddings(all_captions)
    X = embeddings.astype(np.float64)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # 计算不同rank下的信息量分布
    max_rank = min(X.shape)
    ranks = range(1, max_rank + 1)
    
    # 存储分析结果
    low_rank_info = []
    residual_info = []
    cumulative_variance = []
    frobenius_norms_lowrank = []
    frobenius_norms_residual = []
    
    for rank in ranks:
        # 截断SVD
        U_r = U[:, :rank]
        s_r = s[:rank]
        Vt_r = Vt[:rank, :]
        
        # 重构和残差
        X_lowrank = U_r @ np.diag(s_r) @ Vt_r
        X_residual = X - X_lowrank
        
        # 计算信息量 (使用Frobenius范数作为信息量度量)
        lowrank_norm = np.linalg.norm(X_lowrank, 'fro')
        residual_norm = np.linalg.norm(X_residual, 'fro')
        
        frobenius_norms_lowrank.append(lowrank_norm)
        frobenius_norms_residual.append(residual_norm)
        
        # 计算解释方差比例
        explained_var = np.sum(s_r**2) / np.sum(s**2)
        cumulative_variance.append(explained_var)
        
        # 计算每个caption在低秩和残差中的贡献
        lowrank_caption_norms = np.linalg.norm(X_lowrank, axis=1)
        residual_caption_norms = np.linalg.norm(X_residual, axis=1)
        
        low_rank_info.append(lowrank_caption_norms)
        residual_info.append(residual_caption_norms)
    
    # ==================== 创建可视化 ====================
    fig = plt.figure(figsize=(8, 4))
    
    # 1. 主图：低秩vs残差信息量随主成分数量变化
    plt.subplot(1, 2, 1)
    plt.plot(ranks, frobenius_norms_lowrank, 'b-o', linewidth=1, markersize=5, label='Low-rank Part')
    plt.plot(ranks, frobenius_norms_residual, 'r-s', linewidth=1, markersize=5, label='Residual Part')
    plt.axhline(y=np.linalg.norm(X, 'fro'), color='black', linestyle='--', alpha=0.7, label='Original Matrix')
    
    plt.xlabel('Number of Components (Rank)', fontsize=11)
    plt.ylabel('Information Content (Frobenius Norm)', fontsize=11)
    plt.title('Information Distribution', fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 2. 解释方差比例曲线
    plt.subplot(1, 2, 2)
    plt.plot(ranks, cumulative_variance, 'g-o', linewidth=1, markersize=5)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% Threshold')
    
    plt.xlabel('Number of Components (Rank)', fontsize=11)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=11)
    plt.title('Cumulative Information Capture', fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    
    plt.tight_layout(pad=2)
    plt.savefig('./Nuscenes/results/1.1 check_rank/information_decomposition.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ==================== 数值分析输出 ====================
    print("="*60)
    print("信息量分解分析结果:")
    print("="*60)
    
    original_norm = np.linalg.norm(X, 'fro')
    print(f"原始矩阵总信息量 (Frobenius范数): {original_norm:.4f}")
    
    print(f"\n不同rank下的信息量分布:")
    print("-" * 50)
    print("Rank | Low-rank | Residual | Explained Var%")
    print("-" * 50)
    
    for i, rank in enumerate([1, 3, 5, 7, 10]):
        if rank <= len(frobenius_norms_lowrank):
            lowrank = frobenius_norms_lowrank[rank-1]
            residual = frobenius_norms_residual[rank-1]
            var_explained = cumulative_variance[rank-1] * 100
            print(f"{rank:4d} | {lowrank:8.3f} | {residual:8.3f} | {var_explained:9.1f}%")
    
    # 找到"最优"分割点 (80%信息量阈值)
    threshold_80 = next((i for i, var in enumerate(cumulative_variance) if var >= 0.8), len(cumulative_variance)-1)
    print(f"\n推荐rank (80%信息量阈值): {threshold_80 + 1}")
    
    
    return X, frobenius_norms_lowrank, frobenius_norms_residual, cumulative_variance

if __name__ == "__main__":
    input_folder = './Nuscenes/results_caption/'
    
    # 确保输出目录存在
    os.makedirs('./Nuscenes/results/1.1 check_rank/', exist_ok=True)
    
    # 执行信息量分解分析
    results = analyze_information_decomposition(input_folder)
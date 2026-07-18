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

def plot_matrix_comparison(input_folder):
    """
    第一张图：原始矩阵、重构矩阵、残差矩阵对比（统一color bar）
    """
    # 读取数据
    csv_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.csv')])
    first_file = csv_files[1]
    csv_path = os.path.join(input_folder, first_file)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 按模型分组获取captions
    models = ['deepseek-vl2-tiny', 'Qwen2-VL-7B-Instruct', 'llava-1.5-7b-hf']
    all_captions = []
    caption_labels = []
    
    for model in models:
        model_rows = [row for row in rows if row['Model'] == model]
        if model_rows:
            sample_size = 4 if model == 'deepseek-vl2-tiny' else 3
            captions = [row['Caption'] for row in model_rows[:sample_size]]
            all_captions.extend(captions)
            caption_labels.extend([f"{model.split('-')[0]}_{i+1}" for i in range(len(captions))])
    
    # 计算embedding
    embeddings = compute_embeddings(all_captions)
    X = embeddings.astype(np.float64)
    
    # SVD分解
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = 5  # 选择rank=5进行演示
    
    # 截断到rank
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vt_r = Vt[:rank, :]
    
    # 重构矩阵
    X_reconstructed = U_r @ np.diag(s_r) @ Vt_r
    
    # 计算残差
    Residual = X - X_reconstructed
    
    # 只显示前40维以便可视化
    X_display = X[:, :40]
    X_recon_display = X_reconstructed[:, :40]
    Residual_display = Residual[:, :40]
    
    # 计算统一的colorbar范围
    all_data = np.concatenate([X_display.flatten(), 
                              X_recon_display.flatten(), 
                              Residual_display.flatten()])
    vmin, vmax = np.percentile(all_data, [2, 98])  # 使用2%和98%分位数避免极值影响
    
    # ==================== 创建第一张图 ====================
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    
    # 1. 原始矩阵
    im1 = axes[0].imshow(X_display, cmap='RdBu_r', aspect='auto', 
                         vmin=vmin, vmax=vmax)
    axes[0].set_title('Original Matrix (first 40 dims)', fontsize=11)
    axes[0].set_ylabel('Captions', fontsize=11)
    axes[0].set_xlabel('Feature Dimensions', fontsize=11)
    axes[0].set_yticks(range(len(caption_labels)))
    axes[0].set_yticklabels(caption_labels, fontsize=9)
    axes[0].set_xticks(range(0, 40, 10))
    
    # 2. 重构矩阵
    im2 = axes[1].imshow(X_recon_display, cmap='RdBu_r', aspect='auto',
                         vmin=vmin, vmax=vmax)
    axes[1].set_title('Low-rank Matrix', fontsize=11)
    axes[1].set_ylabel('Captions', fontsize=11)
    axes[1].set_xlabel('Feature Dimensions', fontsize=11)
    axes[1].set_yticks(range(len(caption_labels)))
    axes[1].set_yticklabels(caption_labels, fontsize=9)
    axes[1].set_xticks(range(0, 40, 10))
    
    # 3. 残差矩阵
    im3 = axes[2].imshow(Residual_display, cmap='RdBu_r', aspect='auto',
                         vmin=vmin, vmax=vmax)
    axes[2].set_title('Residual Matrix', fontsize=11)
    axes[2].set_ylabel('Captions', fontsize=11)
    axes[2].set_xlabel('Feature Dimensions', fontsize=11)
    axes[2].set_yticks(range(len(caption_labels)))
    axes[2].set_yticklabels(caption_labels, fontsize=9)
    axes[2].set_xticks(range(0, 40, 10))
    
    # 添加统一的colorbar
    fig.subplots_adjust(wspace=0.7, right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Feature Value', rotation=270, labelpad=20, fontsize=11)
       
    plt.savefig('./Nuscenes/results/1.1 check_rank/matrix_comparison multi.jpg', dpi=300, bbox_inches='tight')
    
    return X, X_reconstructed, Residual, caption_labels

if __name__ == "__main__":
    input_folder = './Nuscenes/results_caption/'
    
    # 确保输出目录存在
    os.makedirs('./Nuscenes/results/1.1 check_rank/', exist_ok=True)
    
    # 生成矩阵对比图
    print("正在生成矩阵对比图...")
    X, X_reconstructed, Residual, caption_labels = plot_matrix_comparison(input_folder)
    
    # 输出数值信息
    print("="*60)
    print("矩阵分解详细信息:")
    print("="*60)
    print(f"原始矩阵维度: {X.shape}")
    print(f"重构矩阵维度: {X_reconstructed.shape}")
    print(f"残差矩阵维度: {Residual.shape}")
    
    print(f"\n重构质量分析:")
    reconstruction_error = np.linalg.norm(Residual, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(X, 'fro')
    print(f"  重构误差(Frobenius范数): {reconstruction_error:.4f}")
    print(f"  相对误差: {relative_error:.1%}")
    
    print(f"\n各caption的重构质量:")
    residual_norms = np.linalg.norm(Residual, axis=1)
    for label, norm in zip(caption_labels, residual_norms):
        print(f"  {label}: 残差范数 = {norm:.4f}")
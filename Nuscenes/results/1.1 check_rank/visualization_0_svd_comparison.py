import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
import csv
import os

# 计算句子嵌入
def compute_embeddings(answers):
    model = SentenceTransformer('intfloat/e5-large-v2')
    embeddings = model.encode(answers, convert_to_numpy=True)
    return embeddings

# 使用 SVD 进行低秩近似
def low_rank_approximation(embeddings, rank):
    max_rank = min(embeddings.shape[0] - 1, embeddings.shape[1], rank)
    svd = TruncatedSVD(n_components=max_rank)
    low_rank_matrix = svd.fit_transform(embeddings)
    reconstructed = svd.inverse_transform(low_rank_matrix)
    return reconstructed

# 绘制奇异值对比图
def plot_svd_comparison(embeddings, rank, output_path):
    # 原始矩阵 SVD
    U, s_orig, Vt = np.linalg.svd(embeddings, full_matrices=False)

    # 低秩近似 SVD
    M_r = low_rank_approximation(embeddings, rank)
    U_r, s_r, Vt_r = np.linalg.svd(M_r, full_matrices=False)

    # 横坐标
    x = np.arange(len(s_orig))
    plt.figure(figsize=(3.5, 3))

    # 区域填充
    consensus_patch = plt.fill_between(
        x, 0, s_r, color='lightblue', alpha=0.5, label='Consensus Components'
    )
    halluc_patch = plt.fill_between(
        x, s_r, s_orig, color='orange', alpha=0.4, label='Residual Components'
    )

    # 原始矩阵曲线（黑线 + 空心圆点）
    orig_line, = plt.plot(
        x, s_orig, color='black', linestyle='-', linewidth=1,
        marker='o', markersize=5, markerfacecolor='none', markeredgecolor='black',
        label=r"Original Matrix $M$" 
    )

    # 低秩近似曲线（绿色虚线 + 空心方块）
    lowrank_line, = plt.plot(
        x, s_r, color='lightseagreen', linestyle='--', linewidth=1,
        marker='s', markersize=3, markerfacecolor='lightseagreen', markeredgecolor='lightseagreen',
        label=r"Low-rank Matrix $\hat{M}$"     
    )

    # 坐标轴
    plt.xlabel("Singular Value Index")
    plt.ylabel("Amplitude")
    # plt.ylim(bottom=0)

    # 图例：包含线条 + 区域
    plt.legend(handles=[orig_line, lowrank_line, consensus_patch, halluc_patch], loc='upper right', frameon=False)

    # 保存图
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



# 示例：处理一个CSV文件并画图
if __name__ == "__main__":
    input_folder = "./Nuscenes/results_caption/"
    output_folder = "./Nuscenes/results/1.1 check_rank/"
    # These four figures were originally generated from this scene CSV.
    csv_path = os.path.join(input_folder, "0001_e93e98b63d3b40209056d129dc53ceee.csv")
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    models = [
        'deepseek-vl2-tiny',
        'llava-1.5-7b-hf',
        'Qwen2-VL-7B-Instruct',
    ]
    for model in models:
        captions = [row['Caption'] for row in rows if row['Model'] == model]
        embeddings = compute_embeddings(captions)
        plot_svd_comparison(
            embeddings,
            rank=5,
            output_path=os.path.join(output_folder, f"svd_comparison_rank=5 {model}.png"),
        )

    # The original multi-model figure uses these zero-based CSV row indices.
    selected_indices = [1, 2, 3, 11, 12, 13, 21, 22, 23, 24]  # 注意：这里是以0为起点的行号
    captions = [row['Caption'] for i, row in enumerate(rows) if i in selected_indices]
    embeddings = compute_embeddings(captions)
    plot_svd_comparison(
        embeddings,
        rank=5,
        output_path=os.path.join(output_folder, "svd_comparison_rank=5 multi.png"),
    )

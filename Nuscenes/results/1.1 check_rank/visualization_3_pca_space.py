import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os
import csv
import re

def compute_embeddings(answers):
    model = SentenceTransformer('intfloat/e5-large-v2')  #'intfloat/e5-large-v2''all-MiniLM-L12-v2' 
    embeddings = model.encode(answers)
    return embeddings

def extract_keywords(text):
    """提取关键词"""
    # 简单的关键词提取
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # 过滤常见停用词
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                  'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords

def analyze_pca_space(input_folder):
    """只分析主成分空间和语义解释"""
    
    # 数据读取
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
    model_info = []
    
    for model in models:
        model_rows = [row for row in rows if row['Model'] == model]
        if model_rows:
            sample_size = 4 if model == 'deepseek-vl2-tiny' else 3
            # sample_size = 10
            captions = [row['Caption'] for row in model_rows[:sample_size]]
            all_captions.extend(captions)
            caption_labels.extend([f"{model.split('-')[0]}_{i+1}" for i in range(len(captions))])
            model_info.extend([model.split('-')[0]] * len(captions))
    
    # SVD分解
    embeddings = compute_embeddings(all_captions)
    X = embeddings.astype(np.float64)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # 选择rank=5进行分析
    rank = 2
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vt_r = Vt[:rank, :]
    
    X_lowrank = U_r @ np.diag(s_r) @ Vt_r
    X_residual = X - X_lowrank
    residual_norms = np.linalg.norm(X_residual, axis=1)
    
    # 计算残差排序
    sorted_indices = np.argsort(residual_norms)[::-1]
    
    # 提取每个caption的关键词
    all_keywords = []
    caption_keywords = []
    for caption in all_captions:
        keywords = extract_keywords(caption)
        caption_keywords.append(keywords)
        all_keywords.extend(keywords)
    
    # ==================== 单独绘制主成分空间图 ====================
    plt.figure(figsize=(5, 3.5))
   
    projections = U_r * s_r  # 每个caption在主成分上的投影
  
    # 选择前两个主成分进行2D投影可视化
    scatter = plt.scatter(projections[:, 0], projections[:, 1], 
                         c=residual_norms, s=50, cmap='RdYlBu_r', alpha=0.8, edgecolors='black', linewidths=0.5)
    
    # 标注每个点
    for i, label in enumerate(caption_labels):
        plt.annotate(label, (projections[i, 0], projections[i, 1]), 
                    xytext=(2, 2), textcoords='offset points', fontsize=3, 
                    bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.2))
    
    # 美化颜色条
    cbar = plt.colorbar(scatter, label='Residual Norm (Hallucination Score)', pad=0.1)
    cbar.ax.tick_params(labelsize=9)
    
    plt.xlabel(f'1st Principal Component (σ1={s_r[0]:.2f})', fontsize=11)
    plt.ylabel(f'2nd Principal Component (σ2={s_r[1]:.2f})', fontsize=11)
    # plt.title('Captions in Principal Component Space', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./Nuscenes/results/1.1 check_rank/pca_space.jpg', dpi=300, bbox_inches='tight')

    
    # ==================== 打印语义解释 ====================
    print("=" * 80)
    print("SEMANTIC INTERPRETATION OF COMPONENTS")
    print("=" * 80)
    
    print(f"Analysis based on SVD with rank={rank}")
    print(f"Total explained variance: {np.sum(s_r**2)/np.sum(s**2)*100:.1f}%")
    print()
    
    for comp in range(min(5, rank)):  # 分析前5个成分
        # 找到在该成分上投影最大的caption
        comp_projections = np.abs(U_r[:, comp] * s_r[comp])
        top_caption_idx = np.argmax(comp_projections)
        
        print(f"🔍 COMPONENT {comp+1} ANALYSIS:")
        print(f"   Singular Value: σ{comp+1} = {s_r[comp]:.4f}")
        print(f"   Variance Explained: {s_r[comp]**2/np.sum(s**2)*100:.2f}%")
        print(f"   Main Contributor: {caption_labels[top_caption_idx]}")
        print(f"   Caption Content: '{all_captions[top_caption_idx]}'")
        
        # 找出该caption的关键词
        keywords = caption_keywords[top_caption_idx][:5]
        if keywords:
            print(f"   Key Words: {', '.join(keywords)}")
        
        # 分析该成分上的投影分布
        positive_projections = U_r[:, comp] > 0
        negative_projections = U_r[:, comp] < 0
        
        if np.any(positive_projections) and np.any(negative_projections):
            pos_captions = [caption_labels[i] for i in range(len(caption_labels)) if positive_projections[i]]
            neg_captions = [caption_labels[i] for i in range(len(caption_labels)) if negative_projections[i]]
            
            print(f"   Positive Direction: {', '.join(pos_captions[:3])}")
            print(f"   Negative Direction: {', '.join(neg_captions[:3])}")
        
        print()
    
    # ==================== 额外的分析信息 ====================
    print("📈 COMPONENT IMPORTANCE:")
    component_importance = s_r**2 / np.sum(s_r**2)
    for i, importance in enumerate(component_importance):
        print(f"   Component {i+1}: {importance*100:.1f}% of retained variance")
    
    print(f"\n🎯 ANALYSIS SUMMARY:")
    print(f"   • File analyzed: {first_file}")
    print(f"   • SVD rank used: {rank}")
    print(f"   • Dimensions: {X.shape[1]} → {rank}")
    
    return X, U_r, s_r, Vt_r, X_lowrank, X_residual, residual_norms, all_captions, caption_labels

if __name__ == "__main__":
    input_folder = './Nuscenes/results_caption/'
    
    # 确保输出目录存在
    os.makedirs('./Nuscenes/results/1.1 check_rank/', exist_ok=True)
    
    # 执行分析
    results = analyze_pca_space(input_folder)
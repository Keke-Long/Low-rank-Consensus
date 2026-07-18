import os
import pandas as pd
from collections import defaultdict

ROUND2_RESULTS_DIR = "./Nuscenes/baselines/debate_baseline/results_round2/"

def parse_hallucination_check(hallucination_check_str):
    """
    解析hallucination_check字符串，计算幻觉占比
    格式: "1:Yes, 2:No, 3:Yes" 表示第1、3句有幻觉，第2句没有
    返回幻觉句子的比例
    """
    if not hallucination_check_str or hallucination_check_str.strip() == '':
        return None
    
    try:
        # 按逗号分割
        parts = [p.strip() for p in hallucination_check_str.split(',')]
        
        total_sentences = len(parts)
        hallucinated_sentences = 0
        
        for part in parts:
            # 格式: "1:Yes" 或 "2:No"
            if ':' in part:
                _, label = part.split(':', 1)
                if label.strip().lower() == 'yes':
                    hallucinated_sentences += 1
        
        if total_sentences == 0:
            return None
            
        return hallucinated_sentences / total_sentences
        
    except Exception as e:
        print(f"Error parsing: {hallucination_check_str}, error: {e}")
        return None

def calculate_debate_accuracy():
    """计算debate baseline的accuracy和处理时间"""
    
    # 统计数据结构
    model_stats = defaultdict(lambda: {
        'total_captions': 0,
        'total_sentences': 0,
        'hallucinated_sentences': 0,
        'total_time': 0.0,
        'caption_count': 0
    })
    
    # 遍历所有round2结果文件
    csv_files = sorted([f for f in os.listdir(ROUND2_RESULTS_DIR) if f.endswith('_final.csv')])
    
    total_files = 0
    processed_files = 0
    
    for filename in csv_files:
        csv_path = os.path.join(ROUND2_RESULTS_DIR, filename)
        total_files += 1
        
        try:
            df = pd.read_csv(csv_path)
            
            # 检查必需的列
            if 'Model' not in df.columns or 'hallucination_check' not in df.columns or 'Time' not in df.columns:
                print(f"Warning: Missing required columns in {filename}, skipping...")
                continue
            
            processed_files += 1
            
            for _, row in df.iterrows():
                model_name = row['Model']
                hallucination_check = row.get('hallucination_check', '')
                time_str = row.get('Time', '')
                
                # 跳过没有幻觉检测结果的行
                if pd.isna(hallucination_check) or hallucination_check == '':
                    continue
                
                # 解析幻觉检测结果
                parts = [p.strip() for p in str(hallucination_check).split(',')]
                total_sentences = len(parts)
                hallucinated_count = 0
                
                for part in parts:
                    if ':' in part:
                        _, label = part.split(':', 1)
                        if label.strip().lower() == 'yes':
                            hallucinated_count += 1
                
                # 更新统计
                model_stats[model_name]['total_captions'] += 1
                model_stats[model_name]['total_sentences'] += total_sentences
                model_stats[model_name]['hallucinated_sentences'] += hallucinated_count
                
                # 处理时间 (round2时间 * 2)
                try:
                    round2_time = float(time_str)
                    total_time = round2_time * 2  # round1 + round2
                    model_stats[model_name]['total_time'] += total_time
                    model_stats[model_name]['caption_count'] += 1
                except:
                    pass
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return model_stats, total_files, processed_files

def main():
    print("="*80)
    print("Multi-Agent Debate Baseline - Performance Evaluation")
    print("="*80)
    
    model_stats, total_files, processed_files = calculate_debate_accuracy()
    
    print(f"\nDataset Statistics:")
    print(f"  Total files: {total_files}")
    print(f"  Successfully processed: {processed_files}")
    
    if not model_stats:
        print("\nNo data found! Please run debate_hallucination_check.py first.")
        return
    
    # 打印每个模型的结果
    print("\n" + "="*80)
    print("Per-Model Results")
    print("="*80)
    
    all_models_accuracy = []
    all_models_time = []
    
    for model_name in ['deepseek-vl2-tiny', 'Qwen2-VL-7B-Instruct', 'llava-1.5-7b-hf']:
        if model_name in model_stats:
            stats = model_stats[model_name]
            
            # 计算accuracy (无幻觉句子的比例)
            if stats['total_sentences'] > 0:
                hallucination_rate = stats['hallucinated_sentences'] / stats['total_sentences']
                accuracy = 1 - hallucination_rate
            else:
                accuracy = 0.0
            
            # 计算平均处理时间
            if stats['caption_count'] > 0:
                avg_time = stats['total_time'] / stats['caption_count']
            else:
                avg_time = 0.0
            
            all_models_accuracy.append(accuracy)
            all_models_time.append(avg_time)
            
            print(f"\n{model_name}:")
            print(f"  Total captions: {stats['total_captions']}")
            print(f"  Total sentences: {stats['total_sentences']}")
            print(f"  Hallucinated sentences: {stats['hallucinated_sentences']}")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Average processing time: {avg_time:.2f}s (Round1 + Round2)")
            print(f"    └─ Estimated Round1: {avg_time/2:.2f}s")
            print(f"    └─ Round2: {avg_time/2:.2f}s")
    
    # 打印总体统计
    print("\n" + "="*80)
    print("Overall Statistics")
    print("="*80)
    
    if all_models_accuracy:
        overall_accuracy = sum(all_models_accuracy) / len(all_models_accuracy)
        overall_avg_time = sum(all_models_time) / len(all_models_time)
        
        print(f"\nAverage Accuracy (across all models): {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"Average Processing Time (across all models): {overall_avg_time:.2f}s")
        print(f"  └─ Per round: ~{overall_avg_time/2:.2f}s")
    
    # 打印Top-1选择策略建议
    print("\n" + "="*80)
    print("Top-1 Selection Strategy")
    print("="*80)
    
    print("\nFor optimal selection accuracy:")
    if all_models_accuracy:
        best_model_idx = all_models_accuracy.index(max(all_models_accuracy))
        best_model = ['deepseek-vl2-tiny', 'Qwen2-VL-7B-Instruct', 'llava-1.5-7b-hf'][best_model_idx]
        print(f"  Best performing model: {best_model}")
        print(f"  Accuracy: {max(all_models_accuracy):.4f} ({max(all_models_accuracy)*100:.2f}%)")
        print(f"\nNote: To calculate Top-1 selection accuracy, you would need to:")
        print(f"  1. Select the caption with lowest hallucination score from each scene")
        print(f"  2. Check if it's actually the best caption")
        print(f"  3. Calculate: (correct selections) / (total scenes)")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
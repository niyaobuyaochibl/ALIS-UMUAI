"""
创建Amazon Books数据子集
保留20%的数据，确保每个用户至少有5次交互
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def create_subset(input_dir, output_dir, sample_ratio=0.2, min_interactions=5):
    """
    创建数据子集
    
    Args:
        input_dir: 完整数据目录
        output_dir: 输出目录
        sample_ratio: 采样比例
        min_interactions: 最小交互次数
    """
    print(f"从 {input_dir} 创建 {sample_ratio*100}% 数据子集...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载完整数据
    print("加载数据...")
    train = pickle.load(open(f'{input_dir}/train.pkl', 'rb'))
    val = pickle.load(open(f'{input_dir}/val.pkl', 'rb'))
    test = pickle.load(open(f'{input_dir}/test.pkl', 'rb'))
    
    print(f"原始数据: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    # 合并所有数据进行采样
    all_data = pd.concat([train, val, test], ignore_index=True)
    print(f"总交互数: {len(all_data)}")
    
    # 随机采样用户
    all_users = all_data['user_id'].unique()
    print(f"总用户数: {len(all_users)}")
    
    # 采样用户
    n_sample_users = int(len(all_users) * sample_ratio)
    sampled_users = np.random.choice(all_users, n_sample_users, replace=False)
    print(f"采样用户数: {len(sampled_users)}")
    
    # 筛选这些用户的数据
    subset_data = all_data[all_data['user_id'].isin(sampled_users)].copy()
    print(f"采样后交互数: {len(subset_data)}")
    
    # 过滤低频用户和物品
    print(f"\n过滤交互次数 < {min_interactions} 的用户和物品...")
    prev_size = len(subset_data)
    iteration = 0
    
    while True:
        iteration += 1
        user_counts = subset_data['user_id'].value_counts()
        item_counts = subset_data['item_id'].value_counts()
        
        # 过滤
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        subset_data = subset_data[
            subset_data['user_id'].isin(valid_users) & 
            subset_data['item_id'].isin(valid_items)
        ]
        
        print(f"迭代 {iteration}: {len(subset_data)} 交互")
        
        if len(subset_data) == prev_size:
            break
        prev_size = len(subset_data)
    
    # 重新映射ID
    print("\n重新映射用户和物品ID...")
    unique_users = sorted(subset_data['user_id'].unique())
    unique_items = sorted(subset_data['item_id'].unique())
    
    user2id = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item2id = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    id2user = {v: k for k, v in user2id.items()}
    id2item = {v: k for k, v in item2id.items()}
    
    subset_data['user_idx'] = subset_data['user_id'].map(user2id)
    subset_data['item_idx'] = subset_data['item_id'].map(item2id)
    
    # 按时间排序
    subset_data = subset_data.sort_values('timestamp')
    
    # 分割训练/验证/测试集 (80/10/10)
    n = len(subset_data)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_subset = subset_data.iloc[:train_end]
    val_subset = subset_data.iloc[train_end:val_end]
    test_subset = subset_data.iloc[val_end:]
    
    print(f"\n子集统计:")
    print(f"  用户数: {len(unique_users)}")
    print(f"  物品数: {len(unique_items)}")
    print(f"  Train: {len(train_subset)}")
    print(f"  Val: {len(val_subset)}")
    print(f"  Test: {len(test_subset)}")
    print(f"  密度: {len(subset_data) / (len(unique_users) * len(unique_items)) * 100:.4f}%")
    
    # 保存子集数据
    print("\n保存子集数据...")
    with open(output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train_subset, f)
    with open(output_dir / 'val.pkl', 'wb') as f:
        pickle.dump(val_subset, f)
    with open(output_dir / 'test.pkl', 'wb') as f:
        pickle.dump(test_subset, f)
    
    # 保存映射
    mappings = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    with open(output_dir / 'mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    
    # 保存统计信息
    stats = {
        'num_users': len(unique_users),
        'num_items': len(unique_items),
        'num_train': len(train_subset),
        'num_val': len(val_subset),
        'num_test': len(test_subset),
        'density': len(subset_data) / (len(unique_users) * len(unique_items))
    }
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 复制item_texts和item_categories（只保留有效物品）
    try:
        item_texts = pickle.load(open(f'{input_dir}/item_texts.pkl', 'rb'))
        item_categories = pickle.load(open(f'{input_dir}/item_categories.pkl', 'rb'))
        
        # 筛选子集中的物品
        subset_item_texts = {item2id[old_id]: text 
                            for old_id, text in item_texts.items() 
                            if old_id in item2id}
        subset_item_categories = {item2id[old_id]: cat 
                                 for old_id, cat in item_categories.items() 
                                 if old_id in item2id}
        
        with open(output_dir / 'item_texts.pkl', 'wb') as f:
            pickle.dump(subset_item_texts, f)
        with open(output_dir / 'item_categories.pkl', 'wb') as f:
            pickle.dump(subset_item_categories, f)
        
        print(f"保存了 {len(subset_item_texts)} 个物品的文本描述")
    except Exception as e:
        print(f"警告: 无法复制item_texts: {e}")
    
    print(f"\n✅ 子集创建完成！保存到: {output_dir}")
    print(f"\n数据规模对比:")
    print(f"  原始: 603,668 用户, 367,982 物品, 6.9M 交互")
    print(f"  子集: {len(unique_users)} 用户, {len(unique_items)} 物品, {len(subset_data)} 交互")
    print(f"  缩减比例: {len(subset_data) / 6891623 * 100:.1f}%")
    
    return stats

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw/amazon-books-processed')
    parser.add_argument('--output', default='data/raw/amazon-books-subset')
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--min_interactions', type=int, default=5)
    args = parser.parse_args()
    
    np.random.seed(2024)
    create_subset(args.input, args.output, args.ratio, args.min_interactions)




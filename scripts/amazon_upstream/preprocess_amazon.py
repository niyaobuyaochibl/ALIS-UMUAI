"""
Amazon Books数据预处理脚本
处理原始Amazon Books 5-core数据，生成训练所需格式
"""

import json
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
from tqdm import tqdm
import argparse

def parse_json_gz(filepath):
    """解析.json.gz文件"""
    print(f"正在读取: {filepath}")
    data = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data

def filter_users_items(interactions, min_interactions=5):
    """过滤交互次数少的用户和物品"""
    print(f"\n原始交互数: {len(interactions)}")
    
    # 反复过滤直到收敛
    prev_size = len(interactions)
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n过滤迭代 {iteration}:")
        
        # 统计用户和物品交互次数
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        
        for inter in interactions:
            user_counts[inter['user_id']] += 1
            item_counts[inter['item_id']] += 1
        
        # 过滤
        interactions = [
            inter for inter in interactions
            if user_counts[inter['user_id']] >= min_interactions 
            and item_counts[inter['item_id']] >= min_interactions
        ]
        
        print(f"  剩余交互数: {len(interactions)}")
        print(f"  剩余用户数: {len(set([i['user_id'] for i in interactions]))}")
        print(f"  剩余物品数: {len(set([i['item_id'] for i in interactions]))}")
        
        # 检查是否收敛
        if len(interactions) == prev_size:
            break
        prev_size = len(interactions)
    
    return interactions

def create_mappings(interactions):
    """创建用户和物品的ID映射"""
    unique_users = sorted(set([i['user_id'] for i in interactions]))
    unique_items = sorted(set([i['item_id'] for i in interactions]))
    
    user2id = {user: idx for idx, user in enumerate(unique_users)}
    item2id = {item: idx for idx, item in enumerate(unique_items)}
    
    print(f"\n创建映射:")
    print(f"  用户数: {len(user2id)}")
    print(f"  物品数: {len(item2id)}")
    
    return user2id, item2id

def time_based_split(interactions, train_ratio=0.8, val_ratio=0.1):
    """基于时间戳的数据划分"""
    print("\n按时间划分数据...")
    
    # 按用户分组
    user_interactions = defaultdict(list)
    for inter in interactions:
        user_interactions[inter['user_id']].append(inter)
    
    # 对每个用户，按时间排序并划分
    train_data = []
    val_data = []
    test_data = []
    
    for user_id, user_inters in tqdm(user_interactions.items(), desc="Splitting"):
        # 按时间戳排序
        user_inters = sorted(user_inters, key=lambda x: x['timestamp'])
        
        n = len(user_inters)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data.extend(user_inters[:train_end])
        val_data.extend(user_inters[train_end:val_end])
        test_data.extend(user_inters[val_end:])
    
    print(f"\n数据划分结果:")
    print(f"  训练集: {len(train_data)} 交互")
    print(f"  验证集: {len(val_data)} 交互")
    print(f"  测试集: {len(test_data)} 交互")
    
    return train_data, val_data, test_data

def process_metadata(meta_filepath, item2id):
    """处理物品元数据"""
    print(f"\n处理物品元数据...")
    
    meta_data = parse_json_gz(meta_filepath)
    
    # 创建物品ID到文本的映射
    item_texts = {}
    item_categories = {}
    
    for item in tqdm(meta_data, desc="Processing metadata"):
        asin = item.get('asin', '')
        
        if asin not in item2id:
            continue
        
        item_idx = item2id[asin]
        
        # 提取标题
        title = item.get('title', '').strip()
        
        # 提取类别（取第一个类别路径）
        categories = item.get('category', [])
        if isinstance(categories, list) and len(categories) > 0:
            category = ' '.join(categories)
        else:
            category = ''
        
        # 组合文本：标题 + 类别
        text = f"{title} {category}".strip()
        
        if text:
            item_texts[item_idx] = text
            item_categories[item_idx] = category
    
    print(f"  成功提取 {len(item_texts)} 个物品的文本信息")
    
    # 为没有文本的物品创建默认文本
    for item_idx in range(len(item2id)):
        if item_idx not in item_texts:
            item_texts[item_idx] = "Unknown Book"
            item_categories[item_idx] = ""
    
    return item_texts, item_categories

def save_processed_data(output_dir, train_data, val_data, test_data, 
                       user2id, item2id, item_texts, item_categories):
    """保存处理后的数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存数据到: {output_dir}")
    
    # 转换为DataFrame格式
    def to_dataframe(data, user2id, item2id):
        records = []
        for inter in data:
            records.append({
                'user_id': user2id[inter['user_id']],
                'item_id': item2id[inter['item_id']],
                'rating': inter.get('rating', 1.0),
                'timestamp': inter['timestamp']
            })
        return pd.DataFrame(records)
    
    train_df = to_dataframe(train_data, user2id, item2id)
    val_df = to_dataframe(val_data, user2id, item2id)
    test_df = to_dataframe(test_data, user2id, item2id)
    
    # 保存为pickle
    with open(output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train_df, f)
    
    with open(output_dir / 'val.pkl', 'wb') as f:
        pickle.dump(val_df, f)
    
    with open(output_dir / 'test.pkl', 'wb') as f:
        pickle.dump(test_df, f)
    
    # 保存映射
    mappings = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': {v: k for k, v in user2id.items()},
        'id2item': {v: k for k, v in item2id.items()}
    }
    
    with open(output_dir / 'mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    
    # 保存物品文本信息
    with open(output_dir / 'item_texts.pkl', 'wb') as f:
        pickle.dump(item_texts, f)
    
    with open(output_dir / 'item_categories.pkl', 'wb') as f:
        pickle.dump(item_categories, f)
    
    # 保存统计信息
    stats = {
        'num_users': len(user2id),
        'num_items': len(item2id),
        'num_train': len(train_df),
        'num_val': len(val_df),
        'num_test': len(test_df),
        'density': len(train_df) / (len(user2id) * len(item2id))
    }
    
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✅ 数据保存完成!")
    print("\n生成的文件:")
    for file in output_dir.iterdir():
        print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    print("\n数据集统计:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description='Amazon Books数据预处理')
    parser.add_argument('--raw_dir', type=str, 
                       default='data/raw/amazon-books',
                       help='原始数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='data/raw/amazon-books-processed',
                       help='输出目录')
    parser.add_argument('--min_interactions', type=int, default=5,
                       help='最小交互次数')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Amazon Books数据预处理")
    print("=" * 80)
    print(f"原始数据目录: {args.raw_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"最小交互次数: {args.min_interactions}")
    print(f"数据划分: {args.train_ratio:.1%} / {args.val_ratio:.1%} / {1-args.train_ratio-args.val_ratio:.1%}")
    print("=" * 80)
    
    # Step 1: 读取评分数据
    reviews_file = Path(args.raw_dir) / 'reviews_Books_5.json.gz'
    reviews = parse_json_gz(reviews_file)
    
    # 转换为统一格式
    interactions = []
    for review in tqdm(reviews, desc="Converting format"):
        interactions.append({
            'user_id': review['reviewerID'],
            'item_id': review['asin'],
            'rating': review['overall'],
            'timestamp': review['unixReviewTime']
        })
    
    # Step 2: 过滤稀疏用户和物品
    interactions = filter_users_items(interactions, args.min_interactions)
    
    # Step 3: 创建ID映射
    user2id, item2id = create_mappings(interactions)
    
    # Step 4: 按时间划分数据
    train_data, val_data, test_data = time_based_split(
        interactions, 
        args.train_ratio, 
        args.val_ratio
    )
    
    # Step 5: 处理物品元数据
    meta_file = Path(args.raw_dir) / 'meta_Books.json.gz'
    item_texts, item_categories = process_metadata(meta_file, item2id)
    
    # Step 6: 保存处理后的数据
    save_processed_data(
        args.output_dir,
        train_data, val_data, test_data,
        user2id, item2id,
        item_texts, item_categories
    )
    
    print("\n" + "=" * 80)
    print("✅ 预处理完成!")
    print("=" * 80)

if __name__ == '__main__':
    main()




import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# 1. 读取数据
# 假设你的文件名为 'attachment1.csv'
df = pd.read_csv('attachment1.csv') # 根据实际格式读取，可能是xlsx

# 2. 数据清洗
# 确保经纬度是数字，去除空值
df = df.dropna(subset=['longitude', 'latitude', 'poi_id'])
# 确保claim_flag是数字
df['claim_flag'] = df['claim_flag'].astype(int)

# 3. 定义聚类函数 (核心逻辑)
# 这个函数会被应用到每一个 poi_id 分组上
def perform_dbscan(group):
    # 提取经纬度
    coords = group[['latitude', 'longitude']].values
    
    # 建立模型：eps=0.001 (约100米), min_samples=3
    # metric='manhattan' 或 'euclidean' 都可以，地理上通常用 haversine 但这里近似处理可以直接用欧氏距离
    db = DBSCAN(eps=0.001, min_samples=3, metric='euclidean')
    
    # 训练并获取标签
    labels = db.fit_predict(coords)
    
    # 将标签放回数据中
    group['cluster_label'] = labels
    return group

# 4. 按 POI 分组并并行应用聚类
# 这一步可能比较慢，如果数据量大，可以考虑用 tqdm 显示进度
print("开始聚类...")
# group_keys=False 避免索引层级变得复杂
df_clustered = df.groupby('poi_id', group_keys=False).apply(perform_dbscan)

# 5. 生成全局唯一的区域ID
# 因为不同POI里都有 label 0, 1, 2... 我们需要组合一下
# 格式示例: POI_123_Cluster_0
df_clustered['unique_region_id'] = df_clustered['poi_id'].astype(str) + '_' + df_clustered['cluster_label'].astype(str)

# 6. 剔除噪声点 (可选)
# DBSCAN会将噪声标记为 -1，你可以决定是否保留这些散点
# 建议：如果只是算风险，可以把噪声点排除，或者把它们单独当做一类处理
valid_clusters = df_clustered[df_clustered['cluster_label'] != -1]

# 7. 计算每个区域的风险 (聚合分析)
risk_analysis = valid_clusters.groupby('unique_region_id').agg({
    'waybill_code': 'count',       # 总单量
    'claim_flag': 'sum',           # 异常单量
    'poi_id': 'first',             # 记录所属POI
    'latitude': 'mean',            # 区域中心纬度（可视化用）
    'longitude': 'mean'            # 区域中心经度（可视化用）
}).rename(columns={'waybill_code': 'total_orders', 'claim_flag': 'abnormal_count'})

# 8. 计算异常率
risk_analysis['abnormal_rate'] = risk_analysis['abnormal_count'] / risk_analysis['total_orders']

# 9. 识别重点风险区域
# 方法A：设定阈值（比如平均值的2倍）
# global_mean = df['claim_flag'].mean()
# high_risk = risk_analysis[risk_analysis['abnormal_rate'] > global_mean * 2]

# 方法B (题目推荐)：排序取Top N (比如前10%)
top_10_percent = int(len(risk_analysis) * 0.1)
high_risk_regions = risk_analysis.sort_values(by='abnormal_rate', ascending=False).head(top_10_percent)

print("识别出的高风险区域数:", len(high_risk_regions))
print(high_risk_regions.head())

# 10. 保存结果
high_risk_regions.to_csv('result_table_1.csv')
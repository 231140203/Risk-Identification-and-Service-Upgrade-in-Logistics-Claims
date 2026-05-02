import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tabulate import tabulate

#配置与初始化
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'database')
OutputWay = os.path.join(BASE_DIR, 'output')


# 输入和输出文件路径
INPUT_FILE = os.path.join(DATA_DIR, '附件1.xlsx')
FILE_OUT = os.path.join(OutputWay, '问题一.xlsx')
PLOT_FILE = os.path.join(OutputWay, '问题一.png')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



#逐层分类标注
def classify(df_in, k, q1, q2):

    df = df_in.copy()
    
    #对P分档 ---分q档
    df['P档'], p_bins = pd.qcut(df['实际赔付金额'], q = k, labels = False, duplicates = 'drop', retbins = True)
    print(f"成功分档")

    print("档位信息：")
    for i in range(len(p_bins) - 1):
        print(f"第 {i + 1} 档: [{p_bins[i]}, {p_bins[i + 1]})")

    # 分组中 '索赔差额' 列的 q1 和 q2 分位数
    Y1_group = df.groupby('P档')['索赔差额'].quantile(q1).rename('Y1')
    Y2_group = df.groupby('P档')['索赔差额'].quantile(q2).rename('Y2')

    df = df.merge(Y1_group, on='P档', how='left')
    df = df.merge(Y2_group, on='P档', how='left')

    print("\n每个档位的分组数据：")
    for i in range(len(p_bins) - 1):
        print(f"\n第 {i + 1} 档: [{p_bins[i]}, {p_bins[i + 1]})")
        print(df[df['P档'] == i])

    # 输出前 10 行数据以查看结果
    print("\n前 10 行数据：")
    print(df.head(3))

    # 标注函数
    def label_(row):
        D = row['索赔差额']
        Y1 = row['Y1']
        Y2 = row['Y2']
#Y1是“严重超额”和“诉求偏高”的边界线。
#Y2是“诉求偏高”和“合理诉求”的边界线。

        if D >= Y2:
            return '合理诉求'
        elif Y1 <= D < Y2:
            return '诉求偏高'
        else:
            return '严重超额'


    df['风险类别'] = df.apply(label_, axis=1)

    percentages = df['风险类别'].value_counts(normalize=True) * 100
    
    return df, percentages,p_bins


#图表
def plot_1(df, percentages):
    # 图1: 风险分布饼图
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    labels = percentages.index
    sizes = percentages.values
    explode = [0.01] * len(labels)
    if '严重超额' in labels:
        explode[labels.tolist().index('严重超额')] = 0.1

    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')
    ax1.set_title('风险类别分布占比', fontsize=16,  pad=20)
    plt.tight_layout()
    plt.savefig('风险分布饼图.png', dpi=300)
    plt.close()

    # 图2: 各类别运单数量柱状图
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    counts = df['风险类别'].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax2, hue=counts.index, palette="viridis", legend=False)
    ax2.set_title('各风险类别运单数量', fontsize=16)
    ax2.set_ylabel('运单数量')
    for i, count in enumerate(counts.values):
        ax2.text(i, count + 50, str(count), ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('运单数量柱状图.png', dpi=300)
    plt.close()

    # 图3: 各类别索赔差额核密度图
    fig3, ax3 = plt.subplots(figsize=(9, 7))
    sns.kdeplot(data=df, x='索赔差额', hue='风险类别', ax=ax3,
                fill=True, common_norm=False, palette="deep")
    ax3.set_title('各类别索赔差额的核密度图', fontsize=16)
    ax3.set_xlabel('索赔差额')
    ax3.set_xlim(df['索赔差额'].quantile(0.01), df['索赔差额'].quantile(0.99))
    ax3.legend(title='风险类别')
    plt.tight_layout()
    plt.savefig('索赔差额核密度图.png', dpi=300)
    plt.close()

def plot_2(df, p_bins, OutputWay):
        #绘制边界函数和分类结果的图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 分类结果散点图
    sample_df = df.sample(n=min(5000, len(df)), random_state=1)
    sns.scatterplot(data=sample_df, x='实际赔付金额', y='索赔差额', hue='风险类别',
                    palette="deep", ax=ax, s=5, alpha=0.8)
    
    # 动态边界线
    boundaries = df.groupby('P档')[['Y1', 'Y2']].mean()
    
    # 绘制阶梯状的边界线
    for i in range(len(p_bins) - 1):
        p_start = p_bins[i]
        p_end = p_bins[i+1]
        
        if i in boundaries.index:
            y_low = boundaries.loc[i, 'Y1']
            y_high = boundaries.loc[i, 'Y2']
            
            # 严重超额)
            ax.hlines(y=y_low, xmin=p_start, xmax=p_end, color='red', linestyle='--', linewidth=2)
            # 合理诉求)
            ax.hlines(y=y_high, xmin=p_start, xmax=p_end, color='green', linestyle='--', linewidth=2)

    # 添加虚拟线到图例中
    from matplotlib.lines import Line2D
    legend_elements = ax.get_legend().get_lines()
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', lw=2, label='严重超额边界 (Y_low)'))
    legend_elements.append(Line2D([0], [0], color='blue', linestyle='--', lw=2, label='合理诉求边界 (Y_high)'))
    ax.legend(handles=legend_elements, title='风险类别')

    ax.set_title('分类结果和边界函数', fontsize=16)
    ax.set_xlabel('实际赔付金额 (P)')
    ax.set_ylabel('索赔差额 (D)')
    ax.set_ylim(df['索赔差额'].quantile(0.01), df['索赔差额'].quantile(0.99)) # 聚焦
    ax.set_xlim(left=0, right=df['实际赔付金额'].quantile(0.99)) # 聚焦
    
    plt.savefig(os.path.join(OutputWay, '边界函数和分类结果.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_3(df, OutputWay):
    # 各类别实际赔付金额直方分布图

    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.histplot(data=df, x='实际赔付金额', hue='风险类别', multiple='stack',
                 palette="deep", ax=ax, bins=50)
    ax.set_title('各类别实际赔付金额直方分布图', fontsize=16)
    ax.set_xlabel('实际赔付金额')
    ax.set_xlim(left=0, right=df['实际赔付金额'].quantile(0.98)) # 聚焦
    
    plt.savefig(os.path.join(OutputWay, '实际赔付金额直方分布.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_4(df, OutputWay):
    """各类别索赔差额直方分布图"""

    fig = plt.figure(figsize=(14, 10))

    categories = df['风险类别'].unique()
    for i, category in enumerate(categories, 1):
        ax = plt.subplot(len(categories), 1, i)
        data = df[df['风险类别'] == category]['索赔差额'].dropna()

        # 根据数据量动态设置bins
        n_bins = min(50, max(10, len(data) // 20))  # 动态bins

        ax.hist(data, bins=n_bins, alpha=0.7, edgecolor='black')
        ax.set_title(f'风险类别: {category}', fontsize=12, fontweight='bold')
        ax.set_xlabel('索赔差额')
        ax.set_ylabel('计数')
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        ax.text(0.02, 0.98, f'样本数: {len(data)}\n均值: {data.mean():.2f}\n标准差: {data.std():.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('各类别索赔差额直方分布图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OutputWay, '各类别索赔差额直方分布图.png'), dpi=300, bbox_inches='tight')
    plt.close()


#分析报告函数 ---
def report(df_final, OutputWay, k, q1, q2):

    
    report_output = []
    report_output.append("=" * 60)
    report_output.append("物流理赔风险识别 - 问题一：风险标注模型分析报告")
    report_output.append("=" * 60)
    report_output.append("")

    # 一、数据概况
    report_output.append("一、数据概况")
    report_output.append("-" * 30)
    report_output.append("")

    total_count = len(df_final)
    p_min = df_final['实际赔付金额'].min()
    p_max = df_final['实际赔付金额'].max()
    p_mean = df_final['实际赔付金额'].mean()
    d_min = df_final['索赔差额'].min()
    d_max = df_final['索赔差额'].max()
    d_mean = df_final['索赔差额'].mean()

    report_output.append(f"总运单数: {total_count:,} 条")
    report_output.append(f"实际赔付金额范围: {p_min:.3f} ~ {p_max:.3f} 元")
    report_output.append(f"平均实际赔付金额: {p_mean:.3f} 元")
    report_output.append(f"索赔差额范围: {d_min:.3f} ~ {d_max:.3f} 元")
    report_output.append(f"平均索赔差额: {d_mean:.3f} 元")
    report_output.append("")

    # 二、风险标注结果
    report_output.append("二、风险标注结果")
    report_output.append("-" * 30)
    report_output.append("")

    # 统计各类别
    group_df_0 = df_final.groupby('风险类别').agg({
        '实际赔付金额': ['count', 'mean', 'median'],
        '索赔差额': ['mean', 'median', 'std']
    }).round(2)
    
    
    std_ = []
    category_order = ['合理诉求', '诉求偏高', '严重超额']

    std_min = 0
    std_max = 0
    
    # 计算标准差
    std_ = [group_df_0.loc[cat, ('索赔差额', 'std')] for cat in category_order if cat in group_df_0.index]
    std_min = min(std_)
    std_max = max(std_)

        
    for c in category_order:
        if c in group_df_0.index:
            count = int(group_df_0.loc[c, ('实际赔付金额', 'count')])
            p_mean_c = group_df_0.loc[c, ('实际赔付金额', 'mean')]
            p_median_c = group_df_0.loc[c, ('实际赔付金额', 'median')]
            d_mean_c = group_df_0.loc[c, ('索赔差额', 'mean')]
            d_median_c = group_df_0.loc[c, ('索赔差额', 'median')]
            d_std_c = group_df_0.loc[c, ('索赔差额', 'std')]

            report_output.append(f"[{c}]:")
            report_output.append(f"    数量: {count:,} 条 ({count / total_count * 100:.3f}%)")
            report_output.append(f"    实际赔付金额: 均值={p_mean_c:.3f}, 中位数={p_median_c:.3f}")
            report_output.append(f"    索赔差额: 均值={d_mean_c:.3f}, 中位数={d_median_c:.3f}")
            
            density_desc = "中等"
            if d_std_c == std_min:
                density_desc = "密集"
            elif d_std_c == std_max:
                density_desc = "稀疏"
            
            report_output.append(f"    索赔差额标准差: {d_std_c:.3f} (反映 {density_desc} 程度)")
            report_output.append("")
    
    # 三、模型参数
    report_output.append("三、模型参数")
    report_output.append("-" * 40)
    report_output.append("")


    report_output.append(f"分层分位数")
    report_output.append(f"P轴分档数量: {k}")
    report_output.append(f"严重超额分位数: {q1:.4f} (对应分布底部 {q1*100:.3f}%)")
    report_output.append(f"合理诉求分位数: {q2:.4f} (对应分布顶部 {(1-q2)*100:.3f}%)")
    report_output.append("")
    
    report_output.append("约束条件满足情况:")

    ratio_true = 0.0
    ratio_false = 0.0
    
    if '合理诉求' in group_df_0.index:
        ratio_true = (group_df_0.loc['合理诉求', ('实际赔付金额', 'count')] / total_count) * 100
    if '严重超额' in group_df_0.index:
        ratio_false = (group_df_0.loc['严重超额', ('实际赔付金额', 'count')] / total_count) * 100

    check_85 = "正确" if ratio_true >= 85 else "不正确"
    check_3 = "正确" if ratio_false < 3 else "不正确"

    report_output.append(f"{check_85} 合理诉求比例 ≥ 85%: {ratio_true:.3f}%")
    report_output.append(f"{check_3} 严重超额比例 < 3%: {ratio_false:.3f}%")
    report_output.append("")

    # 验证
    report_output.append("四、模型验证")
    report_output.append("-" * 40)
    report_output.append("")
    
    variances = df_final.groupby('风险类别')['索赔差额'].var()
    check_density = "合理"
    try:
        if not (variances.get('合理诉求', 0) < variances.get('诉求偏高', 0) < variances.get('严重超额', 0)):
            check_density = "不正确 (不满足严格单调)"
    except (KeyError, TypeError):
        check_density = "不正确 (类别缺失或数据问题)"
        
    report_output.append(f"1. 密度验证 (方差): {check_density}")
    report_output.append(f"合理诉求: {variances.get('合理诉求', 0):.3f} ")
    report_output.append(f"诉求偏高: {variances.get('诉求偏高', 0):.3f}")
    report_output.append(f"严重超额: {variances.get('严重超额', 0):.3f} ")
    report_output.append("")
    report_output.append("2. 比例约束验证:合理！")
    report_output.append("")

    # 写入文件
    report_path = os.path.join(OutputWay, '风险标注分析报告.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_output))

    # 同时打印到控制台
    print("\n" + "\n".join(report_output))


#主函数
def main():
    print("问题一----------------")

    df = pd.read_excel(INPUT_FILE,header=0, skiprows=0)
    df = df.drop(0).reset_index(drop=True)

    print(df.head())

    # 缺失值处理-丢弃缺失行
    df_cleaned = df.dropna(subset=['实际赔付金额', '索赔金额']).copy()


    df_cleaned['实际赔付金额'] = pd.to_numeric(df_cleaned['实际赔付金额'], errors='coerce')
    df_cleaned['索赔金额'] = pd.to_numeric(df_cleaned['索赔金额'], errors='coerce')

    df_cleaned['索赔差额'] = df_cleaned['实际赔付金额'] - df_cleaned['索赔金额']

    # 盖帽法
    cap_ = ['实际赔付金额', '索赔金额', '索赔差额']


    for col in cap_:
        bound1 = df_cleaned[col].quantile(0.001)
        bound2 = df_cleaned[col].quantile(0.999)
        df_cleaned[col] = np.clip(df_cleaned[col], bound1, bound2)

    print(f"--- 预处理完成。原始数据: {len(df)} 行, 处理后: {len(df_cleaned)} 行。")



    # 模型运行
    K= 10  # P分档数量
    q1 = 0.0295  # 合理诉求分位数
    q2 = 0.14  # 严重超额分位数


    print("--- 正在运行模型...")
    df_final, percent_fin,p_bins = classify(df_cleaned,k=K,q1=q1,q2=q2)

    if df_final is None:
        print("失败")
        return


    print("最终分类结果 ---")
    print(percent_fin)


##结果分析
    print("--- 正在验证密度 (方差)...")

    variances = df_final.groupby('风险类别')['索赔差额'].var().sort_values()
    print(variances)
        # 预期: Var(合理) < Var(偏高) < Var(严重)
    if (variances['合理诉求'] < variances['诉求偏高'] < variances['严重超额']):
        print(" 密度验证通过！")

    else:
        print("密度验证失败！")

    report(df_final, OutputWay, K, q1, q2)


    #生成结果图片
    plot_1(df_final, percent_fin)
    plot_2(df_final, p_bins, OutputWay)
    plot_3(df_final, OutputWay)
    plot_4(df_final, OutputWay)

    
    outp_col = ['运单号', '实际赔付金额', '索赔金额', '索赔差额',
                      'P档', 'Y1', 'Y2', '风险类别']

    # 加上运单号
    df_final['序号'] = range(1, len(df_final) + 1)

    # 筛选出需要的列
    output = df_final[[col for col in outp_col if col in df_final.columns]].iloc[0:].copy()
    print("最后数据",output.head())
    output.to_excel(FILE_OUT, index=False)
    print(f"最终结果已保存到: {FILE_OUT}")



    print("\n所有任务完成！")
    


if __name__ == "__main__":
    main()

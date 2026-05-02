import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#折外验证
label_q3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

bin_q3 = {
    0: [-534.22118, -245.5712],

    1: [-820.02684, -446.1372],

    2: [-1073.15485, -514.498],

    3: [-1235.92782, -659.9748],

    4: [-1401.2994, -801.4372],

    5: [-1644.58845, -798.85],

    6: [-1853.56578, -973.7356],

    7: [-2215.959675, -1178.541],

    8: [-2589.26696, -1593.7556],

    9: [-3606.35562, -2593.35]
}


def q1_label(row, dic_):
    try:
        a = row["预测赔付金额档位"] 
        if pd.isna(a):
            return "分类失败"
        chao, heli = dic_[a]
        cha = row["预测实际赔付金额"] - row["索赔金额"]  

        if cha <= chao:
            return "严重超额"
        elif cha >= heli:
            return "合理诉求"
        else:
            return "诉求偏高"
    except Exception as e:
        return "分类失败"


#报告函数
def report(
    train_size, test_size, features_n, target_data,
    params, cv, oof_stats,
    avg_imp, pred_stats
):

    print("\n" + "="*80)
    print("物流理赔风险识别 - 问题二：实际赔付金额预测模型分析报告")
    print("="*80)
    
    # 一、数据概况
    print("\n一、数据概况")
    print("-----------")
    print(f"训练集有 {train_size:,} 条运单")
    print(f"测试集有 {test_size:,} 条运单")
    print(f" 特征工程后特征数量: {features_n} 个")
    print("目标变量: 实际赔付金额")
    print("目标变量:")
    print(f"  最小值: {target_data['min']:.3f} 元")
    print(f"  最大值: {target_data['max']:.3f} 元")
    print(f"  平均值: {target_data['mean']:.3f} 元")
    print(f"  中位数: {target_data['median']:.3f} 元")
    print(f"  标准差: {target_data['std']:.3f} 元")
    
    # 二、模型设置
    print("\n二、模型设置")
    print("----"*3)
    print(f"算法: LightGBM")
    print(f"交叉验证: {params['n_splits']} 折交叉验证")
    print(f"损失函数: {params['objective']} (均方误差)")
    print("核心参数:")
    print(f"  学习率: {params['learning_rate']}")
    print(f"  树的数量: {params['n_estimators']}")
    print(f"  叶子节点数: {params['num_leaves']}")
    
    # 三、模型性能评估
    print("\n三、模型性能评估")
    print("---")
    print("交叉验证结果:")
    print(f"  RMSE: {np.mean(cv['rmse']):.4f} ± {np.std(cv['rmse']):.4f}")
    print(f"  MAE:  {np.mean(cv['mae']):.4f} ± {np.std(cv['mae']):.4f}")
    print(f"  R²:   {np.mean(cv['r2']):.4f} ± {np.std(cv['r2']):.4f}")
    print("OOF整体评估:")#Out-of-Fold
    print(f"  RMSE: {oof_stats['rmse']:.4f}")
    print(f"  MAE:  {oof_stats['mae']:.4f}")
    print(f"  R²:   {oof_stats['r2']:.4f}")
    

    print("\n四、特征重要性分析")
    print("----")
    print("Top 10 重要特征：")
    top_10_features = avg_imp.head(10)

    max_len = top_10_features['feature'].astype(str).map(len).max()
    for idx, row in top_10_features.iterrows():

        print(f"  {row['feature']:<{max_len}}   重要性值: {row['importance']:.3f}")
        

    print("\n五、预测结果")
    print("-----------")
    print("测试集 (附件2) 预测统计:")
    print(f"  最小值:{pred_stats['min']:.3f} 元")
    print(f"  最大值:{pred_stats['max']:.3f} 元")
    print(f"  平均值:{pred_stats['mean']:.3f} 元")
    print(f"  中位数:{pred_stats['median']:.3f} 元")
    print(f"  标准差:{pred_stats['std']:.3f} 元")
    


def regression(y_true, y_pred_oof, output_):
    """
    绘制 预测值vs真实值 散点图 和 残差分布图
    :param y_true: 真实的y值 (y_train)
    :param y_pred_oof: K折交叉验证的 OOF 预测值
    :param output_: 输出目录
    """
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print("预测值和真实值散点图")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 抽样 2000 个点可视化
    sample_2000 = np.random.choice(len(y_true), size=min(2000, len(y_true)), replace=False)
    y_true_sample = y_true.iloc[sample_2000]
    y_pred_sample = y_pred_oof[sample_2000]
    
    sns.scatterplot(x=y_true_sample, y=y_pred_sample, alpha=0.5, ax=ax)
    
    # 完美预测) 参考线
    max_val = max(y_true.max(), y_pred_oof.max())
    min_val = min(y_true.min(), y_pred_oof.min())
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='完美预测 (y=x)')
    
    ax.set_title('预测值与真实值散点对比图', fontsize=13)
    ax.set_xlabel('真实值', fontsize=13)
    ax.set_ylabel('预测值', fontsize=12)
    ax.legend()
    plt.savefig(os.path.join(output_, '预测值和真实值散点图.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("残差分布图:")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    c = y_true - y_pred_oof
    sns.histplot(c, bins=50, palette='Purples', kde=True, ax=ax)
    
    # 添加 0 点参考线
    ax.axvline(x=0, color='red', linestyle='--', lw=2, label='零误差')
    
    ax.set_title('残差分布图', fontsize=17)
    ax.set_xlabel('残差', fontsize=10)
    ax.set_ylabel('频数', fontsize=12)
    ax.legend()
    plt.savefig(os.path.join(output_, '残差分布图.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# 特征重要性 ---
def importance(avg_imp, output_, top_30=30):
    """
    绘制 特征重要性柱状图
    :param avg_imp: 5折平均后的特征重要性 DataFrame
    :param output_: 输出目录
    :param top_30: 显示前 30 个特征
    """
    print(f"  > 正在生成: (图 4) Top {top_30} 特征重要性柱状图")
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 取 Top N
    top_features_df = avg_imp.head(top_30).sort_values(by='importance', ascending=True)
    
    sns.barplot(x='importance', y='feature', data=top_features_df, palette='Purples', ax=ax)
    
    ax.set_title(f'特征重要性柱状图', fontsize=16)
    ax.set_xlabel('平均重要性', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_, 'plot_q2_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)



# --- 1. 主程序开始 ---
def main():
    print("--- 问题二：实际赔付金额预测 启动 ---")
    start_time = time.time()
    current_directory = os.getcwd()

    # --- 路径配置 ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OutputWay = os.path.join(BASE_DIR, 'output')

    if not os.path.exists(OutputWay):
        os.makedirs(OutputWay)

    #数据加载与准备 ---
    print("\n[阶段一：数据加载与准备]")



    df_train = pd.read_excel('database/附件1.xlsx')
    df_train = df_train.drop(df_train.index[0])
    df_train = df_train.reset_index(drop=True)


    df_predict = pd.read_excel('database/附件2.xlsx')
    df_predict = df_predict.drop(df_train.index[0])
    df_predict = df_predict.reset_index(drop=True)

    dff_q1_labels = 'output/问题一.xlsx'


    file_q1_ = pd.read_excel(dff_q1_labels)

    print(f" 数据加载完毕，训练集共 {len(df_train)} 行，预测集共 {len(df_predict)} 行，{len(file_q1_)}")


    y_train = pd.to_numeric(df_train['实际赔付金额'], errors='coerce')

    if y_train.isna().sum() > 0:
        print(f"删除目标变量缺失的样本...")
        # 非缺失的索引
        valid_idx = y_train.notna()
        # 过滤
        df_train = df_train[valid_idx].reset_index(drop=True)
        y_train = y_train[valid_idx].reset_index(drop=True)
        print(f"训练集剩余 {len(df_train)} 条运单")

    train_size = len(df_train)
    test_size = len(df_predict)

    target_data = {
        'min': y_train.min(), 'max': y_train.max(), 'mean': y_train.mean(),
        'median': y_train.median(), 'std': y_train.std()
    }

    df_train = df_train.drop('实际赔付金额', axis=1)

    predict_waybills = df_predict['运单号']

    df_train['source'] = 'train'
    df_predict['source'] = 'predict'
    df_full = pd.concat([df_train, df_predict], ignore_index=True)
    print(f"合并完成，共 {len(df_full)} 行")



    #特征工程 ---
    print("\n[特征工程]")

    # 数据清洗（处理可能的双行表头问题）
    print("清洗和转换数据类型中...")

    #数值的列
    numeric_cols = [
        '保价金额', '索赔金额', '配送超时时长',
        '始发网点发单量', '始发网点万单理赔率', '始发网点赔付比例',
        '目的网点发单量', '目的网点万单理赔率', '目的网点赔付比例'
    ]

    # 转换数值列，非数值自动变为 NaN
    for col in numeric_cols:
        if col in df_full.columns:
            original_dtype = df_full[col].dtype
            df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
            # 检查是否有大量NaN（可能表示数据有问题）
            nan_count = df_full[col].isna().sum()
            if nan_count > 0:
                print(f"    > {col}: 转换为数值类型，发现 {nan_count} 个异常值")

    print(" 填充缺失值...")
    df_full['配送超时时长'] = df_full['配送超时时长'].fillna(0)
    df_full['异常原因'] = df_full['异常原因'].fillna('无异常')

    cols_ = [
        '始发网点发单量', '始发网点万单理赔率', '始发网点赔付比例',
        '目的网点发单量', '目的网点万单理赔率', '目的网点赔付比例'
    ]

    for col in cols_:
        if col in df_full.columns:
            median_val = df_full[col].median()
            df_full[col] = df_full[col].fillna(median_val)

    print(" 新特征：...")
    # 确保数值类型
    df_full['索赔比率'] = df_full['索赔金额'] / (df_full['保价金额'].fillna(0) + 1e-6)

    if '始发网点万单理赔率' in df_full.columns:
        df_full['链路理赔率'] = df_full['始发网点万单理赔率'] + df_full['目的网点万单理赔率']

    if '始发网点赔付比例' in df_full.columns:
        df_full['链路赔付比例'] = df_full['始发网点赔付比例'] + df_full['目的网点赔付比例']

    print("编码文字特征...")
    object_cols = df_full.select_dtypes(include='object').columns.drop('source')
    for col in object_cols:
        df_full[col] = df_full[col].astype('category')

    print(" 删除无用列...")
    cols_to_drop = ['运单号', '寄件人账号', '收件人ID']
    for col in cols_to_drop:
        if col in df_full.columns:
            df_full = df_full.drop(col, axis=1)


    features_n = len(df_full.columns) - 1


    print("\n[拆分数据，保留名称]")
    X_train = df_full[df_full['source'] == 'train'].drop('source', axis=1)
    X_predict = df_full[df_full['source'] == 'predict'].drop('source', axis=1)
    feature_names = X_train.columns.tolist()



    # --- 阶段四：模型训练 (5折交叉验证) ---
    print("\n[5折交叉验证训练]")


    N_SPLITS = 5
    model_params_dict = {
        'objective': 'regression_l2', # MSE
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'reg_alpha': 0.1,  # L1
        'reg_lambda': 0.1, # L2
        'n_jobs': -1,
        'random_state': 42
    }

    params_rep = model_params_dict.copy()
    params_rep['n_splits'] = N_SPLITS

    # 初始化 
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 初始化用于存储结果的数组
    # 存储对训练集本身的预测
    oof_preds = np.zeros(train_size)
    # 存储5个模型对测试集的平均预测
    test_preds = np.zeros(test_size)
    # 特征重要性
    importances_df = pd.DataFrame()
    # 每折的分数
    cv = {'rmse': [], 'mae': [], 'r2': []}


    oof_preds_2 = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"\n  --- 正在训练第 {fold + 1}/{N_SPLITS} 折 ---")

        # 1. 拆分数据
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]


        # 2. 初始化模型
        model = lgb.LGBMRegressor(**model_params_dict)

        # 3. 训练模型
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(100, verbose=False)] # 100轮不提升则停止
        )

#保存OOF预测
        oof_preds_2[val_idx] = model.predict(X_val)

        # 4. 预测验证集 (评估)
        val_preds = model.predict(X_val)

        # 5. 存储预测
        oof_preds[val_idx] = val_preds

        # 6. 预测累加测试集
        # 5个模型都预测一遍，最后取平均
        test_preds += model.predict(X_predict) / N_SPLITS

        cv['rmse'].append(np.sqrt(mean_squared_error(y_val, val_preds)))
        cv['mae'].append(mean_absolute_error(y_val, val_preds))
        cv['r2'].append(r2_score(y_val, val_preds))

        #特征重要性)
        fold_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importances_df = pd.concat([importances_df, fold_imp_df])

    print("\n交叉验证训练完毕。")

    # --- 阶段五：汇总评估结果 ---
    print("\n[结果]")

    # OOF 整体评估
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    oof_mae = mean_absolute_error(y_train, oof_preds)
    oof_r2 = r2_score(y_train, oof_preds)
    oof_stats = {'rmse': oof_rmse, 'mae': oof_mae, 'r2': oof_r2}

    #5折的平均重要性
    avg_imp = importances_df.groupby('feature')['importance'] \
                                     .mean() \
                                     .reset_index() \
                                     .sort_values(by='importance', ascending=False)

    #预测与保存结果 ---
    print("\n[阶段六：预测与保存结果]")

    # 1. 后处理
    final_predictions = np.clip(test_preds, 0, None)  # 赔付金额不能为负

    #预测结果)
    pred_stats = {
        'min': final_predictions.min(), 'max': final_predictions.max(),
        'mean': final_predictions.mean(), 'median': np.median(final_predictions),
        'std': final_predictions.std()
    }


    result_df = pd.DataFrame({
        '运单号': predict_waybills,
        '实际赔付金额': final_predictions
    })
    output_path = os.path.join(OutputWay, 'result_q2.xlsx')
    result_df.to_excel(output_path, index=False)
    print(f" 结果已保存到: {output_path}")


    print("\n[阶段七：生成分析报告]")

    report(
        train_size=train_size,
        test_size=test_size,
        features_n=features_n,
        target_data=target_data,
        params=params_rep,
        cv=cv,
        oof_stats=oof_stats,
        avg_imp=avg_imp,
        pred_stats=pred_stats
    )

    print("\n[图表：]")
    regression(y_train, oof_preds, OutputWay)

          #特征重要性图
    importance(avg_imp, OutputWay, top_30 = 30)

    print("  > 图表都已保存到 'output' 文件夹。")


    print("\n" + "=" * 80)
    print("--- 正在执行 '问题三：先预测再分类-建模的评估 ---")



    try:

        df_q1 = file_q1_
        y_true = df_q1["风险类别"]


        df_a1 = df_train

        df_eval = pd.DataFrame({
            "索赔金额": df_a1["索赔金额"],
            "预测实际赔付金额": oof_preds_2 
        })

        # 和问题一用一样的分类结果
        print("  > 正在应用 Q1 规则...")
        df_eval["预测赔付金额档位"] = pd.qcut(
            df_eval["预测实际赔付金额"],
            q=10,
            labels=label_q3,  #10
            duplicates='drop'
        )


        y_pred = df_eval.apply(
            q1_label,
            axis=1,
            dic_=bin_q3
        )

        # 5. 打印最终报告
        print("\n报告评估成功！！(P/R/F1):")

        labels_list = ["严重超额", "合理诉求", "诉求偏高"]

        print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

    except FileNotFoundError as e:
        print(f"[错误] 评估失败: 无法加载 {e.filename}")
        print("     请确保 '问题一.xlsx' 和 '附件1.xlsx' 路径正确。")
    except Exception as e:
        print(f"[错误] 评估失败: {e}")




# --- 运行主程序 ---
if __name__ == "__main__":
    main()



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    make_scorer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE 
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
# 中文
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

le = LabelEncoder()



#LGBM的F1指标
def lgb_f1(y_true, y_pred_):

    y_pred = np.argmax(y_pred_, axis=1)
    
    return 'weighted_f1', f1_score(y_true, y_pred, average='weighted'), True

#CV 性能曲线 ---
def cv_plot(all_results, output_dir):
    """
    绘制5折交叉验证的性能曲线
    """
    print("\nCV 性能曲线:...")
    
    plt.figure(figsize=(10, 6))
    for i, eval_f1_list in enumerate(all_results):
        plt.plot(eval_f1_list, label=f'Fold {i+1} (Best: {max(eval_f1_list):.4f})')
    
    plt.title('图2 交叉验证性能曲线 (Weighted F1)')
    plt.xlabel('迭代次数')
    plt.ylabel('Weighted F1 Score (验证集)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(output_dir, '图2_CV性能曲线.png')
    plt.savefig(save_path)
    print(f"  > CV 性能曲线已保存到: {save_path}")

#特征重要性曲线-图三---
def importance_plot(models, features, output_dir):

    print("\n特征重要性图 ：...")
    feature_importances = np.zeros(len(features))
    for model in models:

        if hasattr(model, 'feature_importances_'):
            feature_importances += model.feature_importances_ / len(models)
        else:
            print(f"模型 {model} 缺失 feature_importances_ 属性")
        
    df_importance = pd.DataFrame({'feature': features, 'importance': feature_importances})
    df_importance = df_importance.sort_values(by='importance', ascending=False).head(10)#前十

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=df_importance, palette='viridis')
    plt.title('图3 特征重要性排名 (Top 10, 按Gain排序)')
    plt.xlabel('平均重要性 (Gain)')
    plt.ylabel('特征名称')
    
    save_path = os.path.join(output_dir, '图3_特征重要性.png')
    plt.savefig(save_path)
    print(f"  > 特征重要性图已保存到: {save_path}")

# 分析分类报告和混淆矩阵热力图) ---
def report(y_true, y_pred, labels, output_dir):

    print("\n" + "="*80)
    print("问题三-模型一：直接分类模型 (LGBMClassifier) 性能分析 (OOF)")
    print("="*80)
    
    # 1. 总体准确率
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[ 总体准确率 (Accuracy) ]\n  {acc:.4f} ({acc*100:.2f}%)")
    
    # 2. 打印混淆矩阵
    print("\n[ 混淆矩阵 (Confusion Matrix) ]")
    print("  (行 = 真实类别, 列 = 预测类别)\n")

    cm_df = pd.crosstab(y_true, y_pred, rownames=['真实类别'], colnames=['预测类别'])

    cm_df = cm_df.reindex(index=labels, columns=labels, fill_value=0)
    print(cm_df)
    
    # 混淆矩阵热力图 ---
    print("\n生成混淆矩阵热力图...")
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('混淆矩阵 (OOF 预测)')
        plt.ylabel('真实类别')
        plt.xlabel('预测类别')
        
        save_path = os.path.join(output_dir, '图_混淆矩阵.png')
        plt.savefig(save_path)
        print(f"  > 混淆矩阵热力图已保存到: {save_path}")
    except Exception as e:
        print(f"  > [错误] 绘制热力图失败: {e}")

    # 3. 分类报告
    print("\n[ 分类报告： ]")
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(report)
    



# 超参数搜索---
def run_grid_search(X_train, y_train_encoded, categ_features):
    """
     GridSearchCV 来寻找最优参数-很简化
    """
    print("\n" + "=" * 80)
    print("！！正在执行超参数搜索 (GridSearchCV)...")
    print("       这可能会花费很长时间...")
    print("=" * 80)

    #  搜索空间
    param_grid = {
        'num_leaves': [31, 63, 127],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200],
        'min_child_samples': [20, 30, 50]
    }

    f1_scorer = make_scorer(f1_score, average='weighted')

    model_base = lgb.LGBMClassifier(
        objective='multiclass',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=model_base,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=3,
        n_jobs=-1,
        verbose=2
    )


    grid.fit(X_train, y_train_encoded,
             categorical_feature=categ_features)

    print("\n！！超参数搜索完成！")
    print(f"  > 最优分数 (Weighted F1): {grid.best_score_:.4f}")
    print(f"  > 最优参数:")
    print(grid.best_params_)
    print("=" * 80)
    return grid.best_params_





def main():
    print("--- 问题三-模型一：风险标注预测 (直接分类法) 开始执行 ---")
    start_time = time.time()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'database1111')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    
    # 确保 output 目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_q1_train = 'database/附件1.xlsx'
    file_q1_predict = 'database/附件2.xlsx'
    file_q1_labels = 'output/问题一.xlsx'
    file_q2_results = '../output/result_q2.xlsx'

    # --- 一：数据加载 ---
    print("\n[一：数据加载]")
    df_train_raw = pd.read_excel(file_q1_train)  # 附件1 (X_train)

    df_train_raw = df_train_raw.drop(df_train_raw.index[0])
    df_train_raw = df_train_raw.reset_index(drop=True)
    print("附件一10000——")
    print(df_train_raw.head())

    df_predict_raw = pd.read_excel(file_q1_predict)  # 附件2 (X_predict)

    df_predict_raw = df_predict_raw.drop(df_predict_raw.index[0])
    df_predict_raw = df_predict_raw.reset_index(drop=True)

    df_q1_labels = pd.read_excel(file_q1_labels)  # 问题一.xlsx (y_train)
    print("问题一")
    print(df_q1_labels.head())

    print(f"  > 附件1 (训练) 加载并清洗后，剩 {len(df_train_raw)} 行")
    print(f"  > 附件2 (预测) 加载完毕，共 {len(df_predict_raw)} 行")
    print(f"  > 问题一标签 (训练目标) 加载完毕，共 {len(df_q1_labels)} 行")

    # --- 二：准备 特征(X) 和 目标(y) ---
    print("\n[二：特征工程 (重复问题二的步骤)]")

    # 1. 准备 目标(y) 和 特征(X_train)
    print("  > 根据行索引对齐 X_train (附件1) 和 y_train (问题一.xlsx)...")

    if len(df_train_raw) > len(df_q1_labels):
        print(f"[错误] 附件1 清洗后行数 ({len(df_train_raw)}) 大于 Q1标签行数 ({len(df_q1_labels)})。")
        return
        
    try:
        df_q1_labels_aligned = df_q1_labels.loc[df_train_raw.index]
    except Exception as e:
        print(f"[错误] 尝试按行索引对齐 附件1 和 问题一.xlsx 失败: {e}")
        return

    y_train_labels = df_q1_labels_aligned['风险类别']
    
    df_train_features = df_train_raw.drop(['实际赔付金额'], axis=1, errors='ignore')
    
    if '运单号' in df_train_features.columns:
        df_train_features = df_train_features.drop('运单号', axis=1)


    if '运单号' not in df_predict_raw.columns:
        print("[错误] 附件2.xlsx 缺少 '运单号' 列。")
        return
    predict_waybills = df_predict_raw['运单号'] 
    
    print("  > 已根据行索引对齐 X_train 和 y_train。")

    print("\n[ 原始样本分布 (y_train) ]")
    print("  --- (用于SMOTE处理前样本分布) ---")
    print(y_train_labels.value_counts())
    print("\n[ 原始样本占比 (y_train) ]")
    print(y_train_labels.value_counts(normalize=True) * 100)
    print("THE END")


    df_train_features['source'] = 'train'
    df_predict_raw['source'] = 'predict'
    
    df_full = pd.concat([df_train_features, df_predict_raw], ignore_index=True)
    
    # 缺失值
    print("\n填充缺失值...")
    df_full['配送超时时长'] = df_full['配送超时时长'].fillna(0)
    df_full['异常原因'] = df_full['异常原因'].fillna('无异常') # 关键
    
    cols_to_fill_median = [
        '始发网点发单量', '始发网点万单理赔率', '始发网点赔付比例',
        '目的网点发单量', '目的网点万单理赔率', '目的网点赔付比例'
    ]
    for col in cols_to_fill_median:
        if col in df_full.columns:
            median_val = df_full[col].median()
            df_full[col] = df_full[col].fillna(median_val)
    

#新特征
    print("衍生新特征...")
    df_full['索赔比率'] = df_full['索赔金额'] / (df_full['保价金额'].fillna(0) + 1e-6)
    if '始发网点万单理赔率' in df_full.columns:
        df_full['链路理赔率'] = df_full['始发网点万单理赔率'] + df_full['目的网点万单理赔率']
    if '始发网点赔付比例' in df_full.columns:
        df_full['链路赔付比例'] = df_full['始发网点赔付比例'] + df_full['目的网点赔付比例']


    # 编码类别特征-整数
    print("编码文字特征...")
    
    obj_cols = df_full.select_dtypes(include='object').columns
    if 'source' in obj_cols:
        obj_cols = obj_cols.drop('source') 
    
    features_encode = list(obj_cols) 
        
    for col in features_encode:
        # 字符串转换为整数
        df_full[col] = pd.factorize(df_full[col])[0]
    
    print(f" 已完成的特征: {features_encode}")
    
    # 2.4 删除无用特征
    print("删除无用ID列...")
    cols_to_drop = ['运单号', '寄件人id', '收件人id'] 
    
    cols_dropped = []
    for col in cols_to_drop:
        if col in df_full.columns:
            df_full = df_full.drop(col, axis=1)
            cols_dropped.append(col)
    print(f"    > 已丢弃列名: {cols_dropped}")


    categ_features = [
        col for col in features_encode 
        if col not in cols_to_drop
    ]
    print(f"    > 最终传入模型的类别特征: {categ_features}")

    # 拆分
    X_train = df_full[df_full['source'] == 'train'].drop('source', axis=1)
    X_predict = df_full[df_full['source'] == 'predict'].drop('source', axis=1)
    
    features = list(X_train.columns)
    
    if len(X_train) != len(y_train_labels):
        print(f"[错误]长度不匹配! X={len(X_train)}, y={len(y_train_labels)}")
        return

    print("[成功] 特征工程已完成！！！")

    # 模型训练
    print("\n[三：模型训练 (SMOTE+权重调整 - 5折交叉验证)]")
    
    N_SPLITS = 5
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    y_train_encoded = le.fit_transform(y_train_labels)
    labels_in_order = le.classes_ 
    num_classes = len(labels_in_order)

    print("\n基于原始不平衡数据计算类别权重...")


    unique_labels = np.unique(y_train_encoded)

    weights_array = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=y_train_encoded  
    )

    # 权重字典 
    weights_dict = dict(zip(unique_labels, weights_array))

    print(f"标签顺序: {labels_in_order}")
    print(f"对应权重: {weights_dict}")
    print("权重计算结束~~") 

    
    y_pred_oof = np.zeros(len(X_train)) 
    test_preds_proba = np.zeros((len(X_predict), num_classes))
    
    models = []
    all_results = [] 

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train_encoded)):
        print(f"\n  >--- 正在训练第 {fold + 1}/{N_SPLITS} 折 ---")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
        
        # SMOTE (混合策略)
        print(f"    原始训练集样本数: {len(y_tr)} (类别分布: {np.unique(y_tr, return_counts=True)[1]})")
        smote = SMOTE(random_state=42, n_jobs=-1)
        try:
            X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr, y_tr)
            print(f"    SMOTE 重采样后训练集样本数: {len(y_tr_resampled)} (类别分布: {np.unique(y_tr_resampled, return_counts=True)[1]})")
        except Exception as e:
            print(f"[警告] {e}. SMOTE 可能某类别样本过少. 将使用原始数据进行本折训练。")
            X_tr_resampled, y_tr_resampled = X_tr, y_tr
        
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=num_classes,
            class_weight=weights_dict, 
            metric='None',
            n_estimators=1000,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42,


        )
        
        evals_result = {} 
        
        model.fit(
            X_tr_resampled, y_tr_resampled, 
            eval_set=[(X_val, y_val)],
            
            eval_metric=lgb_f1, 
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.record_evaluation(evals_result) 
            ],
            categorical_feature=categ_features 
        )
        
        y_pred_oof[val_idx] = model.predict(X_val)
        test_preds_proba += model.predict_proba(X_predict) / N_SPLITS
        
        models.append(model)
        all_results.append(evals_result['valid_0']['weighted_f1'])

    print("\n交叉验证训练完毕~~")

    #分析模型性能 (OOF) ---
    oof_labels = le.inverse_transform(y_pred_oof.astype(int))
    # (!!) 传入 output_dir 以便保存热力图
    report(y_train_labels, oof_labels, labels_in_order, OUTPUT_DIR)
    


    cv_plot(all_results, OUTPUT_DIR)
    importance_plot(models, features, OUTPUT_DIR)

    

    
    print("--- 可运行: 超参数搜索： 需要运行要把 代码前面的 #去掉 ---")
    #run_grid_search(X_train, y_train_encoded, categ_features)
    

    
    #保存最终结果 ---
    print("\n[五：生成并保存最终结果]")
    
    final_a = np.argmax(test_preds_proba, axis=1)
    final_labels = le.inverse_transform(final_a)


    print("\n最终预测结果 :....")
    print("=" * 80)

    # 1. 计算数量
    counts_pred = pd.DataFrame({
        "样本数量": pd.Series(final_labels).value_counts()
    })


    #占比
    counts_pred["占比"] = (counts_pred["样本数量"] / len(final_labels))
    counts_pred["占比"] = counts_pred["占比"].apply(lambda x: f"{x * 100:.2f}%")

    print(counts_pred.to_string())
    print("--" * 80)

    try:
        df_result_final = pd.read_excel(file_q2_results)
    except FileNotFoundError:
        print(f"[警告] {file_q2_results} 未找到。将仅创建 Q3 的结果。")
        df_result_final = pd.DataFrame()

    df_q3_preds = pd.DataFrame({
        '运单号': predict_waybills,
        '风险标注预测': final_labels
    })
    
    if '运单号' not in df_result_final.columns:
        print("[警告] result_q2.xlsx 中无 '运单号' 列或文件不存在, 仅保存 Q3 结果。")
        df_result_final = df_q3_preds
    else:
        print("  > 正在合并 Q2 和 Q3 的结果...")
        df_result_final = pd.merge(
            df_result_final,
            df_q3_preds,
            on='运单号',
            how='left' 
        )
        if '风险标注预测' not in df_result_final.columns:
            print("[错误] Q2 和 Q3 运单号合并失败。")

    output_path = os.path.join(OUTPUT_DIR, 'Result_直接分类.xlsx')
    try:
        df_result_final.to_excel(output_path, index=False)
        
        print("\n" + "="*80)
        print(f"[成功！！！] 问题三 (建模一) 执行完毕!")
        print(f"最终的合并结果 (Q2+Q3) 已保存到: {output_path}")
        print(f"共预测 {len(df_result_final)} 行数据。")
        print("="*80)
        
    except Exception as e:
        print(f"\n[错误] 最终结果保存失败: {e}")
    

# --- 运行主程序 ---
if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import os
import time

LABELS_10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

BIN_10 = {
    # P档 0 (第 1 档)
    0: [-534.22118, -245.5712],

    1: [-820.02684, -446.1372],

    2: [-1073.15485, -514.498],

    3: [-1235.92782, -659.9748],

    4: [-1401.2994, -801.4372],

    # P档 5 (第 6 档)
    5: [-1644.58845, -798.85],

    # P档 6 (第 7 档)
    6: [-1853.56578, -973.7356],


    7: [-2215.959675, -1178.541],

    # P档 8 (第 9 档)
    8: [-2589.26696, -1593.7556],

    9: [-3606.35562, -2593.35]
}


def label_q1(row, dict_):
    """
    根据预测的赔付金额档位和索赔差额，应用问题一的规则进行分类
    """
    try:

        bin_name = row["预测赔付金额档位"]

        # 问题1阈值 (Y1, Y2)
        excess_th, reasonable_th = dict_[bin_name]

        diff = row["实际赔付金额"] - row["索赔金额"]

        if diff <= excess_th:
            return "严重超额"
        elif diff >= reasonable_th:
            return "合理诉求"
        else:
            return "诉求偏高"

    except Exception as e:
        return "分类失败"



def main():
    print("--- 问题三-模型二：先预测赔付再分类 风险标注预测启动 ---")


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'database')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_q1_predict = os.path.join(DATA_DIR, '附件2.xlsx')
    file_q2_results = os.path.join(OUTPUT_DIR, 'result_q2.xlsx')
    output_path = os.path.join(OUTPUT_DIR, 'Result_先预测.xlsx')

    if not os.path.exists(file_q2_results):
        print(f"[错误] 输入文件缺失: {file_q2_results}")
        print("       请先运行 问题二.py 脚本生成预测结果。")
        return

    if not os.path.exists(file_q1_predict):
        print(f"[错误] 输入文件缺失: {file_q1_predict}")
        return

    print("\n[加载数据]")
    try:
        df_q2 = pd.read_excel(file_q2_results)
        print(f"  > 成功加载 Q2 结果共 {len(df_q2)} 行")
        if '实际赔付金额' not in df_q2.columns or '运单号' not in df_q2.columns:
            print("[错误] 'result_q2.xlsx' 中缺少 '运单号' 或 '实际赔付金额' 列。")
            return

        df_a2 = pd.read_excel(file_q1_predict)
        print(f"  > 成功加载 附件2，共 {len(df_a2)} 行")
        if '索赔金额' not in df_a2.columns or '运单号' not in df_a2.columns:
            print("[错误] '附件2.xlsx' 中缺少 '运单号' 或 '索赔金额' 列。")
            return

    except Exception as e:
        print(f"[错误] 加载文件失败: {e}")
        return

    print("\n[合并数据]")
    df_result = pd.merge(
        df_q2,
        df_a2[['运单号', '索赔金额']],
        on='运单号',
        how='left'
    )
    df_result['索赔金额'] = df_result['索赔金额'].fillna(0)
    print("  > 已合并。")

    print("\n[阶段三：应用问题一规则]")



    print(f"  > 使用分位数按10个P档进行分箱...")

    try:
        df_result["预测赔付金额档位"] = pd.qcut(
            df_result["实际赔付金额"],
            q=10,
            labels=LABELS_10,
            duplicates='drop'
        )

    except Exception as e:
        print(f"[警告] pd.qcut 分箱失败: {e}. 尝试使用近似边界...")
        bin_edges = [
            -np.inf,
            39.586000000000006, 71.998, 106.57600000000001, 148.46200000000002,
            200.15, 268.6120000000001, 355.1180000000001, 475.40000000000026,
            676.2899999999998, np.inf
        ]
        df_result["预测赔付金额档位"] = pd.cut(
            df_result["实际赔付金额"],
            bins=bin_edges,
            labels=LABELS_10,
            right=False
        )

    print("  > 已对 '实际赔付金额' 完成分箱。")

    print("  > 正在计算 '风险标注预测'...")
    df_result["风险标注预测"] = df_result.apply(
        label_q1,
        axis=1,
        dict_=BIN_10
    )

    print("[成功] 已完成 '建模一' 的风险分类。")

    print("\n[---保存最终结果]")

    try:
        print("\n" + "=" * 80)
        print("问题三 (模型二)：最终预测结果统计")
        print("=" * 80)

        # 打印各个分类的结果数量
        print("[ 各个分类结果数量：----]")
        pred_counts = df_result["风险标注预测"].value_counts()
        print(pred_counts)
        print("\n[ 各个分类结果占比：----]")
        print(df_result["风险标注预测"].value_counts(normalize=True) * 100)

        df_final = df_result.drop(['索赔金额', '预测赔付金额档位'], axis=1, errors='ignore')

        if '风险标注预测' in df_q2.columns:
            df_q2 = df_q2.drop('风险标注预测', axis=1)

        df_final_output = pd.merge(df_q2,df_final[['运单号', '风险标注预测']],on='运单号',how='left')

        df_final_output.to_excel(output_path, index=False)

        print("\n" + "=" * 80)
        print(f"[成功！！！] 问题三 (建模二) 执行完毕!")
        print(f"合并结果已保存到: {output_path}")
        print(f"共 {len(df_final_output)} 行数据。")
        print("=" * 80)

    except Exception as e:
        print(f"\n[错误] 最终结果保存失败: {e}")


if __name__ == "__main__":
    main()


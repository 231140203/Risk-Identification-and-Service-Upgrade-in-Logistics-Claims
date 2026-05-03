# 基于多模型融合的物流理赔风险识别与赔付预测系统

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)
![SMOTE](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-orange.svg)

## 📌 项目背景 (Background)

本项目为 **2025年 MathorCup 数学应用挑战赛（大数据赛道 B题）** 参赛代码仓库（荣获国家级一等奖）。

在物流理赔环节中，全靠人工审核成本极高且效率低下，且“骗保（严重超额）”等高危单据隐藏在海量正常运单中。本项目旨在利用历史理赔数据，构建一套智能化、端到端的风控系统。系统不仅能自动对运单进行风险打标，还能高精度预测未来的实际赔付金额，并在极度不平衡的数据分布下，实现对高风险运单的精准预警拦截。

---

## ✨ 核心创新点 (Core Innovations)

1. **业务驱动的动态分层规则引擎 (Dynamic Stratified Thresholding)**
   摒弃了静态阈值，创新性地采用“分层分位数回归”算法。根据赔付金额动态调整容忍度区间，从数学底层完美硬解了业务方要求“合理诉求≥85%，骗保<3%”的严苛比例约束。
2. **克服“类别极度不平衡”的混合采样技术 (Hybrid Resampling Strategy)**
   针对 <3% 的高风险（严重超额）样本，采用 `SMOTE` 数据层过采样与 `代价敏感学习 (Class Weight)` 算法层加权的混合策略，使得高危运单召回率大幅提升 24.8%。
3. **基于业务逻辑的深度特征工程 (Deep Feature Engineering)**
   运用目标编码（Target Encoding）处理高基数城市特征，并自主衍生出“链路理赔率”、“索赔保价比”等高阶量化指标，极大提升了模型解释性与预测上限。
4. **“黑盒+白盒”双路径验证架构 (Dual-path Risk Evaluation)**
   用端到端的直接分类法（黑盒）保障高危风险拦截，用“先预测后划分”的间接法（白盒）保障理赔结果透明可追溯。

---

## 📁 核心文件结构与说明 (File Structure)

本仓库包含了完整的模型管线代码，按业务逻辑顺序执行如下：

| 文件名 | 功能描述 |
| :--- | :--- |
| `问题一_动态风险标注主程序.py` | **数据清洗与动态打标：** 包含 99.9%“盖帽法”数据平滑处理、内存监控与数据校验体系。执行 `pd.qcut` 分层分位数算法，生成自适应动态风险阈值并完成数据自动化标注。 |
| `问题二.py` | **高精度回归预测：** 执行全局深度特征工程。训练 `LightGBM` 与 `XGBoost` 回归模型，通过 5折交叉验证优化超参数，输出预测赔付金额及特征重要性评估。 |
| `问题三_直接分类.py` | **不平衡数据分类（黑盒路径）：** 利用 `SMOTE` 合成少数类样本，结合动态代价敏感权重，训练多分类模型。直接端到端预测未知运单的风险等级。 |
| `问题三_先预测.py` | **间接分类验证（白盒路径）：** 读取问题二的金额预测结果，映射回问题一的动态规则档位。通过 If-Else 逻辑进行硬分类，并输出与直接分类法的对比验证分析报告。 |

---

## 🛠️ 环境依赖与运行说明 (Environment & Execution)

### 1. 依赖安装
本项目使用 Python 3.9+ 编写。请在运行前安装核心依赖库：
```bash
pip install pandas numpy lightgbm xgboost scikit-learn imbalanced-learn matplotlib seaborn tabulate

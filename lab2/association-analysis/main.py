import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from typing import List, Tuple

class AssociationAnalyzer:
    """
    一个用于执行关联分析的工具类。
    封装了不同任务的分析流程。
    """

    def run_basket_analysis(self, min_sup: float = 0.3, min_conf: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        任务1: 对购物篮数据进行Apriori分析。
        
        :param min_sup: 最小支持度阈值。
        :param min_conf: 最小置信度阈值。
        :return: 包含频繁项集和关联规则的元组。
        """
        # 数据集定义
        dataset: List[List[str]] = [
            ['牛奶', '面包', '黄油'], ['牛奶', '啤酒', '尿布'], ['面包', '黄油', '饼干'],
            ['牛奶', '尿布', '饼干'], ['啤酒', '尿布'], ['牛奶', '尿布', '面包', '黄油'],
            ['啤酒', '饼干'], ['啤酒', '尿布', '饼干'], ['牛奶', '尿布', '面包', '黄油'],
            ['尿布', '面包', '黄油']
        ]

        # 数据格式转换: list -> one-hot
        te = TransactionEncoder()
        df_encoded = pd.DataFrame(te.fit_transform(dataset), columns=te.columns_)

        # 核心算法: 查找频繁项
        frequent_itemsets = apriori(df_encoded, min_support=min_sup, use_colnames=True)

        # 规则生成: 从频繁项中提取
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

        return frequent_itemsets, rules

    def run_book_analysis(self, path: str = '../source/MLBOOK_DATA.xlsx') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        任务2: 对图书数据集进行关联分析。
        
        :param path: Excel文件路径。
        :return: 包含频繁项集和关联规则的元组。
        """
        # 从文件加载数据
        raw_df = pd.read_excel(path)
        
        # 特征选择
        cols_to_analyze: List[str] = ['价格描述', '出版社', '作者', '包装', '畅销程度']
        
        # 数据预处理: 提取并清理
        processed_df = raw_df[cols_to_analyze].fillna('未知')
        transactions: List[list] = processed_df.values.tolist()

        # 数据编码: 转换为布尔矩阵
        te = TransactionEncoder()
        df_encoded = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

        # 运行Apriori算法 (FP-growth好像也行，但这个更基础)
        frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

        # 生成所有可能的规则
        all_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        
        # 结果排序: 按置信度排序以找到"最强"的规则
        final_rules = all_rules.sort_values(by='confidence', ascending=False)

        return frequent_itemsets, final_rules


# 主程序入口
if __name__ == "__main__":
    analyzer = AssociationAnalyzer()

    print("========== 任务1: 购物篮分析 ==========")
    basket_freq, basket_rules = analyzer.run_basket_analysis()
    print("\n--- 频繁项集 ---")
    print(basket_freq)
    print("\n--- 关联规则 (min_support=0.3, min_confidence=0.7) ---")
    print(basket_rules)

    print("\n\n========== 任务2: 图书数据分析 ==========")
    book_freq, book_rules = analyzer.run_book_analysis()
    print("\n--- 频繁项集 (Top 10) ---")
    print(book_freq.head(10))
    print("\n--- 关联规则 (min_support=0.05, min_confidence=0.6) (Top 10) ---")
    print(book_rules.head(10))


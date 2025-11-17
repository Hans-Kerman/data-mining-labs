import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def run_task_1():
    """
    执行实验任务1：
    利用Apriori算法对预设的购物篮数据进行关联分析。
    """
    print("=" * 50)
    print("🚀 开始执行实验任务1: 基础Apriori关联分析")
    print("=" * 50)

    # 1. 实验数据：根据文档中的“表1.事务表”创建数据集
    dataset = [
        ['牛奶', '面包', '黄油'],
        ['牛奶', '啤酒', '尿布'],
        ['面包', '黄油', '饼干'],
        ['牛奶', '尿布', '饼干'],
        ['啤酒', '尿布'],
        ['牛奶', '尿布', '面包', '黄油'],
        ['啤酒', '饼干'],
        ['啤酒', '尿布', '饼干'],
        ['牛奶', '尿布', '面包', '黄油'],
        ['尿布', '面包', '黄油']
    ]

    print(f"原始交易数据共 {len(dataset)} 条。")

    # 2. 数据预处理：将交易数据转换为Apriori算法要求的 one-hot 编码格式
    # TransactionEncoder 可以将 [['a','b'], ['b']] 转换为：
    #    a      b
    # 0  True   True
    # 1  False  True
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print("\n[步骤1: 数据预处理完成]")
    print("转换后的 one-hot 编码数据样本:")
    print(df.head())

    # 3. 挖掘频繁项集：使用 Apriori 算法
    # min_support=0.3 表示最小支持度为 30%
    # 在10个事务中，一个项集至少要出现 10 * 0.3 = 3 次
    min_support_threshold = 0.3
    frequent_itemsets = apriori(df, min_support=min_support_threshold, use_colnames=True)

    print(f"\n[步骤2: 使用Apriori挖掘频繁项集 (最小支持度={min_support_threshold:.0%})]")
    print("找到的频繁项集如下:")
    print(frequent_itemsets)

    # 4. 生成关联规则
    # min_threshold=0.7 表示最小置信度为 70%
    min_confidence_threshold = 0.7
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_threshold)

    print(f"\n[步骤3: 从频繁项集中生成强关联规则 (最小置信度={min_confidence_threshold:.0%})]")

    if rules.empty:
        print("在当前置信度阈值下，未发现强关联规则。")
    else:
        # 整理输出，使其更易读
        rules['antecedents'] = rules['antecedents'].apply(lambda a: ', '.join(list(a)))
        rules['consequents'] = rules['consequents'].apply(lambda c: ', '.join(list(c)))
        
        print("发现的强关联规则如下:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    print("\n✅ 实验任务1完成！")


def run_task_2():
    """
    执行实验任务2：
    对 'MLBOOK_DATA.xlsx' 数据集进行关联分析。
    """
    print("\n" + "=" * 50)
    print("🚀 开始执行实验任务2: 图书数据关联分析")
    print("=" * 50)

    # 1. 加载数据
    data_path = '../source/MLBOOK_DATA.xlsx' # 使用相对路径访问上一级目录的source文件夹
    try:
        df_book = pd.read_excel(data_path)
    except FileNotFoundError:
        print(f"错误: 无法在 '{data_path}' 找到数据文件。请确保文件路径正确。")
        return

    print(f"成功加载数据，共 {len(df_book)} 条记录。")
    print("原始数据样本:")
    print(df_book.head())

    # 2. 数据预处理
    # 选取需要分析的列
    # ✅【修正点】将英文列名替换为Excel文件中的实际中文列名
    columns_to_analyze = ['价格描述', '出版社', '作者', '包装', '畅销程度']
    
    # 检查列是否存在，避免再次出错
    missing_cols = [col for col in columns_to_analyze if col not in df_book.columns]
    if missing_cols:
        print(f"\n错误: 以下列名在数据文件中不存在: {missing_cols}")
        print(f"请检查您的Excel文件，可用的列有: {list(df_book.columns)}")
        return
        
    df_analysis = df_book[columns_to_analyze].copy()

    # 处理缺失值：将 NaN 填充为 '未知'，这样 '未知' 本身也可以作为一个分析项
    df_analysis.fillna('未知', inplace=True)
    
    # 构造交易数据：将每一行（代表一本书）的属性值作为一个交易列表
    # 例如，一行 ['较高', '清华大学出版社', '平装'] 会变成一个 "购物篮"
    transactions = df_analysis.apply(lambda row: row.tolist(), axis=1).tolist()

    print("\n[步骤1: 数据预处理完成]")
    print("转换后的交易数据样本 (前5条):")
    for t in transactions[:5]:
        print(t)

    # 3. 转换为 one-hot 编码
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    # 4. 挖掘频繁项集 (使用 FP-growth，因为它通常比 Apriori 更快)
    # 对于真实数据集，支持度阈值通常需要设置得比较低才能发现有意义的模式
    min_support_threshold = 0.05  # 设定为 5%，可根据需要调整
    frequent_itemsets = fpgrowth(df_onehot, min_support=min_support_threshold, use_colnames=True)

    print(f"\n[步骤2: 使用FP-growth挖掘频繁项集 (最小支持度={min_support_threshold:.0%})]")
    print(f"共找到 {len(frequent_itemsets)} 个频繁项集。")

    # 5. 生成并分析关联规则
    min_confidence_threshold = 0.6  # 设定为 60%，可根据需要调整
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_threshold)

    print(f"\n[步骤3: 生成关联规则 (最小置信度={min_confidence_threshold:.0%})]")

    if rules.empty:
        print("在当前阈值下，未发现强关联规则。请尝试降低 'min_support' 或 'min_confidence'。")
    else:
        # 为了更好地分析，我们关注提升度(lift) > 1 的规则
        # lift > 1 表示 A 的出现对 B 的出现有积极的促进作用
        strong_rules = rules[rules['lift'] > 1].sort_values(by='lift', ascending=False)
        
        # 整理输出
        strong_rules['antecedents'] = strong_rules['antecedents'].apply(lambda a: ', '.join(list(a)))
        strong_rules['consequents'] = strong_rules['consequents'].apply(lambda c: ', '.join(list(c)))

        print(f"共发现 {len(strong_rules)} 条提升度(lift)>1 的强关联规则。")
        print("\n💡 关联规则分析 (按提升度降序排列，展示前20条):")
        print("解读提示: 'antecedents' -> 'consequents' 意味着购买了前项的顾客，也很可能购买后项。")
        print("'lift' (提升度) > 1 表示正相关，值越大，关联性越强。")
        
        print(strong_rules.head(20)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        # 结果分析示例
        print("\n--- 结果分析示例 ---")
        print("1. 比如我们可能发现规则 {出版社: 机械工业出版社} -> {包装: 平装}，并且lift很高。")
        print("   这可能意味着：'机械工业出版社' 出版的书绝大多数都是 '平装' 的，这是一个非常强的内在属性关联。")
        print("2. 又比如规则 {价格描述: 较高, 畅销程度: 畅销} -> {包装: 精装}。")
        print("   这可能说明：高价位的畅销书通常会采用 '精装' 的包装策略来匹配其市场定位。")
        print("--------------------")
        
    print("\n✅ 实验任务2完成！请根据输出结果，撰写你的分析报告。")



if __name__ == "__main__":
    run_task_1()
    run_task_2()

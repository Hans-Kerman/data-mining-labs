import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# 任务1：购物篮
def task1_basket():
    dataset = [
        ['牛奶', '面包', '黄油'], ['牛奶', '啤酒', '尿布'], ['面包', '黄油', '饼干'],
        ['牛奶', '尿布', '饼干'], ['啤酒', '尿布'], ['牛奶', '尿布', '面包', '黄油'],
        ['啤酒', '饼干'], ['啤酒', '尿布', '饼干'], ['牛奶', '尿布', '面包', '黄油'],
        ['尿布', '面包', '黄油']
    ]
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit_transform(dataset), columns=te.columns_)
    freq = apriori(df, min_support=0.3, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.7)
    return freq, rules

# 任务2：图书数据
def task2_books(path='../source/MLBOOK_DATA.xlsx'):
    df = pd.read_excel(path)
    cols = ['价格描述', '出版社', '作者', '包装', '畅销程度']
    df = df[cols].fillna('未知')
    transactions = df.values.tolist()
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
    freq = fpgrowth(df, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.6)
    return freq, rules[rules['lift'] > 1]

# 示例调用
if __name__ == "__main__":
    f1, r1 = task1_basket()
    print(f1)
    print(r1)
    f2, r2 = task2_books()
    print(f2)
    print(r2)
import pandas as  pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

class Metrics :

    def __init__(self):
        pass

    def FrequentPatternMining(self, X):
        te = TransactionEncoder()
        te_ary = te.fit(X).transform(X)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        print(df)
        # print support
        print(apriori(df, min_support=0.1))
        # print frequent itemset
        frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        frequent_itemsets
        print("frequent itemset at min support = 0.6")
        print(frequent_itemsets)

    def FrequentPatternMining(self, X):
        te = TransactionEncoder()
        te_ary = te.fit(X).transform(X)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        print(df)
        # print support
        print(apriori(df, min_support=0.1))
        # print frequent itemset
        frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        frequent_itemsets
        print("frequent itemset at min support = 0.6")
        print(frequent_itemsets)
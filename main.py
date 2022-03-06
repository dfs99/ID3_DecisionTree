import pandas as pd
from ID3DecisionTree import ID3DecisionTree

if __name__ == '__main__':
    tree = ID3DecisionTree(pd.read_csv("dataset/data.csv"))
    tree.generate_tree(True)

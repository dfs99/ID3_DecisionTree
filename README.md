# ID3 Decision Tree Algorithm

The repository itself contains an implementation for the well-known 
ID3 algorithm. The implementation carried out uses a LIFO to apply
backtracking instead of recursive calls. 

###### In order to use the implementation you must consider the following advices:

**1-.) The ID3 implementation takes as parameter a pd.Dataframe 
that only works with discrete attributes.** 

**2-.) The class to be classified will be always the pd.Dataframe's last column.**

**3-.) The method 'generate_tree()' has verbose=True as default. It will print out
traces of the tree.**

###### An example is given below:

    import pandas as pd
    from ID3DecisionTree import ID3DecisionTree
    
    if __name__ == '__main__':
        tree = ID3DecisionTree(pd.read_csv("dataset/any_csv.csv"))
        tree.generate_tree()

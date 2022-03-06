import pandas as pd
import numpy as np


class ID3Node(object):
    """

        ID3Node is used to represent either a leaf or no-leaf node
        at ID3 decision tree.

        class Attributes:
            => TARGET_CLASS: contains the attribute that it's being classified.

        Attributes:
            => _instances: contains a given dataset as a pd.DataFrame
            => _attributes: a set of remaining attributes.
            => _parent: pointer to the parent node. If root, points to None.
            => _is_leaf: In order to indicate whether the node is a leaf or not. {True, False}
            => _children: A list that contains pointers to its children ID3Nodes.
            => _attr_selected: Attribute selected after info gain evaluation. It stores the best
                               attribute in terms of info gain.
            => _val_used: contains the edge's value that precedes the current node.
            => _verbose: contains as '\t' as its depth in order to print it out.

    """
    TARGET_CLASS = None

    __slots__ = ('_instances',
                 '_attributes',
                 '_parent',
                 '_is_leaf',
                 '_children',
                 '_attr_selected',
                 '_val_used',
                 '_verbose')

    def __init__(self, instances, attributes, parent=None, val_used=None):
        self._instances = instances
        self._attributes = attributes
        self._parent = parent
        self._is_leaf = False
        self._children = []
        self._attr_selected = None
        self._val_used = val_used
        self._verbose = "\t"

    def __str__(self):
        msg = f"{self._verbose}The edge it comes from has value <{self._val_used}>. ID3Node"
        msg += " is a leaf node and " if self._is_leaf else f" has selected attr<{self._attr_selected}> to split data and "
        msg += f"contains {len(self._instances)} instances on it.\n"
        if ID3Node.TARGET_CLASS is not None and self._is_leaf:
            for val in ID3Node.TARGET_CLASS._values:
                msg += f"{self._verbose}\tHas {len(self._instances.loc[self._instances[ID3Node.TARGET_CLASS._label] == val])} instances that are evaluated with value <{val}>.\n"
        return msg

    def __hash__(self):
        to_cipher = f"ID3Node:"
        for attr in self._attributes:
            to_cipher += f"attr{attr._label},{attr._values}"
        return hash(to_cipher)

    def __eq__(self, other):
        if isinstance(other, ID3Node):
            if self.__hash__() == other.__hash__():
                return True
        return False


class DiscreteAttribute(object):
    """
        DiscreteAttribute is used to represent an attribute that it's,
        in fact, discrete.

        Attributes:
            => _label: The attribute's identifier.
            => _values: The set of values the discrete attribute has.

    """
    __slots__ = ('_label', '_values')

    def __init__(self, label: str, values: set):
        self._label = label
        self._values = values

    def __str__(self):
        return f"Discrete attribute with label {self._label} has the following values: {self._values}"

    def __hash__(self):
        return hash(f"DiscreteAttribute:{self._label},with_values:{self._values}")

    def __eq__(self, other):
        if isinstance(other, DiscreteAttribute):
            if self.__hash__() == other.__hash__():
                return True
        return False

    def __copy__(self):
        return DiscreteAttribute(self._label, self._values)


class ClassToMeasure(DiscreteAttribute, object):
    """
        ClassToMeasure is used to represent an attribute that it's,
        in fact, discrete and it's being classified.

        Attributes:
            => _label: The attribute's identifier.
            => _values: The set of values the discrete attribute has.

    """
    __slots__ = ()

    def __init__(self, label, values):
        super().__init__(label, values)

    def __str__(self):
        return f"Discrete class to classify with label {self._label} has the following values: {self._values}"


class ID3DecisionTree(object):
    """
        ID3DecisionTree is used to represent an ID3 Decision Tree
        given a dataset with discrete attributes.

        Notice that the _class must always be the last column from the pd.Dataframe.
        This constraint is useful to automatize the process.

        Attributes:
             => _attributes: set of discrete attributes represented as DiscreteAttribute instances.
             => _class: a class attribute represented as ClassToMeasure instance.
             => _dataset: pd.DataFrame

    """

    @staticmethod
    def _data_handler(dataset):
        """
        Note that the last attribute from Dataframe will be chosen
        as the class attribute.
        Given a dataset, it returns a set of DiscreteAttributes and a
        ClassToMeasure instance.
        """
        attributes = set()
        for attr in list(dataset.keys())[::-1][1:]:
            attributes.add(DiscreteAttribute(attr, set(dataset[attr].unique())))
        to_classify = list(dataset.keys())[::-1][0]
        return attributes, ClassToMeasure(to_classify, dataset[to_classify].unique())

    def __init__(self, dataset):
        self._attributes, self._class = ID3DecisionTree._data_handler(dataset)
        self._dataset = dataset

    def generate_tree(self, verbose=True):
        ID3Node.TARGET_CLASS = self._class
        root = ID3Node(self._dataset.copy(), self._attributes.copy())
        self._id3_generator(root)
        if verbose:
            lifo = [root]
            while len(lifo) > 0:
                current_node = lifo.pop(0)
                print(f"{current_node}")
                for c in current_node._children:
                    lifo.insert(0, c)

    def _info_gain(self, attr: DiscreteAttribute, dataset: pd.DataFrame):
        info_gain = 0
        for val in attr._values:
            entropy_for_val = 0
            for class_val in self._class._values:
                if len(dataset.loc[dataset[attr._label] == val]) == 0:
                    continue
                probability = (len(
                    dataset.loc[(dataset[attr._label] == val) & (dataset[self._class._label] == class_val)]) / len(
                    dataset.loc[dataset[attr._label] == val]))
                if probability != 0:
                    entropy_for_val -= probability * np.log2(probability)
            info_gain += (len(dataset.loc[dataset[attr._label] == val]) / len(dataset)) * entropy_for_val
        return info_gain

    def info_gain(self, root):
        best_attr = None
        best_info_gain = 1.1  # important, must be a little bit more than 1.
        for attr in root._attributes:
            current_info_gain = self._info_gain(attr, root._instances)
            #print(f"ganancia informacion para attr {attr} => {current_info_gain}")
            if current_info_gain < best_info_gain:
                best_info_gain = current_info_gain
                best_attr = attr
        #print()
        return best_attr

    def _id3_generator(self, node):
        lifo_stack = [node]
        while len(lifo_stack) > 0:
            current_node = lifo_stack.pop(0)
            if current_node._is_leaf:
                continue
            best_attr = self.info_gain(current_node)
            current_node._attr_selected = best_attr._label
            evaluated_attr = set()
            evaluated_attr.add(best_attr)
            for val in best_attr._values:
                instances_subset = current_node._instances.loc[current_node._instances[best_attr._label] == val]
                attrs_subset = current_node._attributes - evaluated_attr
                is_leaf = False
                for class_val in self._class._values:
                    if len(instances_subset.loc[instances_subset[self._class._label] == class_val]) == len(
                            instances_subset):
                        is_leaf = True
                child = ID3Node(instances_subset, attrs_subset, current_node, val)
                child._verbose += current_node._verbose
                if is_leaf:
                    child._is_leaf = True
                lifo_stack.insert(0, child)
                current_node._children.append(child)

import pandas as pd
import numpy as np
import sys


class ID3Node(object):
    __slots__ = ('_instances', '_attributes', '_parent', '_is_leaf', '_children')

    def __init__(self, instances, attributes, parent=None):
        #print(type(instances))
        self._instances = instances
        self._attributes = attributes
        self._parent = parent
        self._is_leaf = False
        self._children = []

    def __str__(self):
        msg = f"ID3Node contains {len(self._instances)} instances and has the following attributes:\n"
        for attr in self._attributes:
            msg += attr.__str__() + "\n"
        msg += f"Furthermore is_leave is evaluated as<{self._is_leaf}>."
        return msg

    def __hash__(self):
        #todo: revisar, no me convence...
        to_cipher = "ID3Node:"
        for attr in self._attributes:
            to_cipher += f"attr{attr._label},{attr._values}"
        return hash(to_cipher)

    def __eq__(self, other):
        if isinstance(other, ID3Node):
            if self.__hash__() == other.__hash__():
                return True
        return False


class DiscreteAttribute(object):
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
    __slots__ = ()
    def __init__(self, label, values):
        super().__init__(label, values)

    def __str__(self):
        return f"Discrete class to classify with label {self._label} has the following values: {self._values}"


class ID3(object):

    def __init__(self, dataset):
        self._attributes, self._class = ID3._data_handler(dataset)
        self._dataset = dataset
        # print(self._attributes, self._class)

    def generate_tree(self):
        root = ID3Node(self._dataset.copy(), self._attributes.copy())
        self._id3_generator(root)
        # print all the tree.
        lifo = [root]
        while len(lifo) > 0:
            current_node = lifo.pop(0)
            for c in current_node._children:
                lifo.insert(0, c)
            print('*'*15)
            print(current_node)
            print('*' * 15)


    @staticmethod
    def _data_handler(dataset):
        """
        Asumir que el último atributo es la clase que queremos clasificar.

        :param dataset:
        :return:
        """
        attributes = set()
        for attr in list(dataset.keys())[::-1][1:]:
            attributes.add(DiscreteAttribute(attr, set(dataset[attr].unique())))
        to_classify = list(dataset.keys())[::-1][0]
        return attributes, ClassToMeasure(to_classify, dataset[to_classify].unique())

    def _info_gain(self, attr: DiscreteAttribute, dataset: pd.DataFrame):
        info_gain = 0
        for val in attr._values:
            entropy_for_val = 0
            for class_val in self._class._values:
                #print('+'*25)
                #print(dataset)
                #print('+' * 25)
                if len(dataset.loc[dataset[attr._label] == val]) == 0:
                    continue
                probability = (len(dataset.loc[(dataset[attr._label] == val) & (dataset[self._class._label] == class_val)]) / len(dataset.loc[dataset[attr._label] == val]))
                if probability != 0:
                    entropy_for_val -= probability * np.log2(probability)
            info_gain += (len(dataset.loc[dataset[attr._label] == val]) / len(dataset)) * entropy_for_val
        #print(f"Info Gain  for {attr} => {info_gain}")
        return info_gain

    def info_gain2(self):
        best_attr = None
        best_info_gain = 1.1    # un poco más de 1 por si acaso
        for attr in self._attributes:
            current_info_gain = self._info_gain(attr, self._dataset)
            if current_info_gain < best_info_gain:
                best_info_gain = current_info_gain
                best_attr = attr
        return best_attr


    def info_gain(self, root):
        #print("*"*25)
        best_attr = None
        best_info_gain = 1.1  # un poco más de 1 por si acaso
        for attr in root._attributes:
            current_info_gain = self._info_gain(attr, root._instances)
            if current_info_gain < best_info_gain:
                best_info_gain = current_info_gain
                best_attr = attr
            #print(attr)
        #print("*" * 25)
        return best_attr

    def _id3_generator(self, node):
        lifo_stack = [node]
        while len(lifo_stack) > 0:
            current_node = lifo_stack.pop(0)
            if current_node._is_leaf:
                continue
            best_attr = self.info_gain(current_node)
            best_attr = best_attr.__copy__()
            evaluated_attr = set()
            evaluated_attr.add(best_attr)
            for val in best_attr._values:
                instances_subset = current_node._instances.loc[current_node._instances[best_attr._label] == val]
                attrs_subset = current_node._attributes - evaluated_attr
                is_leaf = False
                for class_val in self._class._values:
                    if len(instances_subset.loc[instances_subset[self._class._label] == class_val]) == len(instances_subset):
                        is_leaf = True
                child = ID3Node(instances_subset, attrs_subset)
                child._parent = current_node
                if is_leaf:
                    child._is_leaf = True
                lifo_stack.insert(0, child)
                current_node._children.append(child)



    def _id3_generator2(self, node):
        print("*"*25)
        print("Node info")
        for x in node._attributes:
            print(f"tiene el attr {x}")
        print("tiene un total de instances", len(node._instances))
        print(f"es nodo hoja? {node._is_leaf}")
        print("*" * 25)
        if node._is_leaf:
            return None
        else:
            best_attr = self.info_gain(node)
            print(f"Atributo seleccionado {best_attr._label}")
            best_attr = best_attr.__copy__()
            evaluated_attr = set()
            evaluated_attr.add(best_attr)
            for val in best_attr._values:
                print(f"\t Para el valor: {val}")
                instances_subset = node._instances.loc[node._instances[best_attr._label] == val]
                attrs_subset = node._attributes - evaluated_attr
                is_leaf = False
                for class_val in self._class._values:
                    if len(instances_subset.loc[instances_subset[self._class._label] == class_val]) == len(instances_subset):
                        is_leaf = True
                if is_leaf:
                    #print("NOOODO HOJAAAAAAAAAAAAA")
                    leaf = ID3Node(instances_subset, attrs_subset)
                    leaf._is_leaf = True
                    leaf._parent = node
                    return self._id3_generator(leaf)
                else:
                    no_leaf = ID3Node(instances_subset, attrs_subset)
                    no_leaf._parent = node
                    return self._id3_generator(no_leaf)


data = pd.read_csv("dataset/data2.csv")
# print(data.head())
# print(data['Edad'].unique())
# x = data.loc[data['Edad'] == 'Joven']
# print(x)
# print(data)
# print(list(data.keys()))
# print(type({1,2,3,5}))

arbol = ID3(data)
#for a in arbol._attributes:
    #print(a)
#    arbol._info_gain(a, arbol._dataset)
arbol.generate_tree()
#print(f"Best attribute => {arbol.info_gain2()}")
""" 
    
    seleccionar el attr actual con IG. => generando el nodo raiz.
    for valor in atributo:
        generar sucesores que contengan aquellas instancias con ese valor en atributo.
        si todas instancias son de una misma clase, nodo hoja 
        else:
            llamada recursiva (instancias nodo sin ese atributo)
    

"""

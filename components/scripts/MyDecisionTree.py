import math
import numpy as np # type: ignore
import pandas as pd # type: ignore

class Node:
    def __init__(self, attribute, parent_attr, parent_attr_value, names):
        self.names = names
        self.attribute = attribute
        self.lower_child = None
        self.upper_child = None
        self.parent_attr = parent_attr
        self.parent_attr_value = parent_attr_value

    def add_child(self, child, lower):
        if lower:
            self.lower_child = child
        else:
            self.upper_child = child

    def show(self, file=None, _prefix="", _last=True):
        print(_prefix, "`- " if _last else "|- ", "" if self.parent_attr is None else "if " + self.names[self.parent_attr].lower() + " is " + self.parent_attr_value,  " then " + self.attribute if self.lower_child == None else "", sep="", file=file)
        _prefix += "   " if _last else "|  "

        if self.upper_child is not None:
            self.upper_child.show(file, _prefix, False)
        if self.lower_child is not None:
            self.lower_child.show(file, _prefix, True)

class MyDecisionTree:
    def __init__(self):
        pass
    
    def fit(self, training, y_train):
        # move category to end column
        training["category"] = y_train
        self.names = {key:value for key, value in enumerate(training.columns)}
        self.training = training.values.tolist()
        self.tree = self.train_DT(self.training, "Root", None, None)
        
    def entropy(self, group):
        if len(group) == 0:
            return 0
        tally = {}
        for entry in group:
            if entry[-1] not in tally.keys():
                tally[entry[-1]] = 1
            else:
                tally[entry[-1]] += 1
        props = {key:value/len(group) for key, value in tally.items()}
        entr = 0
        for key, value in props.items():
            entr += -value*math.log2(value)
        return entr
    
    def show(self):
        self.tree.show()
        
    def cant_split_further(self, rows):
        for entry in rows:
            for i in range(len(entry) - 1):
                if entry[i] != rows[0][i]:
                    return False
        return True

    def get_max_class(self, rows):
        classes = [row[-1] for row in rows]
        tally = {}
        for cla in classes:
            if cla not in tally.keys():
                tally[cla] = 1
            else:
                tally[cla] += 1
        return max(tally, key=tally.get)

    def find_optimal_split(self, rows):
        entr = math.inf
        split = None
        split_groups = None
        for i in range(len(rows[0])-1):
            # Split on attribute i
            values = sorted(set([row[i] for row in rows]))
            # values = sorted(values)

            for val in values:
                # Split in two groups on val
                # lower is inclusive
                upper = []
                lower = []
                for row in rows:
                    if row[i]>val:
                        upper.append(row)
                    else:
                        lower.append(row)
                # calculate entropy
                test_entr = self.entropy(upper) + self.entropy(lower)
                # keep best split
                if test_entr < entr:
                    entr = test_entr
                    split = (i, val)
                    split_groups = {">"+str(val):upper, "≤"+str(val):lower}
        # i, val = split
        # values = sorted(set([row[i] for row in rows]))
        # for index, value in enumerate(values):
        #     if index == len(values) - 1:
        #         print("i dont think this is possible")
        #         print(split)
        #         break
        #     elif value == val:
        #         new_val = np.mean([val, values[index+1]])
        #         split = (i, new_val)
        #         break
        return split, split_groups, entr


    def train_DT(self, rows, parent_majority, parent_split, parent_attr_value):
        # if there are no examples with the attribute value of the branch.
        if len(rows) == 0:
            return Node(parent_majority, parent_split, parent_attr_value, self.names)
        entr = self.entropy(rows)
        # if entropy is zero (all the same class)
        if entr == 0:
            class_ = rows[0][-1]
            return Node(class_, parent_split, parent_attr_value, self.names)
        # if can't split any further
        elif self.cant_split_further(rows):
            class_ = self.get_max_class(rows)
            return Node(class_, parent_split, parent_attr_value, self.names)
        else:
            majority = self.get_max_class(rows)
            split, split_groups, new_entr = self.find_optimal_split(rows)
            # return the majority class if entropy doesn't improve
            if round(new_entr, 6) >= round(entr, 6):
                return Node(majority, parent_split, parent_attr_value, self.names)
            node = Node(majority, parent_split, parent_attr_value, self.names)
            for key, group in split_groups.items():
                lower=False if key[0] == ">" else True
                node.add_child(self.train_DT(group, majority, split[0], key), lower)
            return node

    def predict_value(self, tree, test_row):
        if tree.lower_child is None or tree.upper_child is None:
            return tree.attribute
        else:
            lower = tree.lower_child
            upper = tree.upper_child
            if float(test_row[upper.parent_attr]) > float(upper.parent_attr_value[1::]):
                return self.predict_value(upper, test_row)
            else:
                return self.predict_value(lower, test_row)
            
    def predict(self, test):
        test = test.values.tolist() 
        predict_class = [self.predict_value(self.tree, row) for row in test]        
        return predict_class



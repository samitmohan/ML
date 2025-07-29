# https://www.deep-ml.com/problems/20

'''
Write a Python function that implements the decision tree learning algorithm for classification. 
The function should use recursive binary splitting based on entropy and information gain to build a decision tree. 
It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as input, 
and return a nested dictionary representing the decision tree.

Output: Best match for given data to split:
{
            'Outlook': {
                'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}},
                'Overcast': 'Yes',
                'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}
            }
        }
'''

'''
Entropy is inversely proportional to information gain
low entropy -> high info gain,
high entropy -> low info gain.
giny impurity for leaf node = 1 - (prob(yes)^2) - (prob(no)^2)
lower gini impurity -> lowest impurity -> highest info gain -> root val pick
e = - sum(p(x)) * log(p(x)) {p(x) : number of times this class has occurred / total nodes}
stopping criteria: max depth, min num of samples, min impurity decrease (iteration decide when to stop dividing the node)

STEPS:

    Select Attribute: Choose the attribute with the highest information gain.
    Split Dataset: Divide the dataset based on the values of the selected attribute.
    Recursion: Repeat the process for each subset until:
        All data is perfectly classified, or
        No remaining attributes can be used to make a split.

'''

from collections import Counter
import math

def calc_entropy(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def calc_info_gain(examples, attr, target_attr):
    total_labels = [ex[target_attr] for ex in examples] # no no yes yes
    total_ent = calc_entropy(total_labels)
    n = len(examples)
    rem = 0.0
    for v in set(ex[attr] for ex in examples):
        subset_labels = [ex[target_attr] for ex in examples if ex[attr] == v]
        rem += (len(subset_labels) / n) * calc_entropy(subset_labels)
    return total_ent - rem


# from scratch
def get_majority_class(examples, target_attr):
    """
    Finds the most common class label in a list of examples.
    """
    labels = [ex[target_attr] for ex in examples]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str, parent_examples: list[dict] = None) -> dict:
    # Base Case 1: If no examples are left, return the majority class of the parent.
    # This handles cases where a split results in an empty subset.
    if not examples:
        return get_majority_class(parent_examples, target_attr)

    # Base Case 2: If all remaining examples have the same class, return that class.
    first_label = examples[0][target_attr]
    if all(ex[target_attr] == first_label for ex in examples):
        return first_label

    # Base Case 3: If there are no more attributes to split on, return the majority class.
    if not attributes:
        return get_majority_class(examples, target_attr)

    # Find the best attribute to split on by calculating information gain
    gains = {attr: calc_info_gain(examples, attr, target_attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    # Build the tree recursively
    tree = {best_attr: {}}
    remaining_attributes = [attr for attr in attributes if attr != best_attr]

    # For each unique value of the best attribute, create a new branch
    for value in set(ex[best_attr] for ex in examples):
        subset = [ex for ex in examples if ex[best_attr] == value]
        # Recursive call to build the subtree
        subtree = learn_decision_tree(subset, remaining_attributes, target_attr, examples)
        tree[best_attr][value] = subtree
    return tree

def main():
    # The full "Play Tennis" dataset
    examples = [
        {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
        {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
        {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'PlayTennis': 'No'},
        {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong', 'PlayTennis': 'Yes'},]
    attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    print(learn_decision_tree(examples, attributes, 'PlayTennis'))

main()
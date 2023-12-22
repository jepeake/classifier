# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Load Clean & Noisy Datasets
clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_data = np.loadtxt(('wifi_db/noisy_dataset.txt'))  

class Node:
    def __init__(self, attribute, value, left, right, label, is_leaf, depth):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.label = label
        self.is_leaf = is_leaf
        self.depth = depth
        self.parent = None
        self.X = None
        self.Y = depth
        self.mod = 0
        self.is_leftmost = True
    
    def calculate_final_x(self, mod_sum):
            """
            Calculates the final x positions of the nodes in the tree by shifting them by the calculated amount.

            Args:
                mod_sum (int): The amount to shift the nodes by.

            Returns:
                None
            """
            self.X += mod_sum
            mod_sum += self.mod
            if self.left:
                self.left.calculate_final_x(mod_sum)
            if self.right:
                self.right.calculate_final_x(mod_sum)

    def calculate_init_x(self, node_size, sibling_distance):
            """
            Calculates the initial x-coordinate for each node in the decision tree.

            The x-coordinate is determined using a post-order traversal approach, where the position of a node is calculated based on its children's positions.

            If the node is a leaf, its x-coordinate is set to the left sibling's x-coordinate plus the node size and sibling distance.

            If the node has two children, its x-coordinate is set to the midpoint between its children's x-coordinates.

            If the node is on the right, it is shifted to the right to avoid overlapping with its left sibling.

            Parameters:
            - node_size: The space that each node takes horizontally; used to determine the distance between nodes.
            - sibling_distance: The additional distance between nodes that are siblings, providing space between different branches.

            Return:
            - None

            """
            # Post Order Traversal
            if self.left:
                self.left.calculate_init_x(node_size, sibling_distance)
            if self.right:
                self.right.calculate_init_x(node_size, sibling_distance)
            
            # If a Leaf
            if self.is_leaf:
                # If not Left Node, set X to Left Sibling + Spacing
                if not self.is_leftmost:
                    self.X = self.parent.left.X + node_size + sibling_distance
                else:
                    self.X = 0
            # Has 2 Children, Position Between Children
            else:
                midpoint = (self.left.X + self.right.X) / 2
                if self.is_leftmost:
                    self.X = midpoint
                # If Node on Right, may need to Shift it & its Children
                else:
                    self.X = self.parent.left.X + node_size + sibling_distance
                    self.mod = self.X - midpoint
            
            if not self.is_leaf and (not self.is_leftmost):
                # If Node has Children, Check if Subtrees Overlap & Shift Accordingly
                self.check_conflicts(node_size, sibling_distance)
    
    def check_conflicts(self, node_size, sibling_distance):
        """
        Checks for conflicts between nodes and calculates the distance needed to shift the entire subtree.

        Parameters:
            node_size (int): The size of the node.
            sibling_distance (int): The distance between siblings.

        Returns:
            None
        """
        min_distance = node_size + sibling_distance # Min Distance Between Nodes
        shift_val = 0
        node_contour = {}
        self.get_left_contour(0, node_contour)
        
        # Only for Right Nodes, Gather Right Contour of Left Sibling & Calculate Distance needed to Shift Entire Subtree
        if not self.is_leftmost:
            left_sibling = self.parent.left
            sib_right_contour = {}
            left_sibling.get_right_contour(0, sib_right_contour)

            for level in range(self.Y + 1, min(max(node_contour.keys()), max(sib_right_contour.keys()))):
                distance = node_contour[level] - sib_right_contour[level]
                if (distance + shift_val < min_distance):
                    shift_val = min_distance - distance
            if shift_val > 0:
                self.X += shift_val + 2
                self.mod += shift_val + 1
                shift_val = 0
    
    def get_left_contour(self, mod_sum, contour_dict):
            """
            Recursively computes the left contour of the subtree rooted at this node.
            The left contour is defined as the minimum x-coordinate of all nodes at a given y-coordinate.
            The results are stored in the contour_dict dictionary.

            Parameters:
                mod_sum (int): the sum of all mod values from the root to this node.
                contour_dict (dict): a dictionary to store the left contour values for each y-coordinate.

            Returns:
                None
            """
            if self.Y not in contour_dict:
                contour_dict[self.Y] = self.X + mod_sum
            else:
                contour_dict[self.Y] = min(contour_dict[self.Y], self.X + mod_sum)

            mod_sum += self.mod
            if self.left:
                self.left.get_left_contour(mod_sum, contour_dict)
            if self.right:
                self.right.get_left_contour(mod_sum, contour_dict)
    
    def get_right_contour(self, mod_sum, contour_dict):
            """
            Recursively traverses the right subtree of the current node and updates the contour dictionary with the maximum
            x-coordinate of each y-coordinate encountered

            Parameters:
                mod_sum (int): the sum of the modifier values of all the ancestors of the current node
                contour_dict (dict): a dictionary that maps y-coordinates to the maximum x-coordinate encountered at that
                y-coordinate

            Returns:
                None
            """
            if self.Y not in contour_dict:
                contour_dict[self.Y] = self.X + mod_sum
            else:
                contour_dict[self.Y] = max(contour_dict[self.Y], self.X + mod_sum)

            mod_sum += self.mod
            if self.left:
                self.left.get_right_contour(mod_sum, contour_dict)
            if self.right:
                self.right.get_right_contour(mod_sum, contour_dict)
    
    def plot_tree(self, tree_height):
            """
            Plots Decision Tree with this Node as its Root

            Parameters:
                tree_height (int): the height of the tree to be plotted

            Returns:
                None
            """
            output_str = None
            if self.is_leaf:
                output_str = f"Leaf: {self.label}"
            else:
                output_str = f"x{self.attribute} > {self.value}"
            for child in [self.left, self.right]:
                if child:
                    plt.plot([self.X, child.X], [tree_height - self.Y, tree_height - child.Y], 'k-')
                    child.plot_tree(tree_height)
            plt.text(self.X, tree_height - self.Y, output_str, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', pad=0.2))

def decision_tree_learning(training_dataset, depth=0):
    """
    Recursive Function to Create the Decision Tree

    Parameters:
    - training_dataset (array): a NumPy array where the last column holds the label
    - depth (int): the current depth of the tree

    Returns:
    - tuple: a tuple containing the node & the depth of the tree at that node
    """
    # Get the Unique Labels from the Last Column
    unique_labels = np.unique(training_dataset[:, -1])
    # Base Case: if only 1 Unique Label, Return a Leaf Node
    if len(unique_labels) == 1:
        return (Node(attribute=None, value=None, left=None, right=None, label=unique_labels[0], is_leaf=True, depth=depth), depth)
    
    # Case 2: More than 1 Label remaining -> Continue Splitting/Learning
    attribute, value, left, right = find_split(training_dataset)
    left_node, left_depth = decision_tree_learning(left, depth + 1)
    right_node, right_depth = decision_tree_learning(right, depth + 1)
    node = Node(attribute, value, left_node, right_node, None, False, depth)
    left_node.parent = node
    right_node.parent = node
    right_node.is_leftmost = False
    return (node, max(left_depth, right_depth))

def find_split(dataset):
    """
    Find the Best Attribute to Split Dataset by Maximizing the Information Gain

    Parameters:
    - dataset (array): a NumPy array where the last column holds the label

    Returns:
    - tuple: a tuple containing the index of the best attribute to split on, the best split value, & the left & right datasets after the split
    """
    # Initialise Values
    max_information_gain = 0
    best_split = None
    best_left_dataset = None
    best_right_dataset = None

    # For Each Attribute
    for attribute in range(dataset.shape[1] - 1): 
        # Sort the Values of the Attributes
        sorted_indices = np.argsort(dataset[:, attribute]) # Selects all Values in Column Specified by Attributes & Returns Sorted Indices
        sorted_data = dataset[sorted_indices] # Creates Sorted Dataset based on Indices Returned by Argsort
        # For Each of the Sorted Attributes
        for i in range(1, len(sorted_data)):
            if sorted_data[i, attribute] != sorted_data[i-1, attribute]: # If the 2 Samples are Not Equal
                # Calculate the Midpoint Between Each Pair of Samples
                mid = (sorted_data[i, attribute] + sorted_data[i-1, attribute]) / 2.0
                
                # Split the Dataset on this Midpoint
                left_subset = sorted_data[sorted_data[:, attribute] <= mid]
                right_subset = sorted_data[sorted_data[:, attribute] > mid]
                
                # Calculate the Information Gain based on this Split
                gain = information_gain(dataset, left_subset, right_subset)
                if gain > max_information_gain:
                    max_information_gain = gain
                    best_split = (attribute, mid)
                    best_left_dataset = left_subset
                    best_right_dataset = right_subset

    # Returns the Best Split based upon Max Information Gain (after calculating for each pair of samples)
    return best_split[0], best_split[1], best_left_dataset, best_right_dataset

def entropy_set(dataset):
    """
    Calculate the Entropy of a Dataset

    Parameters:
    - dataset (array): a NumPy array where the last column holds the label

    Returns:
    - float: the calculated entropy of the dataset
    """
    total_samples = len(dataset)
    if total_samples == 0:
        return 0 # Return Entropy of 0 if the Dataset is Empty

    # Count Number of Unique Labels in the Dataset
    _, count = np.unique(dataset[:, -1], return_counts=True) 
    probabilities = count / total_samples

    # Entropy Calculation
    H = -np.sum(probabilities * np.log2(probabilities)) 
    return H


def entropy_subsets(left_dataset, right_dataset):
    """
    Calculate the Weighted Average Entropy of the Subsets

    Parameters:
    - left_dataset (array): the left subset of the dataset
    - right_dataset (array): the right subset of the dataset

    Returns: 
    - float: the weighted average entropy of the subsets
    """
    total_samples = len(left_dataset) + len(right_dataset)
     # Calculate the Weighted Entropy for Left & Right Subsets
    return (len(left_dataset) / total_samples * entropy_set(left_dataset) + 
            len(right_dataset) / total_samples * entropy_set(right_dataset))

def information_gain(parent_dataset, left_child, right_child):
    """
    Calculate the Information Gain from Splitting a Dataset into two subsets.

    Parameters:
    - parent_dataset (array): the dataset before splitting
    - left_child (array): the left subset after splitting
    - right_child (array): the right subset after splitting

    Returns:
    - float: the information gain from the split
    """
    # Information Gain = Entropy of Parent Dataset - Weighted Average Entropy of Subsets
    return entropy_set(parent_dataset) - entropy_subsets(left_child, right_child)

def predict_sample(tree, sample):
    """
    Predict a Samples Label using the Trained Tree

    Parameters:
    - tree (dict): the decision tree
    - sample: sample to predict

    Returns:
    - predicted label for the sample
    """
    # Base Case: if Current Node is a Leaf - Return its Label
    if tree.is_leaf:
        return tree.label
    # Recurse Down the Tree to The Next Child Until a Leaf Node is Hit
    if sample[tree.attribute] > tree.value:
        return predict_sample(tree.right, sample)
    else:
        return predict_sample(tree.left, sample)

def evaluate(test_db, trained_tree):
    """
    Evaluate the Accuracy of a Trained Tree using the Test Dataset

    Parameters:
    - test_db (array): the test dataset
    - trained_Tree (dict): the trained decision tree

    Returns:
    - confusion matrix for the test set prediction
    """
    #Initialise Variables
    total_samples = test_db.shape[0]
    predictions = []
    gold = test_db[:, -1]
    classes = np.unique(gold)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int32)

    # For Each Sample in Test Dataset, Predict the Labels
    for i in range(total_samples):
        prediction = predict_sample(trained_tree, test_db[i])
        predictions.append(prediction)

    # Create Confusion Matrix based on Predictions
    for i in range(len(gold)):
        row = np.where(classes == gold[i])
        col = np.where(classes == predictions[i])
        confusion[row, col] += 1

    return confusion

def evaluation_metrics(confusion_matrix):
    """
    Calculate Evaluation Metrics from Confusion Matrix

    Parameters:
    - confusion_matrix (array): confusion matrix from classification predictions

    Returns:
    - tuple containing accuracy, precision, recall, and F1 score
    """
    # Accuracy
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0

    # Precision, Recall, & F1 for each class
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0, where=(np.sum(confusion_matrix, axis=0) > 0))
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1, where=(np.sum(confusion_matrix, axis=1) > 0))
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # Case where Precision + Recall = 0
    f1_scores = np.nan_to_num(f1_scores)

    return accuracy, precision, recall, f1_scores

def k_fold_cross_validation(dataset, k=10):
    """
    Run K-Fold Cross Validation on a Dataset

    Parameters:
    - dataset (array): the dataset to perform cross-validation on
    - k (int): the number of folds

    Returns: 
    - mean accuracy, precision, recall, F1 score across all folds
    """
    confusion_matrices = []

    #Shuffle Dataset and Find Fold Size
    np.random.shuffle(dataset)
    fold_size = len(dataset) // k

    # Evaluate for Each Fold
    for i in range(k):
        start_index = i * fold_size
        end_index = start_index + fold_size if i != (k-1) else len(dataset)
        test_db = dataset[start_index:end_index]
        training_db = np.vstack((dataset[:start_index], dataset[end_index:]))
        
        tree, _ = decision_tree_learning(training_db)

        # Store Confusion Matrix Created from Each Fold
        confusion_matrices.append(evaluate(test_db, tree))

    # Final Mean Confusion Matrix
    mean_matrix = np.mean(confusion_matrices, axis=0)
    print(mean_matrix)

    # Calculate Mean Evaluation Metrics
    metrics = evaluation_metrics(mean_matrix)
    
    return metrics       
        
if __name__ == "__main__":
    print("-------------------")
    print("Confusion Matrices:")
    print("-------------------")
    print('Clean Dataset:')
    clean_mean_accuracy, clean_mean_precision, clean_mean_recall, clean_mean_f1 = k_fold_cross_validation(clean_data)
    print("-------------------")
    print('Noisy Dataset:')
    noisy_mean_accuracy, noisy_mean_precision, noisy_mean_recall, noisy_mean_f1 = k_fold_cross_validation(noisy_data)
    print("-------------------")

    # Print Evaluation Results Results to 4dp
    print("Evaluation Metrics:")
    print("-------------------")
    print(f"Clean Dataset: Mean Accuracy = {clean_mean_accuracy:.4f}, Mean Precision = {[round(p, 4) for p in clean_mean_precision]}, Mean Recall = {[round(r, 4) for r in clean_mean_recall]}, Mean F1 = {[round(f, 4) for f in clean_mean_f1]}")
    print("-------------------")
    print(f"Noisy Dataset: Mean Accuracy = {noisy_mean_accuracy:.4f}, Mean Precision = {[round(p, 4) for p in noisy_mean_precision]}, Mean Recall = {[round(r, 4) for r in noisy_mean_recall]}, Mean F1 = {[round(f, 4) for f in noisy_mean_f1]}")
    print("-------------------")  
    
    # Plot Created Tree
    root_node, max_depth = decision_tree_learning(clean_data, 0)
    node_size = 1
    sibling_distance = 1
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(12,6))
    root_node.calculate_init_x(node_size, sibling_distance)
    root_node.calculate_final_x(0)
    root_node.plot_tree(max_depth)

    plt.axis('off')
    plt.show()
import numpy as np

#%%
from torch_geometric.datasets import ZINC
dataset = ZINC(root = 'data', split='train')

#%%
edge_features = np.empty((2, 0))
edge_labels = np.empty((1,0))

ngraphs = 10000

for idx in range(ngraphs):
    
    graph = dataset[idx]
    
    features = np.empty((2, len(graph.edge_attr)))
    
    for edge_idx, edge  in enumerate(graph.edge_index.transpose(0, 1)):
        features[:, edge_idx] = [graph.x[edge[0]].item(), graph.x[edge[1]].item()] 
        
    edge_features = np.append(edge_features, features, axis = 1)
    edge_labels = np.append(edge_labels, graph.edge_attr)
        
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

classifier = DecisionTreeClassifier()
x_train, x_test, y_train, y_test = train_test_split(edge_features.T, edge_labels)

classifier.fit(x_train, y_train)
print("Accuracy:", classifier.score(x_test, y_test))

#%%
import pickle

with open('C:/Users/mathw/OneDrive/Desktop/College/Classes/Spring 24/COMP 459/COMP459GraphGenerator/edge_labeler.pkl','wb') as f:
    pickle.dump(classifier,f)
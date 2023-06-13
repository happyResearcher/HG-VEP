import numpy as np
import os

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import HGTConv, Linear

torch.manual_seed(42)
from early_stop_v1 import EarlyStopping
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


#%% Build model #  https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['Vulnerability'])


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    train_mask = data['Vulnerability'].train_mask
    test_mask = data['Vulnerability'].test_mask
    train_loss = F.cross_entropy(out[train_mask], torch.eye(2)[data['Vulnerability'].y[train_mask].long()]) #
    test_loss = F.cross_entropy(out[test_mask], torch.eye(2)[data['Vulnerability'].y[test_mask].long()])
    train_loss.backward()
    optimizer.step()
    return float(train_loss),float(test_loss)

@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
    accs = []
    for split in ['train_mask', 'test_mask']:
        mask = data['Vulnerability'][split]
        acc = (pred[mask] == data['Vulnerability'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs

#%%Load the heterogeneous graph data
cur_dir = os.getcwd()
Features = ["graph_all","graph_affect","graph_affect_example","graph_example"]
for Feature in Features:
    if Feature=="graph_all":
        # Load the heterogeneous graph data
        data = torch.load(cur_dir+'/data/Vulnerability_hetero_graph_data_destination_balanced.pt')
        print(data)
    elif  Feature=="graph_affect":
        data = torch.load(cur_dir+'/data/Vulnerability_hetero_graph_data_destination_balanced.pt')
        del data['Vendor']
        del data['Vendor', 'HAS_PRODUCT', 'Product']

        del data['Weakness']
        del data['Weakness', 'HAS_EXAMPLE', 'Vulnerability']
        print(data)
    elif Feature=="graph_affect_example":
        data = torch.load(cur_dir+'/data/Vulnerability_hetero_graph_data_destination_balanced.pt')
        del data['Vendor']
        del data['Vendor', 'HAS_PRODUCT', 'Product']
        print(data)

    elif  Feature=="graph_example":
        data = torch.load(cur_dir+'/data/Vulnerability_hetero_graph_data_destination_balanced.pt')
        del data['Vendor']
        del data['Vendor', 'HAS_PRODUCT', 'Product']


        del data['Product']
        del data['Product', 'AFFECTED_BY', 'Vulnerability']
        print(data)

    data = T.ToUndirected()(data)
    model = HGT(hidden_channels=128, out_channels=2, num_heads=2, num_layers=2)
    modelname=["HGT"]

    device = torch.device('cpu')
    data, model = data.to(device), model.to(device)

    #initialize the model by calling it once
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)


    #%%
    best_model_path = os.path.join(cur_dir + '/data/', 'early_stop_model')
    os.makedirs(best_model_path, exist_ok=True)
    best_model_path = os.path.join(best_model_path,modelname[0]+Feature+'_best.pt')
    metric='Acc'
    early_stopping = EarlyStopping(save_path=best_model_path, verbose=(True), patience=10, delta=0.000001, metric=metric)

    train_acc_list=[]
    validation_acc_list=[]
    for epoch in np.arange(150):
        train_loss,validation_loss = train()
        accs = test()
        print('Test:', accs)
        train_acc_list.append(accs[0])
        validation_acc_list.append(accs[1])
        print('epoch,', epoch, "; train loss:",train_loss,"; validation loss:",validation_loss)
        # early_stopping(validation_loss, model)
        early_stopping(accs[1], model)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch)
            break

    early_stopping.draw_trend(train_acc_list, validation_acc_list)

    train_list = train_acc_list
    test_list = validation_acc_list

    plt.plot(range(1, len(train_list) + 1), train_list, label='Training ' + metric)
    plt.plot(range(1, len(test_list) + 1), test_list, label='Validation ' + metric)

    # find position of check point,-1 means this a minimize problem like loss or cost
    if early_stopping.sign == -1:
        checkpoint = test_list.index(min(test_list)) + 1
    else:
        checkpoint = test_list.index(max(test_list)) + 1

    plt.axvline(checkpoint, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.ylim(min(train_list+test_list), max(train_list+test_list))  # consistent scale
    plt.xlim(0, len(test_list) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cur_dir+'/data/Traning_'+modelname[0]+Feature, dpi=600)
    plt.show()

    model_extract = HGT(hidden_channels=128, out_channels=2, num_heads=2, num_layers=2)
    model_extract.load_state_dict(torch.load(best_model_path))

    #%% save result
    classifier_name = modelname[0]

    parameters = " "

    result_dir=cur_dir+'/data/ML_classification_performance.csv'
    plot_dir=result_dir.replace('.csv','_'+Feature+'ROC.npy') # prepare data for ROC curve

    # evaluate classifier on the test set
    test_mask = data['Vulnerability'].test_mask

    predictions = model_extract(data.x_dict, data.edge_index_dict).argmax(dim=-1)[test_mask]
    y_test = data["Vulnerability"].y[test_mask]

    test_confusion_matrix = confusion_matrix(y_test, predictions)

    report = classification_report(y_test, predictions, labels=[0, 1], target_names=['class 0', 'class 1'],
                                   output_dict=True, zero_division=0)  # output_dict=True
    test_acc = report['accuracy']
    test_pre = report['macro avg']['precision']
    test_rec = report['macro avg']['recall']
    test_f1 = report['macro avg']['f1-score']

    test_class1_pre = report['class 1']['precision']
    test_class1_rec = report['class 1']['recall']
    test_class1_f1 = report['class 1']['f1-score']

    test_class0_pre = report['class 0']['precision']
    test_class0_rec = report['class 0']['recall']
    test_class0_f1 = report['class 0']['f1-score']

    print([classifier_name, parameters])
    print([test_acc, test_pre, test_rec, test_f1,
           test_class1_pre, test_class1_rec, test_class1_f1,
           test_class0_pre, test_class0_rec, test_class0_f1])

    # save the results
    with open(result_dir, 'a', newline='') as f:
        writer = csv.writer(f)
        my_list = [classifier_name, Feature, parameters, test_confusion_matrix, test_acc, test_pre, test_rec, test_f1,
                   test_class1_pre, test_class1_rec, test_class1_f1,
                   test_class0_pre, test_class0_rec, test_class0_f1]
        writer.writerow(my_list)

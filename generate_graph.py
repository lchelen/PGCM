import numpy as np
import math

def generate_graph_with_data(data, length, threshold=0.05):

    node_num = data.shape[1]
    if len(data.shape) == 2:
        dim = 1
        data = np.expand_dims(data, axis=-1)
    if len(data.shape) == 3:
        dim = data.shape[2]
    else:
        print('Wrong data format! Shape of data is:', data.shape)
        exit(0)
    adj_mx = np.zeros((node_num, node_num))
    demand_zero = np.zeros((length, dim))
    for i in range(node_num):
        node_i = data[-length:, i, :]
        adj_mx[i, i] = 1
        if np.array_equal(node_i, demand_zero):
            continue
        else:
            for j in range(i + 1, node_num):
                node_j = data[-length:, j, :]
                distance = math.exp(-(np.abs((node_j - node_i)).sum() / length*dim))
                if distance > threshold:
                    adj_mx[i, j] = 1
                    adj_mx[j, i] = 1
    sparsity = adj_mx.sum() / (node_num * node_num)
    print("Sparsity of the adjacent matrix is: ", sparsity)
    #print(adj_mx)
    return adj_mx

def generate_graph_with_topn(data, topn=3):
    #data shape is [sample_nums, node_nums, dims] or [sample_nums, node_nums]
    length = data.shape[0]
    node_num = data.shape[1]
    if len(data.shape) == 2:
        dim = 1
        data = np.expand_dims(data, axis=-1)
    if len(data.shape) == 3:
        dim = data.shape[2]
    else:
        print('Wrong data format! Shape of data is:', data.shape)
        exit(0)
    adj_mx = np.zeros((node_num, node_num))
    demand_zero = np.zeros((length, dim))
    for i in range(node_num):
        node_i = data[:, i, :]
        if np.array_equal(node_i, demand_zero):
            continue
        else:
            distance_list=[]
            for j in range(node_num):
                node_j = data[:, j, :]
                #the bigger the distance, the similar they are, maximum distance is 1
                distance = math.exp(-(np.abs((node_j - node_i)).sum() / length*dim))
                distance_list.append(distance)
            distance_list = np.array(distance_list)
            topn_index = distance_list.argsort()[-1:-topn-1:-1]
            adj_mx[i, topn_index] = 1
    sparsity = adj_mx.sum() / (node_num * node_num)
    print("Sparsity of the adjacent matrix is: ", sparsity)
    #print(adj_mx)
    return adj_mx

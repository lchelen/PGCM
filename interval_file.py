from GCN2Seq import GCN2S as Network
from generate_graph import generate_graph_with_data
from gcn_layer import First_Approx, Cheb_Poly, Scaled_Laplacian
from GCN2SeqParser import get_parser
from load_data import get_dataloader
from TrainInits import *
from metrics import MAE_torch, RMSE_torch, MAPE_torch,MAE_NP,RMSE_NP
import torch
import time
import torch.nn as nn
import torch.nn.init as init
import os
import numpy as np
from scipy import stats

base_dir = os.path.dirname(os.path.abspath(__file__))

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

def check_gradients(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def eval(model, dataloader, scaler):
    pred = []
    true = []
    for index, (x, y) in enumerate(dataloader):
        batch_pred = model(x)
        pred.append(batch_pred.data)
        true.append(y.data)
    pred = torch.cat(pred, dim=0)
    true = torch.cat(true, dim=0)
    pred = scaler.inverse_transform(pred)
    true = scaler.inverse_transform(true)
    mae = MAE_NP(pred, true,mask_value=2)
    rmse = RMSE_NP(pred, true,mask_value=2)
    mape = MAPE_torch(pred, true, mask_value=15)
    return mae, rmse, mape, pred, true

def main(args):
    if args.dataset == 'SYDNEY':
        from load_data import Load_Sydney_Demand_Data
        data = Load_Sydney_Demand_Data(os.path.join(base_dir, '1h_data_new3.csv'))
        data = np.expand_dims(data, axis=-1)
        args.dim = 1
        print(data.shape)
    adj = generate_graph_with_data(data, len(data), threshold=args.threshold)
    adj = torch.from_numpy(Cheb_Poly(Scaled_Laplacian(adj), 2)).type(torch.float32)

    model = Network(adj, args, dropout=0.15)
    model_path = '/home/canli/upload_file/save_model/15para_model.pkl'
    model.load_state_dict(torch.load(model_path))

    print_model_parameters(model)
    model = model.to(args.device)

    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args.dataset,
                                                                               args.batch_size,
                                                                               args.window,
                                                                               normalizer='max')

    pred_matrix = []
    pred_tensor = torch.Tensor().cuda()
    for i in range(10):
        test_mae, test_rmse, test_mape, pred, true = eval(model, test_dataloader, scaler)
        print('Test---MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}'.format(test_mae, test_rmse, test_mape))
        pred = pred.squeeze()
        true = true.squeeze()
        pred = pred.unsqueeze(0)
        pred_tensor = torch.cat((pred_tensor,pred),0)

    ave = torch.mean(pred_tensor,dim=0)
    std = torch.std(pred_tensor,dim=0)
    ave = ave.cpu().detach().numpy()
    std = std.cpu().detach().numpy()
    true = true.cpu().detach().numpy()
    interval = stats.norm.interval(0.95, ave, std)
    span = interval[1] - interval[0]
    compare = (true < interval[1]) & (true > interval[0])
    per = np.sum(compare) / (ave.shape[0] * ave.shape[1])
    print(per)
    print(np.mean(span))

if __name__ == '__main__':
    args = get_parser().parse_args()
    args = init_device(args)
    init_seed(args)
    start_time = time.time()
    main(args)
    print("Total time cost: %s min ---" % ((time.time() - start_time) / 60))






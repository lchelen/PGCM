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
    model.eval()
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
    return mae, rmse, mape

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

    print_model_parameters(model)
    model.apply(init_weights)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=args.lr_init,
                                 betas=(0.8, 0.999), eps=1e-7)

    lr_scheduler = init_lr_scheduler(optimizer, args)
    criterion = nn.MSELoss(reduction = 'sum')
    criterion.to(args.device)
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args.dataset,
                                                                               args.batch_size,
                                                                               args.window,
                                                                               args.horizon,
                                                                               args.valdays,
                                                                               args.testdays,
                                                                               normalizer='max')
    print('************START TRAINING************')
    n_batch = len(train_dataloader) / args.batch_size       #1920/

    path = '/home/canli/upload_file/save_model/'
    for epoch in range(1, args.epochs+1):
        train_epoch_loss = 0
        epoch_norm = 0
        model.train()
        for index, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_pred = model(x)
            train_loss = criterion(train_pred, y)
            train_loss.backward()
            grad_norm = check_gradients(model)
            epoch_norm  = epoch_norm + grad_norm
            optimizer.step()
            train_epoch_loss = train_epoch_loss + train_loss.data
        print('Epoch {}/{}: train loss: {:.4f}, grad norm: {:.6f}'.format(epoch,args.epochs,
                                                                          train_epoch_loss,
                                                                          (epoch_norm/n_batch)))
        lr_scheduler.step()
        torch.save(model.state_dict(), path + str(epoch) + 'para_model.pkl')

        val_mae, val_rmse, val_mape = eval(model, val_dataloader, scaler)
        print('Val---MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}'.format(val_mae,
                                                                      val_rmse,
                                                                      val_mape))

if __name__ == '__main__':
    args = get_parser().parse_args()
    args = init_device(args)
    init_seed(args)
    start_time = time.time()
    main(args)
    print("Total time cost: %s min ---" % ((time.time() - start_time) / 60))






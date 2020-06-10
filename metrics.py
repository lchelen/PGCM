import numpy as np
import torch

def MAE_torch(pred, true):
    return torch.mean(torch.abs(true-pred))

def RMSE_torch(pred, true):
    return torch.sqrt(torch.mean((pred-true)**2))

def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value - 0.01)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def MAE_NP(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value - 0.01)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        MAE = torch.mean(torch.abs(pred - true))
    return MAE

def RMSE_NP(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value - 0.01)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        RMSE = torch.sqrt(((pred - true) ** 2).mean())
    return RMSE

def SMAPE_torch(pred, true):
    delim = (torch.abs(true)+torch.abs(pred))/2.0
    return torch.mean(torch.abs((true-pred)/delim))

def MAE_np(preds, labels):
    MAE = np.mean(np.absolute(preds-labels))
    return MAE

def RMSE_np(preds, labels):
    RMSE = np.sqrt(np.mean(np.square(preds-labels)))
    return RMSE



def MAPE_np(preds, labels, scaler, mask_value=None):
    if mask_value != None:
        mask = np.where(labels*scaler > (mask_value-0.1), True, False)
        masked_labels = labels[mask]
        masked_preds = preds[mask]
        MAPE = np.mean(np.absolute(np.divide((masked_labels - masked_preds), masked_labels)))
    else:
        MAPE = np.mean(np.absolute(np.divide((labels - preds), labels)))
    return MAPE

def MARE_np(preds, labels, scaler, mask_value=None):
    if mask_value != None:
        mask = np.where(labels * scaler > (mask_value - 0.1), True, False)
        masked_labels = labels[mask]
        masked_preds = preds[mask]
        MARE = np.divide(np.sum(np.absolute((masked_labels - masked_preds))), np.sum(masked_labels))
    else:
        MARE = np.divide(np.sum(np.absolute((labels - preds))), np.sum(labels))
    return MARE

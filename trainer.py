import os, copy, random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import *
from baseline.models import *
from torchmetrics import Accuracy, F1Score, Precision, Recall

# -----------------------------------------------------------------------------
def train(train_loader, val_loader, scheme, fold, exp):

    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # -------------------------------------------------------------------------
    # Import model
    # -------------------------------------------------------------------------
    if 'simpletcn' in exp:
        model = force_model(embed_dim=16).to(device)
        print('simpletcn')
    elif 'mstcn' in exp:
        from baseline.model_MSTCN import MultiStageModel
        num_stages = 4
        num_layers = 10
        model = MultiStageModel(num_stages, num_layers, num_f_maps, features_dim, out_features).to(device)
        print('mstcn')
    elif 'ms2' in exp:
        from baseline.model_MSTCN2 import MS_TCN2
        num_layers_PG = 11
        num_layers_R = 10
        num_R = 3
        model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, out_features).to(device)
        print('ms2')
    elif 'asformer' in exp:
        from baseline.model_ASFormer import MyTransformer
        num_layers = 10
        channel_mask_rate = 0.3
        model = MyTransformer(3, num_layers, 2, 2, num_f_maps, features_dim, out_features, channel_mask_rate).to(device)
        print('asformer')
    elif 'gru' in exp:
        from baseline.LSTM_GRU_CLDNN import MultiLayerGRU
        model = MultiLayerGRU(features_dim, 64, 2, out_features).to(device)
        print('gru')
    elif 'lstm' in exp:
        from baseline.LSTM_GRU_CLDNN import MultiLayerLSTM
        model = MultiLayerLSTM(features_dim, 64, 2, out_features).to(device)
        print('lstm')
    elif 'bidirectional' in exp:
        from baseline.LSTM_GRU_CLDNN import BidirectionalLSTM
        model = BidirectionalLSTM(features_dim, 64, 2, out_features).to(device)
        print('bidirectional')
    elif 'cldnn' in exp:
        from baseline.LSTM_GRU_CLDNN import CLDNN
        model = CLDNN(features_dim, 64, 2, out_features).to(device)
        print('cldnn')
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    sig_m = nn.Sigmoid()

    # Metrics
    acc_metric = Accuracy(task='binary').cuda()
    f1_score_metric = F1Score(task='binary').cuda()
    precision_metric = Precision(task='binary').cuda()
    recall_metric = Recall(task='binary').cuda()
    
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)


    # -------------------------------------------------------------------------
    
    def iter_dataloader(data_loader, model, training, exp):
    
        running_loss = 0.0
        labels_all = torch.empty((0)).to(device)
        output_all = torch.empty((0)).to(device)


        for iter, (inputs, labels) in enumerate(data_loader):
            
            # Move to device
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            # Clear grads
            if training == True:
                optimizer.zero_grad()
            
            # Forward pass
            if 'mstcn' in exp or 'asformer' in exp:
                logits = model.forward(inputs, torch.ones(inputs.size(), device=device))
            elif 'ms2' in exp:
                logits = model.forward(inputs)
            else:
                logits = model.forward(inputs)
                batch_loss = criterion(logits,labels)
                logits = (logits >= 0.0).int()
            if 'ms2' in exp or 'mstcn' in exp or 'asformer' in exp:
                batch_loss = 0
                for p in logits:
                    batch_loss += criterion(p, labels)
                batch_loss /= float(len(logits))
                logits = (sig_m(logits[-1])>=0.5).int()
            
            
            if training == True:
                # Backprop
                batch_loss.backward()
                optimizer.step()

            
            # Running Loss
            running_loss += batch_loss.item()*inputs.size(0)
            
            # Used for metrics
            labels_all = torch.cat((labels_all, labels), 0)
            output_all = torch.cat((output_all, logits), 0)
            
        # Metrics
        acc = acc_metric(output_all,labels_all)
        f1 = f1_score_metric(output_all,labels_all)
        prec = precision_metric(output_all,labels_all)
        rec = recall_metric(output_all,labels_all)
        metrics = [acc,f1,prec,rec]
        
        # Epoch Loss
        loss = running_loss / len(data_loader.dataset) 

        return metrics, loss
    
    # -------------------------------------------------------------------------

    def training(model, train_loader):

        model.train()

        metrics_train, loss_epoch_train = iter_dataloader(train_loader, model, training=True, exp=exp)

        return metrics_train, loss_epoch_train

    # -------------------------------------------------------------------------

    def testing(model, val_loader):

        model.eval()

        with torch.no_grad():

            metrics_val, loss_epoch_val = iter_dataloader(val_loader, model, training=False, exp=exp)

        return metrics_val, loss_epoch_val

    # -------------------------------------------------------------------------

    # Initializations
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    min_val_loss = np.inf

    # TensorBoard
    log_dir = os.path.join(os.getcwd(),'logs', exp, scheme,fold)
    tb = SummaryWriter(log_dir)

    # -------------------------------------------------------------------------

    print('\n>>> Training\n')


    for epoch in range(1, epochs + 1):

        metrics_train, loss_epoch_train = training(model, train_loader)
        metrics_val, loss_epoch_val = testing(model, val_loader)
        print(f'Train - Loss:{loss_epoch_train:.4f}, Accuracy:{metrics_train[0]:.4f}, F1:{metrics_train[1]:.4f}, Precision:{metrics_train[2]:.4f}, Recall:{metrics_train[3]:.4f} \
               || Val - Loss:{loss_epoch_val:.4f}, Accuracy:{metrics_val[0]:.4f}, F1:{metrics_val[1]:.4f}, Precision:{metrics_val[2]:.4f}, Recall:{metrics_val[3]:.4f} Epoch:{epoch}')

        if loss_epoch_val < min_val_loss and metrics_val[1] >= 0.4:
            min_val_loss = loss_epoch_val
            best_metrics = metrics_val
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print('Best model updated: ', min_val_loss)
        # if metrics_val[1] > best_f1:
        #     best_f1 = metrics_val[1]
        #     best_metrics = metrics_val
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     best_epoch = epoch
        #     min_val_loss = loss_epoch_val
        #     print('Best F1 model updated: ', best_f1)

        # Scheduler Update
        scheduler.step()

        tb.add_scalars('Loss',{'Training' : loss_epoch_train, 'Validation' : loss_epoch_val}, epoch)
        tb.add_scalars('Accuracy',{'Training' : metrics_train[0], 'Validation' : metrics_val[0]}, epoch)
        tb.add_scalars('F1', {'Training' : metrics_train[1], 'Validation' : metrics_val[1]}, epoch)
        tb.add_scalars('Precision',{'Training' : metrics_train[2], 'Validation' : metrics_val[2]}, epoch)
        tb.add_scalars('Recall',{'Training' : metrics_train[3], 'Validation' : metrics_val[3]}, epoch)
        tb.flush()

    tb.close()


    return best_model_wts, best_metrics, min_val_loss, best_epoch

import os, copy, random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import *
from models import *
from torchmetrics import Accuracy, F1Score, Precision, Recall

# -----------------------------------------------------------------------------

def train(train_loader, val_loader, scheme, fold):

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
    model = force_model(embed_dim=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Metrics
    acc_metric = Accuracy(task='binary').cuda()
    f1_score_metric = F1Score(task='binary').cuda()
    precision_metric = Precision(task='binary').cuda()
    recall_metric = Recall(task='binary').cuda()
    
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)


    # -------------------------------------------------------------------------
    
    def iter_dataloader(data_loader,model,training):
    
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
            logits = model.forward(inputs)

            # Loss  
            batch_loss = criterion(logits,labels)
            
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

        metrics_train, loss_epoch_train = iter_dataloader(train_loader, model, training=True)

        return metrics_train, loss_epoch_train

    # -------------------------------------------------------------------------

    def testing(model, val_loader):

        model.eval()

        with torch.no_grad():

            metrics_val, loss_epoch_val = iter_dataloader(val_loader, model, training=False)

        return metrics_val, loss_epoch_val

    # -------------------------------------------------------------------------

    # Initializations
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    min_val_loss = np.inf

    # TensorBoard
    log_dir = os.path.join(os.getcwd(),'logs',scheme,fold)
    tb = SummaryWriter(log_dir)

    # -------------------------------------------------------------------------

    print('\n>>> Training\n')


    for epoch in range(1, epochs + 1):

        metrics_train, loss_epoch_train = training(model, train_loader)
        metrics_val, loss_epoch_val = testing(model, val_loader)
        print(f'Train - Loss: {loss_epoch_train:.4f}, Accuracy:{metrics_train[0]:.4f} || Val - Loss: {loss_epoch_val:.4f}, Accuracy:{metrics_val[0]:.4f}, Epoch:{epoch}')

        if loss_epoch_val < min_val_loss:
            min_val_loss = loss_epoch_val
            best_metrics = metrics_val
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        
        # Scheduler Update
        scheduler.step()

        tb.add_scalars('Loss',{'Training' : loss_epoch_train, 'Validation' : loss_epoch_val}, epoch)
        tb.add_scalars('Accuracy',{'Training' : metrics_train[0], 'Validation' : metrics_val[0]},epoch)
        tb.flush()

    tb.close()


    return best_model_wts, best_metrics, min_val_loss, best_epoch

from utils import *
from trainer import *
from dataloader import *
from baseline.models import *
import time
import argparse



def cross_valid(fold, scheme, exp):

    print(f'Training on fold {fold}')

    train_loader, val_loader = dataset(scheme,fold,batch_size_train=batch_size_train,batch_size_val=batch_size_val)

    # Training
    best_model_wts, best_metrics, min_val_loss, best_epoch = train(train_loader, val_loader, scheme, fold, exp)
    if not os.path.exists(os.path.join(os.getcwd(),'models',exp)):
        os.makedirs(os.path.join(os.getcwd(),'models',exp))
    torch.save(best_model_wts, os.path.join(os.getcwd(),'models',exp)+ '/' + scheme + '_Fold' + fold + '_best.pth')

    print(f'Best reults | Val Loss: {min_val_loss:.4f}, Epoch: {best_epoch} | Acc: {best_metrics[0]:.4f}, f1-score: {best_metrics[1]:.4f}, Precision: {best_metrics[2]:.4f}, Recall: {best_metrics[3]:.4f}')
    print('')
    return min_val_loss, best_metrics, best_epoch

# Cross-Val Scheme
parser = argparse.ArgumentParser()
parser.add_argument('-exp', default='test/ms2', help='model and exp name', type=str)
args = parser.parse_args()

# for scheme in schemes:
schemes = ['louo']
fold_list = ['1','2','3','4','5','6']

# Training on each fold
for scheme in schemes:
    start = time.time()
    folds_loss = []
    folds_acc = []
    folds_f1 = []
    folds_prec = []
    folds_rec = []
    folds_epoch = []
    print ('>>> Training on scheme: ', scheme)
    for fold in fold_list:
        fold_min_val_loss, fold_best_metrics, fold_best_epoch = cross_valid(fold, scheme, exp=args.exp)

        folds_loss.append(fold_min_val_loss)
        folds_acc.append(fold_best_metrics[0])
        folds_f1.append(fold_best_metrics[1])
        folds_prec.append(fold_best_metrics[2])
        folds_rec.append(fold_best_metrics[3])
        folds_epoch.append(fold_best_epoch)

    end = time.time()
    elapsed_time = end - start
    print('>>> Training complete: {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('')

    # Print Average Accuracy and Min Average Loss
    sum_acc = sum_f1 = sum_prec = sum_rec = 0
    for i in range(len(folds_acc)):
        print(f'Fold {i+1} -- Acc: {folds_acc[i]:.4f}, f1-score: {folds_f1[i]:.4f}, Precision {folds_prec[i]:.4f}. Recall: {folds_rec[i]:.4f} | Loss: {folds_loss[i]:.4f} | Epoch: {folds_epoch[i]}')
        sum_acc += folds_acc[i]
        sum_f1 += folds_f1[i]
        sum_prec += folds_prec[i]
        sum_rec += folds_rec[i]

    print(f'Average results on {scheme}: Acc: {sum_acc/6.0:.4f}, f1-score: {sum_f1/6.0:.4f}, Precision: {sum_prec/6.0:.4f}, Recall: {sum_rec/6.0:.4f}')



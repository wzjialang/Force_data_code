#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
import pandas as pd
from scipy import io
from baseline.eval import get_labels_start_end_time, levenstein, edit_score, f_score
import warnings
warnings.filterwarnings("ignore")


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])
        self.fc1 = nn.Linear(num_classes, num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.ModuleList([copy.deepcopy(nn.Linear(num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        out_fc = self.fc1(self.avgpool(out).flatten(1))
        outputs = out_fc.unsqueeze(0)
        for i, s in enumerate(self.stages):
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            out_fc = self.fc[i](self.avgpool(out).flatten(1))
            outputs = torch.cat((outputs, out_fc.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, vid_list_file_tst, features_path, sample_rate, results_dir):
        best_acc_test = 0
        best_f1_test = 0
        best_edit_test = 0
        overlap = [.1, .25, .5]
        replace_values = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
        replacement_values = [-100, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        logger.info("Fold: %s" % save_dir)

        for epoch in range(num_epochs):
            self.model.train()
            self.model.to(device)
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size) #torch.Size([8, 512, 735]) torch.Size([8, 735]) torch.Size([8, 10, 735])
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            current_acc = float(correct)/total
            logger.info("[Train] epoch %d: epoch loss = %f, acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               current_acc))
            

    # def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, device, sample_rate):
            self.model.eval()
            tp_test, fp_test, fn_test = np.zeros(3), np.zeros(3), np.zeros(3)
            current_f1_test = np.zeros(3)
            correct_test = 0
            total_test = 0
            edit_test = 0
            with torch.no_grad():
                df = pd.read_csv(vid_list_file_tst, header=None)
                list_of_vids = df.iloc[:, 0].tolist()
                for vid in list_of_vids:
                    #print vid
                    alldata = io.loadmat(features_path + vid.split('.')[0] + '.mat')
                    features = alldata['F'].transpose()
                    features = features[:, ::sample_rate]
                    gt_test = alldata['GT']
                    gt_test = [replacement_values[replace_values.index(x)] if x in replace_values else x for x in gt_test[0]]
                    gt_test = torch.from_numpy(np.asarray(gt_test).squeeze())
                    gt_test = gt_test.to(device)
                    input_x = torch.tensor(features, dtype=torch.float)
                    input_x.unsqueeze_(0)
                    input_x = input_x.to(device)
                    predictions_test = self.model(input_x, torch.ones(input_x.size(), device=device))
                    _, predictions_test = torch.max(predictions_test[-1].data, 1)
                    predictions_test = predictions_test.squeeze()
                    for i in range(len(gt_test)):
                        total_test += 1
                        if gt_test[i] == predictions_test[i]:
                            correct_test += 1
                    edit_test += edit_score(predictions_test, gt_test, bg_class=[-100])

                    for s in range(len(overlap)):
                        tp1, fp1, fn1 = f_score(predictions_test, gt_test, overlap[s], bg_class=[-100])
                        tp_test[s] += tp1
                        fp_test[s] += fp1
                        fn_test[s] += fn1
                
                current_acc_test = float(correct_test)/total_test
                current_edit_test = (1.0*edit_test)/len(list_of_vids)

                for s in range(len(overlap)):
                    precision = tp_test[s] / float(tp_test[s]+fp_test[s])
                    recall = tp_test[s] / float(tp_test[s]+fn_test[s])

                    f1_test = 2.0 * (precision*recall) / (precision+recall)

                    f1_test = np.nan_to_num(f1_test)
                    current_f1_test[s] = f1_test
                    
                logger.info("[Test] epoch %d: Acc = %.4f, Edit: %.4f, F1: %s" % (epoch + 1, current_acc_test, current_edit_test, str(current_f1_test)))
            

                if current_acc_test > best_acc_test:
                    best_acc_test = current_acc_test
                    best_f1_test = current_f1_test
                    best_edit_test = current_edit_test
                    torch.save(self.model.state_dict(), save_dir + "/best" + ".pth")
                    torch.save(optimizer.state_dict(), save_dir + "/best" + ".opt")
                    logger.info("[Updated] epoch %d: Acc = %.4f, Edit: %.4f, F1: %s" % (epoch + 1, current_acc_test, current_edit_test, str(current_f1_test)))

                    for vid in list_of_vids:
                        alldata_updated = io.loadmat(features_path + vid.split('.')[0] + '.mat')
                        features_updated = alldata_updated['F'].transpose()
                        features_updated = features_updated[:, ::sample_rate]
                        gt_updated = alldata_updated['GT']
                        gt_updated = [replacement_values[replace_values.index(x)] if x in replace_values else x for x in gt_updated[0]]
                        gt_updated = torch.from_numpy(np.asarray(gt_updated).squeeze())
                        gt_updated = gt_updated.to(device)
                        input_x_updated = torch.tensor(features_updated, dtype=torch.float)
                        input_x_updated.unsqueeze_(0)
                        input_x_updated = input_x_updated.to(device)
                        predictions_updated = self.model(input_x_updated, torch.ones(input_x_updated.size(), device=device))
                        _, predictions_updated = torch.max(predictions_updated[-1].data, 1)
                        predictions_updated = predictions_updated.squeeze()
                        predictions_updated = predictions_updated.cpu().numpy().astype(int)
                        f_name = vid.split('/')[-1].split('.')[0]
                        np.savetxt(results_dir + "/" + f_name + '.csv', predictions_updated, delimiter=',', fmt='%i')
        return best_acc_test, best_edit_test, best_f1_test

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
import warnings
warnings.filterwarnings("ignore")


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])
        self.fc1 = nn.Linear(num_classes, num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.ModuleList([copy.deepcopy(nn.Linear(num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        out_fc = self.fc1(self.avgpool(out).flatten(1))
        outputs = out_fc.unsqueeze(0)
        for i, R in enumerate(self.Rs):
            out = R(F.softmax(out, dim=1))
            out_fc = self.fc[i](self.avgpool(out).flatten(1))
            outputs = torch.cat((outputs, out_fc.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out
    
class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
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

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        # logger.add(sys.stdout, colorize=True, format="{message}")

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
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1) # [bz, max_len]
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
                    predictions_test = self.model(input_x)
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
                        predictions_updated = self.model(input_x_updated)
                        _, predictions_updated = torch.max(predictions_updated[-1].data, 1)
                        predictions_updated = predictions_updated.squeeze()
                        predictions_updated = predictions_updated.cpu().numpy().astype(int)
                        f_name = vid.split('/')[-1].split('.')[0]
                        np.savetxt(results_dir + "/" + f_name + '.csv', predictions_updated, delimiter=',', fmt='%i')
        return best_acc_test, best_edit_test, best_f1_test

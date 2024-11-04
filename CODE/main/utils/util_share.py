# cython: language_level=3
# -*-coding: utf-8 -*-
# @Time : 2022/4/12 13:06
# @Author :

import glob
import json
import os
import random
import sys
import numpy as np
from sklearn.metrics import classification_report, f1_score,confusion_matrix,accuracy_score,precision_score,recall_score,average_precision_score
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import shutil
from collections import Counter,defaultdict
import geopandas as gpd
import pandas as pd


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class WData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)

def collate(sample):
    graphs, labels = map(list, zip(*sample))
    return graphs[0].batch(graphs), torch.tensor([int(i) for i in labels], dtype=torch.long)
    # return graphs, torch.tensor([int(i) for i in labels], dtype=torch.long)

def load_data(path,Graph,connect=False): # TODO
    data = json.load(open(path, encoding="utf-8"))
    labels = []
    graphs = []
    for i in data:
        node_feature = None
        if connect:
            edges_dict = defaultdict(list)
            for idx in i['graph_index']:
                feature = i['datas']['edge_features'][idx[-1]]
                edges_dict[idx[0]].append(feature)
                edges_dict[idx[1]].append(feature)
            edges_feature_default = np.zeros((len(i['datas']['node_features']),len(i['datas']['edge_features'][0])))
            for idx in edges_dict:
                edges_feature_default[idx] += np.mean(np.float64(edges_dict[idx]),axis=0)
            node_feature = np.hstack((np.float64(i['datas']['node_features']),edges_feature_default))
        else:
            node_feature = np.float64(i['datas']['node_features'])
        graphs.append(Graph(i['graph_index'], node_feature, i['datas']['edge_features'],i["file_path"]))
        labels.append(i['datas']['label'])
    all_data = list(zip(graphs, torch.Tensor(labels)))
    print("data length: {}".format(labels.__len__()))
    return all_data

def training(graphs,labels,optimizer,model,loss_func,DEVICE):
    model.train()
    if isinstance(graphs,list):
        graphs = [i.to(DEVICE) for i in graphs]
    else:
        graphs = graphs.to(DEVICE)
    labels = labels.to(DEVICE)
    prediction = model(graphs)
    if isinstance(loss_func,list):
        loss = loss_func[0](prediction[0], labels)
        for i in range(1,len(loss_func)):
            loss = loss + loss_func[i](prediction[i], labels)
    else:
        loss = loss_func(prediction, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if isinstance(prediction,list):
        prediction = np.mean(np.float64([i.detach().cpu().numpy() for i in prediction]),axis=0)
    else:
        prediction = prediction.detach().cpu().numpy()
    return loss, prediction, labels.detach().cpu().numpy()

def train_epoch(train_loader,optimizer,model,loss_func,DEVICE,epoch,config_,scheduler=True):
    epoch_loss = 0
    preds_tr = []
    labels_tr = []
    total_data = len(train_loader)
    if scheduler:
        loop = tqdm(enumerate(train_loader), total =total_data)
    else:
        loop = enumerate(train_loader)
    for iter, (g, labels) in loop:
        loss_tr, prediction_tr, label_tr = training(g,labels,optimizer,model,loss_func,DEVICE)
        preds_tr.extend(prediction_tr.tolist())
        labels_tr.extend(label_tr.tolist())
        epoch_loss += loss_tr.detach().item()
        if scheduler:
            loop.set_description(f'Epoch [{epoch}/{config_.EPOCHS}]')
            loop.set_postfix(loss = loss_tr.detach().item(), lr=optimizer.param_groups[0]['lr'])
            loop.update(1)
    return epoch_loss, loss_tr, total_data, preds_tr, labels_tr

def validing(graphs, labels, model, loss_func, DEVICE):
    model.eval()
    if isinstance(graphs,list):
        graphs = [i.to(DEVICE) for i in graphs]
    else:
        graphs = graphs.to(DEVICE)
    labels = labels.to(DEVICE)
    prediction = model(graphs)
    if isinstance(loss_func,list):
        loss = loss_func[0](prediction[0], labels)+loss_func[1](prediction[1], labels)+loss_func[2](prediction[2], labels)
    else:
        loss = loss_func(prediction, labels)
    if isinstance(prediction,list):
        prediction = np.mean(np.float64([i.detach().cpu().numpy() for i in prediction]),axis=0)
    else:
        prediction = prediction.detach().cpu().numpy()
    return loss, prediction, labels.detach().cpu().numpy()

def valid_epoch(valid_loader, model, loss_func, DEVICE):
    epoch_loss = 0
    preds_va = []
    labels_va = []
    total_data = len(valid_loader)
    with torch.no_grad():
        for iter, (g, labels) in enumerate(valid_loader):
            loss_va, prediction_va, label_va = validing(g, labels, model, loss_func, DEVICE)
            preds_va.extend(prediction_va.tolist())
            labels_va.extend(label_va.tolist())
            epoch_loss += loss_va.detach().item()
    return epoch_loss, total_data, preds_va, labels_va

def testing(graphs, labels, model, DEVICE):
    model.eval()
    if isinstance(graphs,list):
        graphs = [i.to(DEVICE) for i in graphs]
    else:
        graphs = graphs.to(DEVICE)
    labels = labels.to(DEVICE)
    prediction = model(graphs)
    if isinstance(prediction,list):
        prediction = np.mean(np.float64([i.detach().cpu().numpy() for i in prediction]),axis=0)
    else:
        prediction = prediction.detach().cpu().numpy()
    return prediction, labels.detach().cpu().numpy()

def test_epoch(test_loader, model, DEVICE):
    preds_te = []
    labels_te = []
    loop = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for iter, (g, labels) in loop:
            prediction_te, label_te = testing(g, labels, model, DEVICE)
            preds_te.extend(prediction_te.tolist())
            labels_te.extend(label_te.tolist())
            loop.set_description(f'Testing')
            loop.update(1)
    return preds_te, labels_te

def test_model(test_loader, model,path, DEVICE):
    m_state_dict = torch.load(path)
    model.load_state_dict(m_state_dict)
    model.to(DEVICE)
    model.eval()
    preds_te = []
    labels_te = []
    file_paths = []
    loop = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for iter, (g, labels) in loop:
            file_paths.extend(g.file_path)
            prediction_te, label_te = testing(g, labels, model, DEVICE)
            preds_te.extend(prediction_te.tolist())
            labels_te.extend(label_te.tolist())
            loop.set_description(f'Testing')
            loop.update(1)
    return preds_te, labels_te, file_paths

def test_for_each_model(test_loader, model, path_, DEVICE,multi_num,cfg_,error_save_path=True,data_name='test'):
    paths = []
    reports = []
    mtrxs = []
    acc_s = []
    ps = []
    rs = []
    f1s = []
    error_statistics = []
    one_class_error_statistics = defaultdict(list)
    for i in range(multi_num):
        path = glob.glob(os.path.join(path_,str(i),"GCNE_BEST_V_ACC_*.pt"))[-1]
        paths.append(path)
        m_state_dict = torch.load(path)
        model.load_state_dict(m_state_dict)
        model.to(DEVICE)
        model.eval()
        preds_te = []
        labels_te = []
        file_paths = []
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        with torch.no_grad():
            for iter, (g, labels) in loop:
                file_paths_ = [g_idx.file_path for g_idx in g] if isinstance(g, list) else g.file_path
                file_paths.extend(file_paths_)
                prediction_te, label_te = testing(g, labels, model, DEVICE)
                preds_te.extend(prediction_te.tolist())
                labels_te.extend(label_te.tolist())
                loop.set_description(f'Testing')
                loop.update(1)
        f1,report,mtrx,acc,p,r = accs(preds_te,labels_te,True)
        f1s.append(f1)
        acc_s.append(acc)
        ps.append(p)
        rs.append(r)
        reports.append(report)
        mtrxs.append(mtrx)

        if error_save_path:
            error_statistic,one_class_error_statistic = view_error_classifications_multi(preds_te, labels_te,file_paths,cfg_,str(i))
            error_statistics.extend(error_statistic)
            for k,v in one_class_error_statistic.items():
                one_class_error_statistics[k].extend(v)

    metric = []
    for report,mtrx,idx,path,acc,p,r,f1 in zip(reports,mtrxs,range(multi_num),paths,acc_s,ps,rs,f1s):
        print(f"Model {idx}, Path: {path}:")
        print(f"ACC: {acc}, P: {p}, R: {r}, F1: {f1}")
        print(report)
        print(mtrx)
        # 获取report中的值
        report = report.split('\n')
        report = report[2:-5]
        report = [i.split(' ') for i in report]
        report = [i for i in report if i != ['']]
        report = [[float(j) for j in i if j != '' and j != 'accuracy'][-2] for i in report]
        mtrx = mtrx.tolist()
        metric.append({
            "ACC":acc,
            "P":p,
            "R":r,
            "report":report,
            "MTRX":mtrx,
            'F1':f1
        })
    err = Counter(error_statistics)
    # 排序
    err = sorted(err.items(), key=lambda x: x[1], reverse=True)
    # 格式化打印
    # print("Error Statistics:")
    # for i in err:
        # print(i[0], i[1])
    # print(len(err))
    # print("One Class Error Statistics:")
    one_errors = []
    for k,v in one_class_error_statistics.items():
        tmp_err = Counter(v)
        tmp_err = sorted(tmp_err.items(), key=lambda x: x[1], reverse=True)
        one_errors.append({k:tmp_err})
        # for i in tmp_err:
            # print(k,i[0], i[1])
        # print(len(tmp_err))
    json_save = {
        "metric":metric,
        "error_statistics":err,
        "one_class_error_statistics":one_errors
    }
    save_dir = os.path.split(cfg_.JSON_SAVE_PATH)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    s_path = cfg_.JSON_SAVE_PATH.replace('.json',f'_{data_name}.json')
    with open(s_path,'w',encoding='utf-8') as f:
        json.dump(json_save,f,ensure_ascii=False,indent=4)

def  easy_test(test_loader, model, path_, DEVICE,multi_num,restore=None):
    paths = []
    reports = []
    mtrxs = []
    acc_s = []
    ps = []
    rs = []
    f1s = []
    for i in range(multi_num):
        path = glob.glob(os.path.join(path_,str(i),"GCNE_BEST_V_ACC_*.pt"))[-1] # TODO
        # path = glob.glob(os.path.join(path_,str(i),"GCNE_00040.pt"))[-1]
        paths.append(path)
        m_state_dict = torch.load(path)
        model.load_state_dict(m_state_dict)
        model.to(DEVICE)
        model.eval()
        preds_te = []
        labels_te = []
        file_paths = []
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        with torch.no_grad():
            for iter, (g, labels) in loop:
                file_paths_ = [g_idx.file_path for g_idx in g] if isinstance(g,list) else g.file_path
                file_paths.extend(file_paths_)
                prediction_te, label_te = testing(g, labels, model, DEVICE)
                preds_te.extend(prediction_te.tolist())
                labels_te.extend(label_te.tolist())
                loop.set_description(f'Testing')
                loop.update(1)
        f1,report,mtrx,acc,p,r = accs(preds_te,labels_te,True)
        f1s.append(f1)
        acc_s.append(acc)
        ps.append(p)
        rs.append(r)
        reports.append(report)
        mtrxs.append(mtrx)

    for report,mtrx,idx,path,acc,p,r,f1 in zip(reports,mtrxs,range(multi_num),paths,acc_s,ps,rs,f1s):
        print(f"Model {idx}, Path: {path}:")
        print(f"ACC: {acc}, P: {p}, R: {r}, F1: {f1}")
        print(report)
        print(mtrx)

def  test_for_show(test_loader, model, path_, DEVICE,multi_num,restore=None,parent_dir=None):
    paths = []
    for i in range(multi_num):
        if parent_dir is not None:
            if i != int(parent_dir):
                continue
        path = glob.glob(os.path.join(path_,str(i),"GCNE_BEST_V_ACC_*.pt"))[-1] # TODO
        paths.append(path)
        m_state_dict = torch.load(path)
        model.load_state_dict(m_state_dict)
        model.to(DEVICE)
        model.eval()
        preds_te = []
        labels_te = []
        file_paths = []
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        with torch.no_grad():
            for iter, (g, labels) in loop:
                file_paths_ = [g_idx.file_path for g_idx in g] if isinstance(g,list) else g.file_path
                file_paths.extend(file_paths_)
                prediction_te, label_te = testing(g, labels, model, DEVICE)
                preds_te.extend(prediction_te.tolist())
                labels_te.extend(label_te.tolist())
                loop.set_description(f'Testing')
                loop.update(1)
    # preds_te = np.argmax(preds_te, axis=1)
    # all_file_data = []
    # origin_shp_file_dir = r'/to/path/DATA/California_1000/02extracted_data'
    # for idx, file_idx in enumerate(file_paths):
    #     d,n = file_idx.split('_')
    #     n = n + '.shp'
    #     tmp_path = os.path.join(origin_shp_file_dir,d,n)
    #     tmp_file = gpd.read_file(tmp_path)
    #     tmp_file['predict'] = preds_te[idx]
    #     tmp_file['label'] = labels_te[idx]
    #     tmp_file['right'] = int(preds_te[idx] == labels_te[idx])
    #     tmp_file['symbol'] = str(labels_te[idx]) + '_' + str(preds_te[idx])
    #     all_file_data.append(tmp_file)
    # all_file_data = pd.concat(all_file_data)
    # all_file_data.to_file(r'/to/path/SAVE/NEW_1000/re.shp')
    f1, report, mtrx, acc, p, r = accs(preds_te, labels_te, True)
    print(f"ACC: {acc}, P: {p}, R: {r}, F1: {f1}")
    print(report)
    print(mtrx)


def accs(prediction, labels,test=False):
    prediction = np.argmax(prediction, axis=1)
    f1 = f1_score(labels, prediction, average='macro')
    acc = accuracy_score(labels, prediction)
    p = precision_score(labels, prediction, average='macro')
    r = recall_score(labels, prediction, average='macro')
    if test:
        return f1, classification_report(labels, prediction, digits=4),confusion_matrix(labels, prediction),acc,p,r
    return f1

def build_dataloader_from_file(path,batch_size,shuffle,Graph_class,num_workers=0):
    data = load_data(path,Graph_class)
    return DataLoader(WData(data), batch_size=batch_size, shuffle=shuffle, collate_fn=collate,num_workers=num_workers)

def make_dirs(config_,isDebug,dir=None):
    save_time = config_.SAVE_TIME if not isDebug else "debug"
    if dir is not None:
        save_time = str(dir)
    # 建立模型保存文件夹
    save_path = os.path.join(config_.SAVE_PATH, save_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def model_restore(model, path,pretrain=False):
    start = 0
    if path is not None:
        m_state_dict = torch.load(path)
        start = int(path.split('\\')[-1].split('.')[-2].split('_')[-1])
        if pretrain:
            new_state_dict = {}
            for k,v in m_state_dict.items():
                if k != 'linear_3.weight' and k != 'linear_3.bias':
                    new_state_dict[k] = v
            re_gcne = model.get_restore_gcne()
            re_gcne.load_state_dict(new_state_dict)
            model.restore(re_gcne)
        else:
            model.load_state_dict(m_state_dict)
    return model, start

def print_info(config_):
    """
    打印配置信息
    """
    def print_config(config_):

        config = config_.__dict__
        for i in config:
            if i != 'data':
                print("[INFO]:{}:{}".format(i, str(getattr(config_,i))))
    current_device = torch.cuda.current_device()
    print("[INFO]:python version:{}".format(sys.version))
    print("[INFO]:pytorch version:{}".format(torch.__version__))
    print("[INFO]:cuda version:{}".format(torch.version.cuda))
    print("[INFO]:cudnn version:{}".format(torch.backends.cudnn.version()))
    print("[INFO]:gpu numbers:{}".format(torch.cuda.device_count()))
    print("[INFO]:gpu name:{}".format(torch.cuda.get_device_name(current_device)))
    print("[INFO]:gpu memory:{}".format(torch.cuda.get_device_properties(current_device).total_memory))
    print("[INFO]:gpu memory cached:{}".format(torch.cuda.memory_cached(current_device)))
    print("[INFO]:gpu memory allocated:{}".format(torch.cuda.memory_allocated(current_device)))
    print_config(config_)
    current_device = torch.cuda.current_device()
    isDebug = True if sys.gettrace() else False
    if isDebug:
        config_.WANDB = False
    else:
        config_.WANDB = True
    return isDebug

def save_model(model, save_path, epoch, v_ac, best_acc,config_,isDebug,epoch_now):
    if v_ac > best_acc:
        last_best = glob.glob(os.path.join(save_path, "GCNE_BEST_V_ACC_*.pt"))
        for i in last_best:
            os.remove(i)
        torch.save(model.state_dict(), os.path.join(
            save_path, "GCNE_BEST_V_ACC_{}.pt".format(epoch_now)))
        best_acc = v_ac
    if epoch % config_.SAVE_STEP == 0 and not isDebug:
        torch.save(model.state_dict(), os.path.join(
            save_path, "GCNE_{}.pt".format(str(epoch).zfill(5))))
    return best_acc

def save_error_classifications(pic_path,save_path,suffix='.svg'):
    file_base_dir,file_name = os.path.split(save_path)
    if not os.path.exists(file_base_dir):
        os.makedirs(file_base_dir)
    file_name = file_name + suffix
    pic_path = os.path.join(pic_path,file_name)
    save_path = save_path + suffix
    shutil.copy(pic_path,save_path)


def view_error_classifications(p, l,file_paths, config_):
    p = np.argmax(p, axis=1)
    save_base_path = os.path.join(config_.ERROR_SAVE_PATH, config_.SAVE_TIME)
    for i in tqdm(range(len(p)),desc='Saving Error Classifications'):
        if p[i] != l[i]:
            save_path = os.path.join(save_base_path,str(l[i]) + '_' + str(p[i]),file_paths[i])
            save_error_classifications(config_.PNG_PATH,save_path)

def view_error_classifications_multi(p, l,file_paths, config_,idx):
    error_statistic = []
    one_class_error_statistic = defaultdict(list)
    p = np.argmax(p, axis=1)
    save_base_path = os.path.join(config_.ERROR_SAVE_PATH,idx)
    for i in tqdm(range(len(p)),desc='Saving Error Classifications'):
        if p[i] != l[i]:
            error_statistic.append(file_paths[i])
            one_class_error_statistic[str(l[i]) + '_' + str(p[i])].append(file_paths[i])
            save_path = os.path.join(save_base_path,str(l[i]) + '_' + str(p[i]),file_paths[i])
            save_error_classifications(config_.PNG_PATH,save_path)
    return error_statistic,one_class_error_statistic


# def test_model_multi(test_loader, model, path_, DEVICE,multi_num):
#     results = []
#     for i in range(multi_num):
#         path = glob.glob(os.path.join(path_,str(i),"GCNE_BEST_V_ACC_*.pt"))[-1]
#         print(path)
#         m_state_dict = torch.load(path)
#         model.load_state_dict(m_state_dict)
#         model.to(DEVICE)
#         model.eval()
#         preds_te = []
#         labels_te = []
#         file_paths = []
#         loop = tqdm(enumerate(test_loader), total=len(test_loader))
#         with torch.no_grad():
#             for iter, (g, labels) in loop:
#                 file_paths.extend(g.file_path)
#                 prediction_te, label_te = testing(g, labels, model, DEVICE)
#                 preds_te.extend(prediction_te.tolist())
#                 labels_te.extend(label_te.tolist())
#                 loop.set_description(f'Testing')
#                 loop.update(1)
#         results.append(preds_te)
#     results = np.mean(results,axis=0)
#     return results,labels_te,file_paths

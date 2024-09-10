import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import FusionVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


num_workers = 8
criterion  = nn.CrossEntropyLoss()


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = FusionVitNet(args, True)
        self. init_lr=args["init_lrate"]
        self.weight_decay=args["init_weight_decay"] if args["init_weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args
        self.task_sizes = []

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()

    def incremental_train(self, data_manager,layers=None):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.args['train_batch_size'], shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.args['test_batch_size'], shuffle=False, num_workers=num_workers)
        self.args["mode"] = "train"
        self._train(self.train_loader, self.test_loader, layers)

    def _train(self, train_loader, test_loader, layers):

        self._network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD([
                {'params': self._network.backbone.parameters(), 'lr': self.args["init_lrate"]},
                {'params': self._network.fusion_module.parameters(), 'lr': self.args["init_lrate"]},
                {'params': self._network.fc.parameters(), 'lr': self.args["init_lrate"]},
                                    ], momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['init_epoch'], eta_min=self.min_lr)
            # self._init_train(train_loader, test_loader, optimizer, scheduler,layers)
            self.setup_RP()
            self.replace_fc(train_loader)
        else:
            self.replace_fc(train_loader)


    def setup_RP(self):
        self.initiated_G=False
        self._network.fc.use_RP=True
        M=self._network.fc.in_features 
        self.Q=torch.zeros(M,self.args["all_classes"])
        self.G=torch.zeros(M,M)

    def replace_fc(self,trainloader):
        self._network = self._network.eval()
        Features_f = []
        label_list = []

        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding = self._network(data)["features"]
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y=target2onehot(label_list,self.args['all_classes']) 
        if self.args['Decorrelation']:
            Features_h=Features_f 
            self.Q=self.Q+Features_h.T @ Y 
            self.G=self.G+Features_h.T @ Features_h 
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T 
            self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(self._device)

        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(label_list==class_index).nonzero().squeeze(-1)
                class_prototype=Features_f[data_index].mean(0)
                self._network.fc.weight.data[class_index]=class_prototype #for cil, only new classes get updated

    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9) 
        num_val_samples=int(Features.shape[0]*0.8) 
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:] 
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:] 
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T 
            Y_train_pred=Features[num_val_samples::,:]@Wo.T 
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:])) 
        ridge=ridges[np.argmin(np.array(losses))] 
        logging.info("Optimal lambda: "+str(ridge))
        return ridge 



    def _init_train(self, train_loader, test_loader, optimizer, scheduler,layers):

        prog_bar = tqdm(range(self.args['init_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs,layers)["logits"]
                loss= F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['init_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)



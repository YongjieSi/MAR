import json
import random
import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.data_manager_for_fscil import DataManager
from utils.toolkit import count_parameters
import os
from utils.averager import AverageMeter
import pandas as pd
import numpy as np
import pandas


def train(args):
    _set_device(args)
    _train(args)

def _train(args):
    try:
        os.mkdir("/data/syj/MAR/logs/{}".format(args['model_name']))
    except:
        pass
    logfilename = '/data/syj/MAR/logs/{}/{}_seed{}_{}_{}_{}_{}initcls_{}way_{}shot_bepochs{}_blrate{}_ilrate{}'.format(args['model_name'], args['prefix'], args['seed'], args['convnet_type'], args['convnet_type'],
                                                args['dataset'], args['init_cls'], args['increment'], args['shot'], args['init_epoch'], args['init_lrate'], args['init_lrate'])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'],args['shot'])

    _set_random(args["seed"])
    print_args(args)

    model = factory.get_model(args['convnet_type'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    sess_acc_dict = {}
    sess_acc_per_class = {}

    for task in range(args['nb_tasks']):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        nt_strat_time = time.time()
        model.incremental_train(data_manager)
        nt_end_time = time.time()
        spend_time = nt_end_time - nt_strat_time
        logging.info('Task{} spend {} seconds to train'.format(task, spend_time))
        cnn_accy, nme_accy, acc_dict, cls_sample_count, acc_per_class = model.eval_task()
        sess_acc_dict[f'sess {task}'] = acc_dict
        sess_acc_per_class[f'sess {task}'] = acc_per_class
        logging.info(f"sess {task} acc_dict:{acc_dict}")
        logging.info(f"sess {task} acc_per_class: {acc_per_class}")
        model.after_task()

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])

            nme_curve['top1'].append(nme_accy['top1'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))

    out_dict = {}
    out_dict['cur_acc'] = []
    out_dict['former_acc'] = []
    out_dict['both_acc'] = []
    for k, v in sess_acc_dict.items():
        out_dict['cur_acc'].append(v['cur_acc'])
        out_dict['former_acc'].append(v['former_acc'])
        out_dict['both_acc'].append(v['all_acc'])
    out_dict['cur_acc'].append(np.round(np.mean(out_dict['cur_acc'], axis=0),4))
    out_dict['cur_acc'].append(np.round((out_dict['cur_acc'][0]-out_dict['cur_acc'][9]),4))
    out_dict['former_acc'].append(np.round(sum(out_dict['former_acc'])/9,4))
    out_dict['former_acc'].append(np.round((out_dict['former_acc'][1]-out_dict['former_acc'][9]),4))
    out_dict['both_acc'].append(np.round(np.mean(out_dict['both_acc'], axis=0),4))
    out_dict['both_acc'].append(np.round((out_dict['both_acc'][0]-out_dict['both_acc'][9]),4))
    out_df = pandas.DataFrame(out_dict)
    out_df = out_df.T
    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None) 
    pandas.set_option('display.width', None)
    pandas.set_option('display.max_colwidth', None)
    logging.info(f"final output:{out_dict}")
    logging.info(f"\n****************************************Pretty Output********************************************\
                \n{out_df}\
                \n***********************************************************************************************")
    return out_dict



def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))

import os, shutil, argparse
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import yaml

from lmdb_data_loader_A import LmdbDataset

from models.resnet_conformer_audio import ResnetConformer_sed_doa_nopool

from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from utils.cls_tools.cls_compute_seld_results_2024 import ComputeSELDResults
from utils.write_csv import write_output_format_file

from utils.sed_doa import SedDoaResult, process_foa_input_sed_doa_labels, SedDoaLoss, SedDoaLoss_2024

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def main(args):
    # 设置log
    log_output_folder = os.path.dirname(args['result']['log_output_path'])
    os.makedirs(log_output_folder, exist_ok=True)
    logging.basicConfig(filename=args['result']['log_output_path'], filemode='w', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args['model']['type'] == 'seddoa_nopool':
        data_process_fn = process_foa_input_sed_doa_labels
        result_class = SedDoaResult
        criterion = SedDoaLoss_2024(loss_weight=[0.1,1,0.5])
        model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], in_dim=args['model']['in_dim'], out_dim=args['model']['out_dim'])

    test_split = [4]
    test_dataset = LmdbDataset(args['data']['test_lmdb_dir'], test_split, normalized_features_wts_file=args['data']['norm_file'],
                                ignore=args['data']['test_ignore'], segment_len=None, data_process_fn=data_process_fn)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=False, 
        num_workers=args['train']['test_num_workers'], collate_fn=test_dataset.collater
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    #pdb.set_trace()
    model = model.to(device)
    logger.info(model)
    set_random_seed(12332)

    if args['model']['pre-train']:
        model.load_state_dict(torch.load(args['model']['pre-train_model'], map_location=device))
    logger.info(model)

    start_time = time.time()
    model.eval()
    test_loss = []
    test_result = result_class(segment_length=args['data']['segment_len'])
    #pdb.set_trace()
    for data in test_dataloader:
        input = data['input'].to(device)
        target = data['target'].to(device)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            test_loss.append(loss.item())
        test_result.add_items(data['wav_names'], output)
    output_dict = test_result.get_result()
    test_time = time.time() - start_time
    
    # 保存测试集CSV文件
    dcase_output_val_dir = os.path.join(args['result']['dcase_output_dir'], 'best_results')
    # if os.path.exists(dcase_output_val_dir):
    #     shutil.rmtree(dcase_output_val_dir)
    os.makedirs(dcase_output_val_dir, exist_ok=True)
    for csv_name, perfile_out_dict in output_dict.items():
        output_file = os.path.join(dcase_output_val_dir, '{}.csv'.format(csv_name))
        write_output_format_file(output_file, perfile_out_dict)
    
    #根据保存的CSV文件进行结果评估
    score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
    val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_val_dir)
    logger.info('F/AE/Dist_err/Rel_dist_err/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_F, val_LE, val_dist_err, val_rel_dist_err, val_seld_scr)))
    print('F/AE/Dist_err/Rel_dist_err/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_F, val_LE, val_dist_err, val_rel_dist_err, val_seld_scr)))
    print('Classwise results on unseen test data')
    print('Class\tF\tAE\tdist_err\treldist_err\tSELD_score')
    # pdb.set_trace()
    for cls_cnt in range(0,13):
        print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
            cls_cnt,
            classwise_test_scr[1][cls_cnt],
            classwise_test_scr[2][cls_cnt],
            classwise_test_scr[3][cls_cnt],
            classwise_test_scr[4][cls_cnt],
            classwise_test_scr[6][cls_cnt]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', help='name of config')
    input_args = parser.parse_args()

    # 不同任务使用不同配置文件
    # foa_dev_seddoa_nopool
    # foa_dev_accdoa_nopool
    # foa_dev_multi_accdoa_nopool
    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    main(args)
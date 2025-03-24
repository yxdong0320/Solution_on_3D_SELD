import os, shutil
import numpy as np
import math
import lmdb
import argparse
from tqdm import tqdm
from datum_pb2 import SimpleDatum
import sys
import pdb
#sys.path.append('/work1/ssd/mchu2/competition/DCASE/visual_src')
#from visual_tools import VisualTools


def npy_to_lmdb(npy_data_dir, npy_label_dir, lmdb_out_dir, segment_length_s
    #, visual_tools
      ):

    lmdb_map_size = 1099511627776

    if os.path.exists(lmdb_out_dir):
        shutil.rmtree(lmdb_out_dir)
    os.makedirs(lmdb_out_dir)
    
    env = lmdb.open(lmdb_out_dir, map_size=lmdb_map_size)
    txn = env.begin(write=True)
    lmdb_key_name  = lmdb_out_dir + "/keys.txt"
    total_key_file = open(lmdb_key_name, 'w')

    # segment_length_s = 10
    segment_label_frame_num = int(segment_length_s / 0.1)  # 100
    segment_data_frame_num = segment_label_frame_num * 5  # 500

    Num_count = 0
    #lists = list()
    #for filename in os.listdir(npy_data_dir):
    #    if int(filename[4]) == 1 or int(filename[4]) == 2:
    #        lists.append(filename)

    data_cat_dict = {"2": "dev-test-tau",
                     "4": "train-tau",
                     "6": "dev-train-tau",
                     "7": "dev-train-tau",
                     "8": "dev-test-tau",
                     "9": "dev-train-tau",
                     "10": "dev-test-tau",
                     "12": "dev-train-tau",
                     "13": "dev-train-tau",
                     "14": "dev-train-tau",
                     "15": "dev-test-tau",
                     "16": "dev-test-tau",
                     "21": "dev-train-sony",
                     "22": "dev-train-sony",
                     "23": "dev-test-sony",
                     "24": "dev-test-sony"
    }

    for filename in tqdm(os.listdir(npy_data_dir)):
        # data_cat_num = filename.split("_")[1].split("room")[1]
        # data_cat = data_cat_dict.get(data_cat_num, None)
        # assert data_cat is not None, "data cat is Error!!!!!!!!!!"
        # _filename = filename.split(".")[0]

        data = np.load(os.path.join(npy_data_dir, filename))
        label = np.load(os.path.join(npy_label_dir, filename))
        #pdb.set_trace()

        label_frame_num = min(data.shape[0]//5, label.shape[0])
        data_frame_num = label_frame_num * 5

        segment_num = math.ceil(data_frame_num / segment_data_frame_num)
        for seg_id in range(segment_num):
            segment_data = data[seg_id*segment_data_frame_num:(seg_id+1)*segment_data_frame_num]
            segment_label = label[seg_id*segment_label_frame_num:(seg_id+1)*segment_label_frame_num]
            wav_name = filename.split('.')[0] + '_seg_{}_{}'.format(segment_num, seg_id)

            datum = SimpleDatum()
            # _, _, visual_data = visual_tools.load_file.load_keypoint_feature(data_cat=data_cat, file_name=_filename)
            datum.data = segment_data.astype(np.float32).tobytes()
            datum.label = segment_label.astype(np.float32).tobytes()
            datum.data_dim = segment_data.reshape(segment_data.shape[0], -1).shape[-1]
            datum.label_dim = segment_label.reshape(segment_label.shape[0], -1).shape[-1]
            datum.wave_name = wav_name.encode()
            txn.put(wav_name.encode(), datum.SerializeToString())
            total_key_file.write('{}\n'.format(wav_name))
            total_key_file.flush()

            Num_count += 1
            if (Num_count % 1000) == 0:
                print("save the %d sample" % Num_count)
                txn.commit()
                txn = env.begin(write=True)
    
    print("save the %d sample" % Num_count)
    txn.commit()
    env.close()
    total_key_file.close()


def load_output_format_file(_output_format_file):
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])-1
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5: #polar coordinates 
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        elif len(_words) == 6: # cartesian coordinates
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict


def get_meta_label(meta_dict, frame_cnt):
    # output [5,13,2], means 5 instance, 13 class, 2 polar dims
    output = np.ones((5, 13, 2), dtype=np.float32) * (-10000)
    if frame_cnt in meta_dict:
        meta_data = meta_dict[frame_cnt]
        for i,instance_data in enumerate(meta_data):
            output[i,instance_data[0],0] = instance_data[2]
            output[i,instance_data[0],1] = instance_data[3]
    return output


def npy_foa_mic_meta_to_lmdb(foa_npy_dir, mic_npy_dir, metadata_dir, lmdb_out_dir):
    lmdb_map_size = 1099511627776

    if os.path.exists(lmdb_out_dir):
        shutil.rmtree(lmdb_out_dir)
    os.makedirs(lmdb_out_dir)
    
    env = lmdb.open(lmdb_out_dir, map_size=lmdb_map_size)
    txn = env.begin(write=True)
    lmdb_key_name  = lmdb_out_dir + "/keys.txt"
    total_key_file = open(lmdb_key_name, 'w')

    segment_length_s = 20
    segment_label_frame_num = int(segment_length_s / 0.1)
    segment_data_frame_num = segment_label_frame_num * 5

    Num_count = 0
    for filename in os.listdir(foa_npy_dir):
        foa_npy_path = os.path.join(foa_npy_dir, filename)
        mic_npy_path = os.path.join(mic_npy_dir, filename)
        metedata_path = os.path.join(metadata_dir, filename.replace('.npy', '.csv'))
        if not os.path.exists(foa_npy_path):
            raise RuntimeError('file not exists: {}'.format(foa_npy_path))
        if not os.path.exists(mic_npy_path):
            raise RuntimeError('file not exists: {}'.format(mic_npy_path))
        if not os.path.exists(metedata_path):
            raise RuntimeError('file not exists: {}'.format(metedata_path))

        foa_npy = np.load(foa_npy_path).reshape(-1,7,64)
        mic_npy = np.load(mic_npy_path).reshape(-1,10,64)
        foa_mic_npy = np.concatenate((foa_npy, mic_npy), axis=1)

        meta_dict = load_output_format_file(metedata_path)
        assert len(foa_npy) == len(mic_npy)
        label_frame_num = len(foa_npy) // 5
        label = []
        for frame_cnt in range(label_frame_num):
            frame_label_npy = get_meta_label(meta_dict, frame_cnt)
            label.append(frame_label_npy)
        label_npy = np.stack(label, axis=0)

        label_frame_num = min(foa_mic_npy.shape[0]//5, label_npy.shape[0])
        data_frame_num = label_frame_num * 5
        segment_num = math.ceil(data_frame_num / segment_data_frame_num)
        for seg_id in range(segment_num):
            segment_data = foa_mic_npy[seg_id*segment_data_frame_num:(seg_id+1)*segment_data_frame_num]
            segment_label = label_npy[seg_id*segment_label_frame_num:(seg_id+1)*segment_label_frame_num]
            wav_name = filename.split('.')[0] + '_seg_{}_{}'.format(segment_num, seg_id)

            datum = SimpleDatum()
            datum.data = segment_data.astype(np.float32).tobytes()
            datum.label = segment_label.astype(np.float32).tobytes()
            datum.data_dim = segment_data.reshape(segment_data.shape[0], -1).shape[-1]
            datum.label_dim = segment_label.reshape(segment_label.shape[0], -1).shape[-1]
            datum.wave_name = wav_name.encode()
            txn.put(wav_name.encode(), datum.SerializeToString())
            total_key_file.write('{}\n'.format(wav_name))
            total_key_file.flush()

            Num_count += 1
            if (Num_count % 1000) == 0:
                print("save the %d sample" % Num_count)
                txn.commit()
                txn = env.begin(write=True)
                
    print("save the %d sample" % Num_count)
    txn.commit()
    env.close()
    total_key_file.close()

def npy_to_lmdb2(npy_data_dir, lmdb_out_dir, segment_length_s):

    lmdb_map_size = 1099511627776

    if os.path.exists(lmdb_out_dir):
        shutil.rmtree(lmdb_out_dir)
    os.makedirs(lmdb_out_dir)
    
    env = lmdb.open(lmdb_out_dir, map_size=lmdb_map_size)
    txn = env.begin(write=True)
    lmdb_key_name  = lmdb_out_dir + "/keys.txt"
    total_key_file = open(lmdb_key_name, 'w')

    # segment_length_s = 10
    segment_label_frame_num = int(segment_length_s / 0.1)  # 100
    segment_data_frame_num = segment_label_frame_num * 5  # 500

    Num_count = 0

    for filename in tqdm(os.listdir(npy_data_dir)):

        data = np.load(os.path.join(npy_data_dir, filename))

        data_frame_num = data.shape[0]

        segment_num = math.ceil(data_frame_num / segment_data_frame_num)
        for seg_id in range(segment_num):
            segment_data = data[seg_id*segment_data_frame_num:(seg_id+1)*segment_data_frame_num]
            wav_name = filename.split('.')[0] + '_seg_{}_{}'.format(segment_num, seg_id)

            datum = SimpleDatum()
            datum.data = segment_data.astype(np.float32).tobytes()
            datum.data_dim = segment_data.reshape(segment_data.shape[0], -1).shape[-1]
            datum.wave_name = wav_name.encode()
            txn.put(wav_name.encode(), datum.SerializeToString())
            total_key_file.write('{}\n'.format(wav_name))
            total_key_file.flush()

            Num_count += 1
            if (Num_count % 1000) == 0:
                print("save the %d sample" % Num_count)
                txn.commit()
                txn = env.begin(write=True)
    
    print("save the %d sample" % Num_count)
    txn.commit()
    env.close()
    total_key_file.close()



def debug_read_lmdb(lmdb_dir):
    env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
    txn = env.begin()
    with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
        keys = f.readlines()
    with txn.cursor() as cursor:
        for key in keys:
            k = key.strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.fromstring(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            label = np.fromstring(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            wav_name = datum.wave_name.decode()
            print(wav_name)
            print('data:', data.shape)
            print('label:', label.shape)

def debug_read_lmdb2(lmdb_dir):
    env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
    txn = env.begin()
    with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
        keys = f.readlines()
    with txn.cursor() as cursor:
        for key in keys:
            k = key.strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.fromstring(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            wav_name = datum.wave_name.decode()
            print(wav_name)
            print('data:', data.shape)


if __name__ == "__main__":
    # 2022
    # 2024
    npy_data_dir = '/disk6/yxdong/Dcase2023/Glass-SELD/feat_label/foa_dev'
    npy_label_dir = '/disk6/yxdong/Dcase2023/Glass-SELD/feat_label/foa_dev_label'
    lmdb_out_dir = '/disk6/yxdong/Dcase2023/Glass-SELD/lmdb_glass_synthdata_len10s'
    segment_length_s=10
    npy_to_lmdb(npy_data_dir, npy_label_dir, lmdb_out_dir,segment_length_s)
    debug_read_lmdb(lmdb_out_dir)

    '''npy_data_dir2 = '/disk3/yxdong/Dcase2023/Data/feat_label/foa_dev'
    #npy_label_dir = '/disk3/yxdong/Dcase2023/Data/data_aug/TestSet/feat_label/foa_dev_label'
    lmdb_out_dir2 = '/disk3/yxdong/Dcase2023/Data/feat_label/lmdb_foa_dev_data_label_test_len10s'
    #segment_length_s=10
    npy_to_lmdb2(npy_data_dir2, lmdb_out_dir2,segment_length_s)
    debug_read_lmdb2(lmdb_out_dir2)'''


    #foa_npy_dir = '/yrfs1/intern/qingwang28/DCASE2022/data/data_aug/ACS/feat_label/foa_dev'
    #mic_npy_dir = '/yrfs1/intern/qingwang28/DCASE2022/data/data_aug/ACS/feat_label/mic_dev'
    #metadata_dir = '/yrfs1/intern/qingwang28/DCASE2022/data/data_aug/ACS/metadata_dev'
    #lmdb_out_dir = '/work1/sppro/hxwu2/DCASE/data/data_aug/ACS/lmdb_foa_mic_meta_dev'
    #npy_foa_mic_meta_to_lmdb(foa_npy_dir, mic_npy_dir, metadata_dir, lmdb_out_dir)

    # 2023
    # python utils/lmdb_tools/convert_lmdb.py -len 10
    '''parser = argparse.ArgumentParser('convert lmdb')
    parser.add_argument('--npy_data_dir', type=str, default='/yrfs2/cv1/jszhang6/qingwang28/Test/test_feat')
    #parser.add_argument('--npy_label_dir', type=str, default='/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/feat_label/foa_dev_label')
    parser.add_argument('--lmdb_out_dir', type=str, default='/yrfs2/cv1/jszhang6/qingwang28/Test/lmdb_testdata_len20s')
    #parser.add_argument('--visual_tool_config_path', type=str, default='/work1/ssd/mchu2/competition/DCASE/visual_src/config.yaml')
    parser.add_argument('--len', type=int, default=20, help='segment length (s)')
    args = parser.parse_args()'''


    '''npy_data_dir = args.npy_data_dir
    #npy_label_dir = args.npy_label_dir
    lmdb_out_dir = args.lmdb_out_dir
    segment_length = args.len
    #visual_tool_config_path=args.visual_tool_config_path
    #visual_tools = VisualTools(visual_tool_config_path)
    # 192h
    # npy_data_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/data_aug/ACS/feat_label/foa_dev'
    # npy_label_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/data_aug/ACS/feat_label/foa_dev_label'
    # lmdb_out_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/data_aug/ACS/feat_label/lmdb_foa_dev_data_label_len10s'
    # 24h
    # npy_data_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/combine_synth_real/feat_label/foa_dev'
    # npy_label_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/combine_synth_real/feat_label/foa_dev_adpit_label'
    # lmdb_out_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/combine_synth_real/feat_label/lmdb_foa_dev_data_adpit_label_len20s'
    # 4h
    # npy_data_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/feat_label/foa_dev'
    # npy_label_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/feat_label/foa_dev_adpit_label'
    # lmdb_out_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/feat_label/lmdb_foa_dev_data_adpit_label_len20s'
    npy_to_lmdb2(npy_data_dir, lmdb_out_dir, segment_length)
    debug_read_lmdb2(lmdb_out_dir)'''
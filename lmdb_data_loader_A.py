import os
import pdb
import numpy as np
import lmdb
import joblib

import torch
from torch.utils.data import Dataset, DataLoader
#from visual_src.visual_tools import VisualTools

from utils.lmdb_tools.datum_pb2 import SimpleDatum

class LmdbDataset(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        #self.visial_tools = VisualTools()
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            #pdb.set_trace()
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)

            wav_name = datum.wave_name.decode()
            if self.segment_len is not None and label.shape[0] < self.segment_len:
                data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
            if self.data_process_fn is not None:
                data, label = self.data_process_fn(data, label)

            #print('feat {}'.format(data.shape))
            #print('label {}'.format(label.shape))
            #print('wavname {}'.format(wav_name))
        return {'data': data, 'label':label, 'wav_name':wav_name}


    def collater(self, samples):
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names

        return out
    
class LmdbDataset_seddoa_sedsde(Dataset):
    def __init__(self, lmdb_dir, split, seddoa_normalized_features_wts_file=None, sedsde_normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        #self.visial_tools = VisualTools()
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.seddoa_spec_scaler = None
        self.sedsde_spec_scaler = None
        if seddoa_normalized_features_wts_file is not None:
            self.seddoa_spec_scaler = joblib.load(seddoa_normalized_features_wts_file)
        if sedsde_normalized_features_wts_file is not None:
            self.sedsde_spec_scaler = joblib.load(sedsde_normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.seddoa_spec_scaler is not None:
                seddoa_data = self.seddoa_spec_scaler.transform(data)
            if self.sedsde_spec_scaler is not None:
                sedsde_data = self.sedsde_spec_scaler.transform(data)
            #pdb.set_trace()
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)

            wav_name = datum.wave_name.decode()
            if self.segment_len is not None and label.shape[0] < self.segment_len:
                seddoa_data = np.pad(seddoa_data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                sedsde_data = np.pad(sedsde_data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
            if self.data_process_fn is not None:
                seddoa_data, label1 = self.data_process_fn(seddoa_data, label)
                sedsde_data, label2 = self.data_process_fn(sedsde_data, label)

            #print('feat {}'.format(data.shape))
            #print('label {}'.format(label.shape))
            #print('wavname {}'.format(wav_name))
        return {'seddoa_data': seddoa_data, 'sedsde_data':sedsde_data, 'label':label1, 'wav_name':wav_name}


    def collater(self, samples):
        seddoa_feats = [s['seddoa_data'] for s in samples]
        sedsde_feats = [s['sedsde_data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        seddoa_collated_feats = np.stack(seddoa_feats, axis=0)
        sedsde_collated_feats = np.stack(sedsde_feats, axis=0)
        collated_labels = np.stack(labels, axis=0)

        out = {}
        out['input_seddoa'] = torch.from_numpy(seddoa_collated_feats)
        out['input_sedsde'] = torch.from_numpy(sedsde_collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names

        return out

class LmdbDataset2(Dataset):
    def __init__(self, lmdb_dir, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        #self.visial_tools = VisualTools()
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                self.keys.append(k.strip())
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)

            wav_name = datum.wave_name.decode()
            #if self.segment_len is not None and data.shape[0] < self.segment_len*5:
            if self.segment_len is not None:
                data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
            if self.data_process_fn is not None:
                data = self.data_process_fn(data)

            #print('feat {}'.format(data.shape))
            #print('wavname {}'.format(wav_name))
        return {'data': data, 'wav_name':wav_name}


    def collater(self, samples):
        feats = [s['data'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        collated_feats = np.stack(feats, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['wav_names'] = wav_names

        return out

class LmdbDataset_10ms(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            if self.segment_len is not None and label.shape[0] < self.segment_len:
                data = np.pad(data, pad_width=((0,self.segment_len*10-data.shape[0]), (0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
            if self.data_process_fn is not None:
                data, label = self.data_process_fn(data, label)
            wav_name = datum.wave_name.decode()
        return {'data': data, 'label':label, 'wav_name':wav_name}

    def collater(self, samples):
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names

        return out

class LmdbDataset_Finetune_ssast(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            if self.segment_len is not None and data.shape[0] < self.segment_len:
                data = np.pad(data, pad_width=((0,self.segment_len-data.shape[0]), (0,0)))
            if self.data_process_fn is not None:
                data, label = self.data_process_fn(data, label)
            wav_name = datum.wave_name.decode()
        return {'data': data, 'label':label, 'wav_name':wav_name}

    def collater(self, samples):
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names

        return out

class LmdbDataset_eval(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                self.keys.append(k.strip())
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            if self.data_process_fn is not None:
                data = self.data_process_fn(data)
            wav_name = datum.wave_name.decode()
        return {'data': data, 'wav_name':wav_name}

    def collater(self, samples):
        feats = [s['data'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        collated_feats = np.stack(feats, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['wav_names'] = wav_names

        return out

class LmdbDataset_fbank_wav(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            label_frame = label.shape[0]
            fea_frame = label_frame * 5
            feat = data[:fea_frame*448].reshape(-1, 448)
            wav = data[fea_frame*448:].reshape(-1, 4)
            if wav.shape[0] > 480000:
                wav = wav[:480000, :]
            if self.spec_scaler is not None:
                feat = self.spec_scaler.transform(feat)
            if self.segment_len is not None and label.shape[0] < self.segment_len:
                # print(self.segment_len*2400-wav.shape[0])
                wav = np.pad(wav, pad_width=((0,self.segment_len*2400-wav.shape[0]), (0,0)))
                feat = np.pad(feat, pad_width=((0,self.segment_len*5-feat.shape[0]), (0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
            if self.data_process_fn is not None:
                feat, label = self.data_process_fn(feat, label)
            wav_name = datum.wave_name.decode()
        return {'data':feat, 'wave':wav, 'label':label, 'wav_name':wav_name}

    def collater(self, samples):
        feats = [s['data'] for s in samples]
        waves = [s['wave'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        collated_feats = np.stack(feats, axis=0)
        collated_waves = np.stack(waves, axis=0)
        collated_labels = np.stack(labels, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['wave'] = torch.from_numpy(collated_waves)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names

        return out

class LmdbDataset_ssast(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            data_ssast = data[:,448:]
            data_ssast = data_ssast.reshape(data_ssast.shape[0]*2, -1)
            data = data[:,:448]
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            if self.segment_len is not None and label.shape[0] < self.segment_len:
                data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                data_ssast = np.pad(data_ssast, pad_width=((0,self.segment_len*10-data_ssast.shape[0]), (0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
            '''
            if data_ssast.shape[0] <= data.shape[0]:
                data_ssast2 = np.zeros((data.shape[0], 768))
                data_ssast2[:data_ssast.shape[0],:] = data_ssast
                for tt in range(data_ssast.shape[0],data.shape[0]):
                    data_ssast2[tt, :] = data_ssast[-1,:]
                data_ssast = data_ssast2
            '''
            if self.data_process_fn is not None:
                data, label = self.data_process_fn(data, label)
            wav_name = datum.wave_name.decode()
        return {'data': data, 'data_ssast': data_ssast, 'label':label, 'wav_name':wav_name}

    def collater(self, samples):
        feats = [s['data'] for s in samples]
        feats2 = [s['data_ssast'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        collated_feats = np.stack(feats, axis=0)
        collated_feats2 = np.stack(feats2, axis=0)
        collated_labels = np.stack(labels, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['input2'] = torch.from_numpy(collated_feats2)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names

        return out


class LmdbDataset_w2v(LmdbDataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file, ignore, segment_len, data_process_fn) -> None:
        super().__init__(lmdb_dir, split, normalized_features_wts_file=normalized_features_wts_file, ignore=ignore, segment_len=segment_len, data_process_fn=data_process_fn)
        self.w2v_feat_dir = '/yrfs4/sppro/hxwu2/DCASE/dcase_2022_task3/data_ori/wav2vec2_xlsr2_feat'
    
    def __getitem__(self, index):
        is_pad = False
        txn = self.env.begin()
        with txn.cursor() as cursor:
            k = self.keys[index].strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            data_frame_num = data.shape[0]
            if self.spec_scaler is not None:
                data = self.spec_scaler.transform(data)
            label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            if self.segment_len is not None and label.shape[0] < self.segment_len:
                data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
                is_pad = True
            if self.data_process_fn is not None:
                data, label = self.data_process_fn(data, label)
            wav_name = datum.wave_name.decode()
        
        items = wav_name.split('_')
        w2v_feat_name = '_'.join(items[:3])
        w2v_feat = np.load(os.path.join(self.w2v_feat_dir, w2v_feat_name+'.npy'))
        segment_num = int(items[-2])
        segment_ind = int(items[-1])
        w2v_feat_segment = w2v_feat[segment_ind*1000:segment_ind*1000+data_frame_num]
        if is_pad:
            w2v_feat_segment = np.pad(w2v_feat_segment, pad_width=((0,self.segment_len*5-w2v_feat_segment.shape[0]), (0,0)))
            
        return {'data': data, 'label':label, 'wav_name':wav_name, 'w2v_feat':w2v_feat_segment}

    def collater(self, samples):
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]
        w2v_feats = [s['w2v_feat'] for s in samples]

        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)
        collated_w2v_feats = np.stack(w2v_feats, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names
        out['w2v_feat'] = torch.from_numpy(collated_w2v_feats)

        return out


if __name__ == "__main__":
    import time
    import tqdm
    from utils.accdoa import process_foa_input_accdoa_labels
    # lmdb_dir = '/work1/sppro/hxwu2/DCASE/data/data_aug/ACS/lmdb_foa_dev_data_label'
    # norm_file = '/yrfs1/intern/qingwang28/DCASE2022/data/data_aug/ACS/feat_label/foa_wts'
    #lmdb_dir = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/feat_label/lmdb_foa_dev_data_label_len10s/'
    lmdb_dir = '/disk3/yxdong/Dcase2023/Data/data_aug/TestSet/lmdb_foa_dev_data_label_test_len10s'
    #norm_file = '/yrfs1/intern/yajiang/Data/Dcase2023Task3/data/feat_label/foa_wts'
    norm_file = '/disk3/yxdong/Dcase2023/Data/data_aug/TestSet/feat_label/foa_wts'
    split = [1,2,3]
    ignore = None
    dataset = LmdbDataset(lmdb_dir, split, normalized_features_wts_file=norm_file, ignore=ignore, segment_len=100, data_process_fn=process_foa_input_accdoa_labels)
    dataloader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, 
        num_workers=1, collate_fn=dataset.collater
    )
    count = 0
    start_time = time.time()
    for data in tqdm.tqdm(dataloader):
        count += 1
        #pdb.set_trace()
        # print(data['input'].shape, data['target'].shape)
        # print(data['wav_names'])
        # print(data['w2v_feat'].shape)
    print(count, time.time() - start_time)

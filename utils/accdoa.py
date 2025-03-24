import numpy as np
import torch
import torch.nn as nn
import os

def process_foa_input_accdoa_labels(feat, label):
    nb_classes = 13
    mel_bins = 64
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    mask = label[:, :nb_classes]
    mask = np.tile(mask, 3)
    label = mask * label[:, nb_classes:]
    return feat, label

def process_foa_input(feat):
    nb_classes = 13
    mel_bins = 64
    nb_ch = 7
    feat = feat.reshape(feat.shape[0], nb_ch, mel_bins)
    feat = np.transpose(feat, (1, 0, 2))
    return feat

class AccdoaResult():
    def __init__(self, segment_length) -> None:
        self.segment_length = segment_length
        self.output_dict = {}

    def get_sed_doa(self, accdoa_in, nb_classes=13):
        x, y, z = accdoa_in[:, :nb_classes], accdoa_in[:, nb_classes:2*nb_classes], accdoa_in[:, 2*nb_classes:]
        sed = np.sqrt(x**2 + y**2 + z**2) > 0.5
        return sed, accdoa_in
    
    def add_item(self, wav_name, seq_result):
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3])
        start_frame = int(items[-1]) * self.segment_length
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        sed_pred, doa_pred = self.get_sed_doa(seq_result)
        for frame_cnt in range(sed_pred.shape[0]):
            output_dict_frame_cnt = frame_cnt + start_frame
            for class_cnt in range(sed_pred.shape[1]):
                if sed_pred[frame_cnt][class_cnt]>0.5:
                    if output_dict_frame_cnt not in self.output_dict[csv_name]:
                        self.output_dict[csv_name][output_dict_frame_cnt] = []
                    self.output_dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])
    
    def add_items(self, wav_names, net_output):
        accdoa = net_output
        if isinstance(accdoa, torch.Tensor):
            accdoa = accdoa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, accdoa[b])

    def get_result(self):
        return self.output_dict

class AccdoaResult_hop():
    def __init__(self, segment_length, hoplen=10) -> None:
        self.segment_length = segment_length
        self.output_dict = {}
        self.dict = {}
        self.hoplen = hoplen * 10

    def get_sed_doa(self, accdoa_in, nb_classes=13):
        x, y, z = accdoa_in[:, :nb_classes], accdoa_in[:, nb_classes:2*nb_classes], accdoa_in[:, 2*nb_classes:]
        sed = np.sqrt(x**2 + y**2 + z**2) 
        return sed, accdoa_in

    def add_item(self, wav_name, seq_result):
        # print(seq_result.shape)
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3]) 
        # start_frame = int(items[-1]) * self.hoplen
        seg_cnt = int(items[-1])
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        self.output_dict[csv_name][seg_cnt] = seq_result
        
    def add_items(self, wav_names, net_output):
        accdoa = net_output
        if isinstance(accdoa, torch.Tensor):
            accdoa = accdoa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, accdoa[b])

    def _process(self):
        tmp = {}
        enddic = {}
        for k, v in self.output_dict.items():
            seg_cnt = self.output_dict[k].keys()
            max_cnt = max(seg_cnt)
            # print(max_cnt)
            endframe = max_cnt*self.hoplen + self.segment_length
            # print(endframe)
            tmp[k] = torch.zeros(endframe, 39)
            divide = torch.zeros(endframe, 39)

            for segid, val in v.items():
                # print(val.shape)
                # print('+++++++++++++++++')
                startframe = segid * self.hoplen
                if val.shape[0] != self.segment_length:
                    divide[startframe:startframe+val.shape[0], :] += 1
                    tmp[k][startframe:startframe+val.shape[0], :] += val
                    endframe = startframe+val.shape[0]
                else:
                    divide[startframe:startframe+self.segment_length, :] += 1
                    tmp[k][startframe:startframe+self.segment_length, :] += val
            tmp[k] /= divide
            enddic[k] = endframe
        self.output_dict = tmp
        # print(k, endframe)
        return enddic

    def calres(self):
        enddic = self._process()
        threshold = np.array((0.6,  0.6,  0.45, 0.3,  0.55, 0.3,  0.55, 0.3,  0.3,  0.55, 0.3,  0.55, 0.6))
        outdir = '/yrfs1/intern/qingwang28/DCASE2022/model_ensemble/Resnet-Conformer-ACCDOA-new-hop_output'
        for csv_name, val in self.output_dict.items():
            np.save(os.path.join(outdir, '{}.npy'.format(csv_name)), val)
            endframe = enddic[csv_name]
            if csv_name not in self.dict:
                self.dict[csv_name] = {}
            print(csv_name, val.shape, endframe)
            sed_pred, doa_pred = self.get_sed_doa(val)
            print(sed_pred.shape, doa_pred.shape)
            print('+++++++++++++++++')
            for frame_cnt in range(endframe):
                # items = wav_name.split('_')
                # csv_name = '_'.join(items[:-3])  
                output_dict_frame_cnt = frame_cnt
                # print(sed_pred.shape)
                for class_cnt in range(sed_pred.shape[1]):
                    # print(sed_pred[frame_cnt][class_cnt])
                    if sed_pred[frame_cnt][class_cnt]>0.5:
                    #if sed_pred[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.dict[csv_name]:
                            self.dict[csv_name][output_dict_frame_cnt] = []
                        self.dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])

    def get_result(self):
        return self.dict

class AccdoaResult_hop_thres():
    def __init__(self, segment_length, threshold, hoplen=10) -> None:
        self.segment_length = segment_length
        self.output_dict = {}
        self.dict = {}
        self.hoplen = hoplen * 10
        self.threshold = threshold

    def get_sed_doa(self, accdoa_in, nb_classes=13):
        x, y, z = accdoa_in[:, :nb_classes], accdoa_in[:, nb_classes:2*nb_classes], accdoa_in[:, 2*nb_classes:]
        sed = np.sqrt(x**2 + y**2 + z**2) 
        return sed, accdoa_in

    def add_item(self, wav_name, seq_result):
        # print(seq_result.shape)
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3]) 
        # start_frame = int(items[-1]) * self.hoplen
        seg_cnt = int(items[-1])
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        self.output_dict[csv_name][seg_cnt] = seq_result
        
    def add_items(self, wav_names, net_output):
        accdoa = net_output
        if isinstance(accdoa, torch.Tensor):
            accdoa = accdoa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, accdoa[b])

    def _process(self):
        tmp = {}
        enddic = {}
        for k, v in self.output_dict.items():
            seg_cnt = self.output_dict[k].keys()
            max_cnt = max(seg_cnt)
            # print(max_cnt)
            endframe = max_cnt*self.hoplen + self.segment_length
            # print(endframe)
            tmp[k] = torch.zeros(endframe, 39)
            divide = torch.zeros(endframe, 39)

            for segid, val in v.items():
                # print(val.shape)
                # print('+++++++++++++++++')
                startframe = segid * self.hoplen
                if val.shape[0] != self.segment_length:
                    divide[startframe:startframe+val.shape[0], :] += 1
                    tmp[k][startframe:startframe+val.shape[0], :] += val
                    endframe = startframe+val.shape[0]
                else:
                    divide[startframe:startframe+self.segment_length, :] += 1
                    tmp[k][startframe:startframe+self.segment_length, :] += val
            tmp[k] /= divide
            enddic[k] = endframe
        self.output_dict = tmp
        # print(k, endframe)
        return enddic

    def calres(self):
        enddic = self._process()
        for csv_name, val in self.output_dict.items():
            endframe = enddic[csv_name]
            if csv_name not in self.dict:
                self.dict[csv_name] = {}
            print(csv_name, val.shape, endframe)
            sed_pred, doa_pred = self.get_sed_doa(val)
            print(sed_pred.shape, doa_pred.shape)
            print('+++++++++++++++++')
            for frame_cnt in range(endframe):
                # items = wav_name.split('_')
                # csv_name = '_'.join(items[:-3])  
                output_dict_frame_cnt = frame_cnt
                # print(sed_pred.shape)
                for class_cnt in range(sed_pred.shape[1]):
                    # print(sed_pred[frame_cnt][class_cnt])
                    if sed_pred[frame_cnt][class_cnt]>self.threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.dict[csv_name]:
                            self.dict[csv_name][output_dict_frame_cnt] = []
                        self.dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])

    def get_result(self):
        return self.dict

class AccdoaResult_hop_thre():
    def __init__(self, segment_length, hoplen=1, model='') -> None:
        self.segment_length = segment_length
        self.output_dict = {}
        self.dict = {}
        self.hoplen = hoplen * 10
        self.model = model
        if self.model != '':
            os.system('mkdir -p model_output/{}'.format(model))
        self.nppath = 'model_output/{}/'.format(model)


    def get_sed_doa(self, accdoa_in, nb_classes=13):
        x, y, z = accdoa_in[:, :nb_classes], accdoa_in[:, nb_classes:2*nb_classes], accdoa_in[:, 2*nb_classes:]
        sed = np.sqrt(x**2 + y**2 + z**2) 
        return sed, accdoa_in

    def add_item(self, wav_name, seq_result):
        # if self.model != '':
        #     npname = self.nppath + wav_name + '.npy'
        #     np.save(npname, seq_result)
        #     print('save {}'.format(npname))
        items = wav_name.split('_')
        csv_name = '_'.join(items[:-3]) 
        # start_frame = int(items[-1]) * self.hoplen
        seg_cnt = int(items[-1])
        if csv_name not in self.output_dict:
            self.output_dict[csv_name] = {}
        self.output_dict[csv_name][seg_cnt] = seq_result

        
    def add_items(self, wav_names, net_output):
        accdoa = net_output
        if isinstance(accdoa, torch.Tensor):
            accdoa = accdoa.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            self.add_item(wav_name, accdoa[b])

    def _process(self):
        tmp = {}
        enddic = {}
        for k, v in self.output_dict.items():
            seg_cnt = self.output_dict[k].keys()
            max_cnt = max(seg_cnt)
            # print(max_cnt)
            endframe = max_cnt*self.hoplen + self.segment_length
            # print(endframe)
            tmp[k] = torch.zeros(endframe, 39)
            divide = torch.zeros(endframe, 39)

            for segid, val in v.items():
                # print(val.shape)
                # print('+++++++++++++++++')
                startframe = segid * self.hoplen
                if val.shape[0] != self.segment_length:
                    divide[startframe:startframe+val.shape[0], :] += 1
                    tmp[k][startframe:startframe+val.shape[0], :] += val
                    endframe = startframe+val.shape[0]
                else:
                    divide[startframe:startframe+self.segment_length, :] += 1
                    tmp[k][startframe:startframe+self.segment_length, :] += val
            tmp[k] /= divide
            enddic[k] = endframe
        # self.output_dict = tmp
        return enddic, tmp

    def calres(self, the=[0.5]*13):
        self.dict = {}
        threshold = the
        enddic, tmp = self._process()
        for csv_name, val in tmp.items():
            endframe = enddic[csv_name]
            if csv_name not in self.dict:
                self.dict[csv_name] = {}
            # print(csv_name, val.shape, endframe)
            sed_pred, doa_pred = self.get_sed_doa(val)
            # print(sed_pred.shape, doa_pred.shape)
            # print('+++++++++++++++++')
            for frame_cnt in range(endframe):
                # items = wav_name.split('_')
                # csv_name = '_'.join(items[:-3]) 
                output_dict_frame_cnt = frame_cnt
                for class_cnt in range(sed_pred.shape[1]):
                    if sed_pred[frame_cnt][class_cnt]>threshold[class_cnt]:
                        if output_dict_frame_cnt not in self.dict[csv_name]:
                            self.dict[csv_name][output_dict_frame_cnt] = []
                        self.dict[csv_name][output_dict_frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+2*13]])

    def get_result(self):
        return self.dict


class MSELoss_mix(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, output, target):
#        a = time.time()
        target_m = target[16:]
        target_sed_mix = torch.ones((target_m.shape[0],target.shape[1],13))
        target_sed_mix = target_sed_mix.cuda()
        t_x,t_y,t_z = target[:16,:,:13],target[:16,:,13:26],target[:16,:,26:39]
        t_xm,t_ym,t_zm = target_m[:,:,:13],target_m[:,:,13:26],target_m[:,:,26:39]
#        c = time.time()
        target_sed = torch.sqrt(t_x**2+t_y**2+t_z**2)
        target_sed_m = torch.sqrt(t_xm**2+t_ym**2+t_zm**2) 
        target_mix0 = (target_sed+target_sed_m)<0.5
#        target_mix1 = (0.5<(target_sed+target_sed_m) and (target_sed+target_sed_m)<1.5)
        target_mix2 = (target_sed+target_sed_m)>1.5
        target_sed_mix = target_sed_mix.masked_fill(target_mix0,0)
#        target_sed_mix.masked_fill(target_mix1,1) 
        target_sed_mix = target_sed_mix.masked_fill(target_mix2,0) 
        target_sed_mix =  target_sed_mix.repeat(1,1,3)
        target_mix = target[:16] + target_sed_mix*target_m
#        pdb.set_trace()
#        c = time.time()
#        print(c-a,'sqrt time')
#        for i in range(target.shape[0]):
#            for j in range(target.shape[1]):
#                for k in range(13):
                      
                    #target_np = target.detach().cpu().numpy()
#                    x,y,z = target[i][j][k],target[i][j][k+13],target[i][j][k+26]
#                    x = x.detach().cpu().numpy()
#                    y = y.detach().cpu().numpy()
#                    z = z.detach().cpu().numpy()
#                    sed = torch.sqrt(x**2+y**2+z**2)
#                    x,y,z = target_m[i][j][k],target_m[i][j][k+13],target_m[i][j][k+26]
                   # x = x.detach().cpu().numpy()
                   # y = y.detach().cpu().numpy()
                   # z = z.detach().cpu().numpy()
#                    sed_m = torch.sqrt(x**2+y**2+z**2)
#                    if target_sed[i][j][k] > 0 and target_sed_m[i][j][k] >0:
#                        print('yyy')
#                        exit()
#                        target_mix[i][j][k] = target[i][j][k]
#                        target_mix[i][j][k+13] = target[i][j][k+13]                        
#                        target_mix[i][j][k+26] = target[i][j][k+26]
#                    else:
#                        target_mix[i][j][k] = target[i][j][k]+target[i][j][k]
#                        target_mix[i][j][k] = target[i][j][k+13]+target[i][j][k+13]
#                        target_mix[i][j][k] = target[i][j][k+26]+target[i][j][k+26]        #

#        b = time.time()
#        print(b-a,'train_time')
        loss = self.criterion(output, target_mix)
        return loss


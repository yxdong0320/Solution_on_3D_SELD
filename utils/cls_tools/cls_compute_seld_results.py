import os
import pdb
import numpy as np

from .SELD_evaluation_metrics_old import SELDMetrics
from .cls_feature_class import FeatureClass
from .parameters import get_params


class ComputeSELDResults(object):
    def __init__(
            self, ref_files_folder=None, use_polar_format=True
    ):
        params = get_params()
        self._use_polar_format = use_polar_format
        self._desc_dir = ref_files_folder if ref_files_folder is not None else os.path.join(params['dataset_dir'], 'metadata_dev')
        self._doa_thresh = params['lad_doa_thresh']

        # Load feature class
        self._feat_cls = FeatureClass(params)
        
        # collect reference files
        self._ref_labels = {}
        # for split in os.listdir(self._desc_dir):
        #     for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
        #         # Load reference description file
        #         gt_dict = self._feat_cls.load_output_format_file(os.path.join(self._desc_dir, split, ref_file), cm2m=True)  # TODO: Reconsider the cm2m conversion
        #         gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
        #         nb_ref_frames = max(list(gt_dict.keys()))
        #         if self.segment_level:
        #             self._ref_labels[ref_file] = [self._feat_cls.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]
        #         else:
        #             self._ref_labels[ref_file] = [self._feat_cls.organize_labels(gt_dict, nb_ref_frames), nb_ref_frames]
        for split in os.listdir(self._desc_dir):
            if os.path.isdir(os.path.join(self._desc_dir, split)):
                for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
                    # Load reference description file
                    gt_dict = self._feat_cls.load_output_format_file(os.path.join(self._desc_dir, split, ref_file))
                    if not self._use_polar_format:
                        gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
                    nb_ref_frames = max(list(gt_dict.keys()))
                    self._ref_labels[ref_file] = [self._feat_cls.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]
            else:
                gt_dict = self._feat_cls.load_output_format_file(os.path.join(self._desc_dir, split))
                if not self._use_polar_format:
                    gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
                nb_ref_frames = max(list(gt_dict.keys()))
                self._ref_labels[split] = [self._feat_cls.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['average']

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        eval = SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            _output_format_file = os.path.join(pred_files_path, pred_file)
            pred_dict = {}
            _fid = open(_output_format_file, 'r')
            for _line in _fid:
                _words = _line.strip().split(',')
                _frame_ind = int(_words[0])
                if _frame_ind not in pred_dict:
                    pred_dict[_frame_ind] = []
                pred_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
            _fid.close()
            if self._use_polar_format:
                pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
            pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])

            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])

        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        return ER, F, LE, LR, seld_scr, classwise_results

def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])

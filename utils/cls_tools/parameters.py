# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    # print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs
    
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights = '', 

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir = '...',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir = '...',

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        # max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # SPATIAL MAP
        spatial_map=False,
        #gaussian_data='/home/cv6/hxwu2/MyProjects/DCASE/seld-dcase2022-main/gaussian/sigma_1.npy',

        # DNN MODEL PARAMETERS
        label_sequence_length=200,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_rnn_layers=2,
        rnn_size=128,        # RNN contents, length of list = number of layers, list value = number of nodes

        self_attn=False,
        nb_heads=4,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,              # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average = 'macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,
        evaluate_distance = True,
        segment_based_metrics = False,
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics
    )

    # ########### User defined parameters ##############
    if argv == '1':
        pass
        # print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        # print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        # print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        # print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        # print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        # print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        # print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '8':
        # print("MIC + SALSA + ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False
        params['nb_cnn2d_filt']=128
        params['nb_rnn_layers']=2
    elif argv == '9':
        # print("MIC + SALSA + ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False
        params['nb_cnn2d_filt']=128
        params['nb_rnn_layers']=3

    elif argv == '10':
        # print("MIC + SALSA + multi ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['nb_cnn2d_filt']=128
        params['nb_rnn_layers']=2
    elif argv == '11':
        # print("MIC + SALSA + multi ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['nb_cnn2d_filt']=128
        params['nb_rnn_layers']=2
    elif argv == '12':
        # print("MIC + SALSA + multi ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['nb_cnn2d_filt']=256
        params['nb_rnn_layers']=2
    elif argv == '13':
        # print("FOA + spatial map\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False
        params['spatial_map'] = True
    elif argv == '999':
        # print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        # print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    params['unique_classes'] = 13

    return params

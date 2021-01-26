import pickle 
class HyperParameters:
    hidden_size = 18
    lstm_layers = 1
    dropout = 0.1
    output_size = 7
    n_targets = 1
    # loss = QuantileLoss()
    attention_head_size = 1
    max_encoder_length = 30
    static_categoricals = ['market_id']
    static_reals = ['step', 'encoder_length', 'nrn_center', 'nrn_scale']
    time_varying_categoricals_encoder = []
    time_varying_categoricals_decoder = []
    # categorical_groups=dataset.variable_groups
    categorical_groups = {}
    # time_varying_reals_encoder=[name for name in dataset.time_varying_known_reals if name in allowed_encoder_known_variable_names]
    #       + dataset.time_varying_unknown_reals
    time_varying_reals_encoder = ['time_idx', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'ly_n_visitors',
                                'ly_nrn', 'ly_dayofweek_sin', 'ly_dayofweek_cos', 'ly_month_sin', 'ly_month_cos',
                                'relative_time_idx']
    # time_varying_reals_decoder=dataset.time_varying_known_reals,
    time_varying_reals_decoder = ['time_idx', 'dayofweek_sin',
                                'dayofweek_cos', 'month_sin', 'month_cos', 'ly_n_visitors', 'ly_nrn', 'ly_dayofweek_sin',
                                'ly_dayofweek_cos', 'ly_month_sin', 'ly_month_cos', 'relative_time_idx']
    # x_reals=dataset.reals
    x_reals = ['step', 'encoder_length', 'nrn_center', 'nrn_scale', 'time_idx', 'dayofweek_sin',
            'dayofweek_cos', 'month_sin', 'month_cos', 'ly_n_visitors', 'ly_nrn', 'ly_dayofweek_sin',
            'ly_dayofweek_cos', 'ly_month_sin', 'ly_month_cos', 'relative_time_idx']
    # x_categoricals = dataset.flat_categoricals,
    x_categoricals = ['market_id']
    hidden_continuous_size = 8
    hidden_continuous_sizes = {}
    embedding_sizes = {'market_id': [1801, 16]}
    embedding_paddings = []
    learning_rate = 0.03
    log_interval = 10
    log_val_interval = 10
    log_gradient_flow = False
    reduce_on_plateau_patience = 4
    monotone_constaints = {}
    share_single_variable_networks = False
    # logging_metrics = ModuleList((0): SMAPE(), (1): MAE(), (2): RMSE(), (3): MAPE())
    # output_transformer = GroupNormalizer(groups=['group'], transformation='softplus')
    reduce_on_plateau_min_lr = 1e-05
    weight_decay: 0.0
    optimizer = 'ranger'
    # embedding_labels_location = "/home/damianos/Documents/projects/tft_walkthrough/pytorch-forecasting/embedding_labels.pickle"
    embedding_labels_location = "/Users/dchristophides/OneDrive - Expedia Group/brain/renewal/V2/data/embedding_labels.pickle"
    with open(embedding_labels_location, 'rb') as f:
        embedding_labels = pickle.load(f)


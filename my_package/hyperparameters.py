import pickle


class HyperParameters:
    """
    these hyperparameters are created at pytorch_forecasting.models.base_model.BaseModelWithCovariates.from_dataset

        embedding_sizes = {
            name: (len(encoder.classes_), get_embedding_size(len(encoder.classes_)))
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
            * {'market_id': (1801, 100)} -> note that this is changed later on by embedding_size = min(embedding_size, self.max_embedding_size)
                                                 where max_embedding_size = hidden_size = 16 so finally {'market_id': (1801, 100)} -> {'market_id': (1801, 16)}

        allowed_encoder_known_variable_names = dataset.time_varying_known_categoricals + dataset.time_varying_known_reals
            * [] + ['time_idx', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'ly_n_visitors', 'ly_nrn', 'ly_dayofweek_sin', 'ly_dayofweek_cos',
              'ly_month_sin', 'ly_month_cos', 'relative_time_idx']

        embedding_labels = {name: encoder.classes_  for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals}
            * dataset.categorical_encoders.items() =  dict_items([('__group_id__group', NaNLabelEncoder()), ('group', NaNLabelEncoder()), ('market_id', NaNLabelEncoder())])
              dataset.categoricals = ['market_id']
              embedding_labels = {'market_id': {'100068': 0, '100082': 1,  '100086': 2, ...} (note encoder.classes_ is a dictionary)

        embedding_paddings = dataset.dropout_categoricals = []

        static_categoricals = dataset.static_categoricals = ['market_id']

        time_varying_categoricals_encoder =[name for name in dataset.time_varying_known_categoricals if name in allowed_encoder_known_variable_names] + dataset.time_varying_unknown_categoricals
            * dataset.time_varying_known_categoricals = []
              dataset.time_varying_unknown_categoricals = []

        time_varying_categoricals_decoder = dataset.time_varying_known_categoricals = []

        static_reals = dataset.static_reals = ['step', 'encoder_length', 'nrn_center', 'nrn_scale']

        time_varying_reals_encoder=[name for name in dataset.time_varying_known_reals if name in allowed_encoder_known_variable_names] + dataset.time_varying_unknown_reals
            * dataset.time_varying_known_reals = ['time_idx', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'ly_n_visitors', 'ly_nrn', 'ly_dayofweek_sin',
                     'ly_dayofweek_cos', 'ly_month_sin', 'ly_month_cos', 'relative_time_idx']
              dataset.time_varying_unknown_reals = []
              time_varying_reals_encoder = ['time_idx', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'ly_n_visitors', 'ly_nrn', 'ly_dayofweek_sin', 'ly_dayofweek_cos',
                 'ly_month_sin', 'ly_month_cos', 'relative_time_idx']

        time_varying_reals_decoder = dataset.time_varying_known_reals
            * time_varying_reals_decoder = ['time_idx', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'ly_n_visitors', 'ly_nrn', 'ly_dayofweek_sin', 'ly_dayofweek_cos',
               'ly_month_sin', 'ly_month_cos', 'relative_time_idx']

        x_reals=dataset.reals
            * ['step', 'encoder_length', 'nrn_center', 'nrn_scale', 'time_idx', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'ly_n_visitors',
               'ly_nrn', 'ly_dayofweek_sin', 'ly_dayofweek_cos', 'ly_month_sin', 'ly_month_cos', 'relative_time_idx']

        x_categoricals = dataset.flat_categoricals = ['market_id']

        categorical_groups=dataset.variable_groups = {}

        used provided in timeseries initialization:
            - static_categoricals
            - time_varying_known_categoricals
            - time_varying_known_reals
            - time_varying_unknown_categoricals
            - time_varying_unknown_reals

        calculated:
            - categoricals
                static_categoricals + time_varying_known_categoricals + time_varying_unknown_categoricals
            - dropout_categoricals -> can be passed in init -> default = []
            - reals = static_reals + time_varying_known_reals + time_varying_unknown_reals
            - flat_categoricals same as categoricals
            - variable_groups grouped variables
    """

    def __init__(self, static_categoricals, time_varying_known_categoricals, time_varying_known_reals, time_varying_unknown_categoricals,
                 time_varying_unknown_reals, static_reals, embedding_sizes, hidden_continuous_sizes=None, embedding_paddings=None, dropout_categoricals=None,
                 variable_groups=None,
                 hidden_size=18, lstm_layers=1, dropout=0.1, output_size=7, n_targets=1, attention_head_size=1, max_encoder_length=30,
                 share_single_variable_networks=False, hidden_continuous_size=8):
        if hidden_continuous_sizes is None:
            hidden_continuous_sizes = {}
        if embedding_paddings is None:
            embedding_paddings = []
        if variable_groups is None:
            variable_groups = {}
        if dropout_categoricals is None:
            dropout_categoricals = []
        self.static_categoricals = static_categoricals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.categoricals = static_categoricals + time_varying_known_categoricals + time_varying_unknown_categoricals
        self.dropout_categoricals = dropout_categoricals
        self.static_reals = static_reals
        self.reals = static_reals + time_varying_known_reals + time_varying_unknown_reals
        self.flat_categoricals = self.categoricals
        self.variable_groups = variable_groups
        self.embedding_paddings = embedding_paddings
        self.embedding_sizes = embedding_sizes
        self.hidden_continuous_sizes = hidden_continuous_sizes
        self.share_single_variable_networks = share_single_variable_networks
        # embedding_labels_location = "/home/damianos/Documents/projects/tft_walkthrough/pytorch-forecasting/embedding_labels.pickle"
        embedding_labels_location = "/Users/dchristophides/OneDrive - Expedia Group/brain/renewal/V2/data/embedding_labels.pickle"  # OK
        with open(embedding_labels_location, 'rb') as f:
            self.embedding_labels = pickle.load(f)

        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.output_size = output_size
        self.n_targets = n_targets
        self.attention_head_size = attention_head_size
        self.max_encoder_length = max_encoder_length
        self.hidden_continuous_size = hidden_continuous_size

        self.allowed_encoder_known_variable_names = self.time_varying_known_categoricals + self.time_varying_known_reals
        self.x_categoricals = self.flat_categoricals = ['market_id']
        self.x_reals = self.reals
        self.time_varying_reals_decoder = self.time_varying_known_reals
        self.time_varying_reals_encoder = [name for name in self.time_varying_known_reals if
                                           name in self.allowed_encoder_known_variable_names] + self.time_varying_unknown_reals
        self.categorical_groups = self.variable_groups
        self.time_varying_categoricals_decoder = self.time_varying_known_categoricals
        self.time_varying_categoricals_encoder = [name for name in self.time_varying_known_categoricals if
                                                  name in self.allowed_encoder_known_variable_names] + self.time_varying_unknown_categoricals
        a = 1

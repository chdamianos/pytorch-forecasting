from torch._C import device
from data import TimeSeriesDataSet , GroupNormalizer
import pandas as pd
import gc
from rnn import LSTM
from sub_modules import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)
from nn import MultiEmbedding
import torch
from typing import List, Union, Dict, Tuple
import torch.nn as nn
import numpy as np
from hyperparameters import HyperParameters


def QuantileLoss(y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    # calculate quantile loss
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - y_pred[..., i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
    losses = torch.cat(losses, dim=2)

    lengths = torch.full((target.size(0),), fill_value=target.size(1),
                         dtype=torch.long, device=target.device)

    losses = losses / losses.size(-1)
    return losses.sum() / lengths.sum()


class Utils:
    @staticmethod
    def decoder_variables(hyperparams: HyperParameters) -> List[str]:
        """List of all decoder variables in model (excluding static variables)"""
        return hyperparams.time_varying_categoricals_decoder + hyperparams.time_varying_reals_decoder

    @staticmethod
    def encoder_variables(hyperparams: HyperParameters) -> List[str]:
        """List of all encoder variables in model (excluding static variables)"""
        return hyperparams.time_varying_categoricals_encoder + hyperparams.time_varying_reals_encoder

    @staticmethod
    def static_variables(hyperparams: HyperParameters) -> List[str]:
        """List of all static variables in model"""
        return hyperparams.static_categoricals + hyperparams.static_reals

    @staticmethod
    def reals(hyperparams: HyperParameters) -> List[str]:
        """List of all continuous variables in model"""
        return list(
            dict.fromkeys(
                hyperparams.static_reals
                + hyperparams.time_varying_reals_encoder
                + hyperparams.time_varying_reals_decoder
            )
        )

    @staticmethod
    def create_mask(size: int, lengths: torch.LongTensor, inverse: bool = False) -> torch.BoolTensor:
        """
        Create boolean masks of shape len(lenghts) x size.

        An entry at (i, j) is True if lengths[i] > j.

        Args:
            size (int): size of second dimension
            lengths (torch.LongTensor): tensor of lengths
            inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.

        Returns:
            torch.BoolTensor: mask
        """
        if inverse:  # return where values are
            return torch.arange(size, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(-1)
        else:  # return where no values are
            return torch.arange(size, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)

    @staticmethod
    def get_attention_mask(encoder_lengths: torch.LongTensor, decoder_length: int, device):
        """
        Returns causal mask to apply for self-attention layer.

        Args:
            self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = Utils.create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask

    @staticmethod
    def expand_static_context(context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        device,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        output_size: Union[int, List[int]] = 7,
        n_targets: int = 1,
        loss=None,
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        Temporal Fusion Transformer for forecasting timeseries - use its :py:meth:`~from_dataset` method if possible.

        Implementation of the article
        `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
        Forecasting <https://arxiv.org/pdf/1912.09363.pdf>`_. The network outperforms DeepAR by Amazon by 36-69%
        in benchmarks.

        Enhancements compared to the original implementation (apart from capabilities added through base model
        such as monotone constraints):

        * static variables can be continuous
        * multiple categorical variables can be summarized with an EmbeddingBag
        * variable encoder and decoder length by sample
        * categorical embeddings are not transformed by variable selection network (because it is a redundant operation)
        * variable dimension in variable selection network are scaled up via linear interpolation to reduce
          number of parameters
        * non-linear variable processing in variable selection network can be shared among decoder and encoder
          (not shared by default)

        Tune its hyperparameters with
        :py:func:`~pytorch_forecasting.models.temporal_fusion_transformer.tuning.optimize_hyperparameters`.

        Args:

            hidden_size: hidden size of network which is its main hyperparameter and can range from 8 to 512
            lstm_layers: number of LSTM layers (2 is mostly optimal)
            dropout: dropout rate
            output_size: number of outputs (e.g. number of quantiles for QuantileLoss and one target or list
                of output sizes).
            n_targets: number of targets. Defaults to 1.
            loss: loss function taking prediction and targets
            attention_head_size: number of attention heads (4 is a good default)
            max_encoder_length: length to encode (can be far longer than the decoder length but does not have to be)
            static_categoricals: names of static categorical variables
            static_reals: names of static continuous variables
            time_varying_categoricals_encoder: names of categorical variables for encoder
            time_varying_categoricals_decoder: names of categorical variables for decoder
            time_varying_reals_encoder: names of continuous variables for encoder
            time_varying_reals_decoder: names of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
                (fallback to hidden_continuous_size if index is not in dictionary)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
                , will log multiple entries per batch. Defaults to -1.
            log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
                variables mapping
                position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
                larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
                This constraint significantly slows down training. Defaults to {}.
            share_single_variable_networks (bool): if to share the single variable networks between the encoder and
                decoder. Defaults to False.
            logging_metrics (nn.ModuleList[LightningMetric]): list of metrics that are logged during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]).
            **kwargs: additional arguments to :py:class:`~BaseModel`.
        """
        super().__init__()
        self.device = device
        if loss is None:
            loss = QuantileLoss
        # processing inputs
        # embeddings
        """
        used to be 
        self.input_embeddings =
        MultiEmbedding(
        (embeddings): ModuleDict(
            (market_id): Embedding(1801, 16)))
        """
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=HyperParameters.embedding_sizes,
            categorical_groups=HyperParameters.categorical_groups,
            embedding_paddings=HyperParameters.embedding_paddings,
            x_categoricals=HyperParameters.x_categoricals,
            max_embedding_size=HyperParameters.hidden_size,
        )

        # continuous variable processing for every real, even if static
        # hidden_continuous_sizes -> custom/specific embedding size 
        # hidden_continuous_size -> default 
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, HyperParameters.hidden_continuous_sizes.get(name, HyperParameters.hidden_continuous_size))
                for name in Utils.reals(HyperParameters)
            }
        )

        # variable selection for static variables
        static_input_sizes = {name: HyperParameters.embedding_sizes[name][1]
                              for name in HyperParameters.static_categoricals}
        static_input_sizes.update(
            {
                name: HyperParameters.hidden_continuous_sizes.get(name, HyperParameters.hidden_continuous_size)
                for name in HyperParameters.static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=HyperParameters.hidden_size,
            input_embedding_flags={name: True for name in HyperParameters.static_categoricals},
            dropout=HyperParameters.dropout,
            prescalers=self.prescalers,
        )
        # variable selection for encoder
        encoder_input_sizes = {
            name: HyperParameters.embedding_sizes[name][1] for name in HyperParameters.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: HyperParameters.hidden_continuous_sizes.get(name, HyperParameters.hidden_continuous_size)
                for name in HyperParameters.time_varying_reals_encoder
            }
        )
        # variable selection for decoder
        decoder_input_sizes = {
            name: HyperParameters.embedding_sizes[name][1] for name in HyperParameters.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: HyperParameters.hidden_continuous_sizes.get(name, HyperParameters.hidden_continuous_size)
                for name in HyperParameters.time_varying_reals_decoder
            }
        )
        # create single variable grns that are shared across decoder and encoder
        if HyperParameters.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, HyperParameters.hidden_size),
                    HyperParameters.hidden_size,
                    HyperParameters.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, HyperParameters.hidden_size),
                        HyperParameters.hidden_size,
                        HyperParameters.dropout,
                    )
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=HyperParameters.hidden_size,
            input_embedding_flags={name: True for name in HyperParameters.time_varying_categoricals_encoder},
            dropout=HyperParameters.dropout,
            context_size=HyperParameters.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}            
            if not HyperParameters.share_single_variable_networks
            else self.shared_single_variable_grns,
        )
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=HyperParameters.hidden_size,
            input_embedding_flags={name: True for name in HyperParameters.time_varying_categoricals_decoder},
            dropout=HyperParameters.dropout,
            context_size=HyperParameters.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not HyperParameters.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=HyperParameters.hidden_size,
            hidden_size=HyperParameters.hidden_size,
            output_size=HyperParameters.hidden_size,
            dropout=HyperParameters.dropout,
        )
        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=HyperParameters.hidden_size,
            hidden_size=HyperParameters.hidden_size,
            output_size=HyperParameters.hidden_size,
            dropout=HyperParameters.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=HyperParameters.hidden_size,
            hidden_size=HyperParameters.hidden_size,
            output_size=HyperParameters.hidden_size,
            dropout=HyperParameters.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            HyperParameters.hidden_size, HyperParameters.hidden_size, HyperParameters.hidden_size, HyperParameters.dropout
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = LSTM(
            input_size=HyperParameters.hidden_size,
            hidden_size=HyperParameters.hidden_size,
            num_layers=HyperParameters.lstm_layers,
            dropout=HyperParameters.dropout if HyperParameters.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=HyperParameters.hidden_size,
            hidden_size=HyperParameters.hidden_size,
            num_layers=HyperParameters.lstm_layers,
            dropout=HyperParameters.dropout if HyperParameters.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(HyperParameters.hidden_size,
                                                      dropout=HyperParameters.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        self.post_lstm_add_norm_encoder = AddNorm(HyperParameters.hidden_size, trainable_add=False)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=HyperParameters.hidden_size,
            hidden_size=HyperParameters.hidden_size,
            output_size=HyperParameters.hidden_size,
            dropout=HyperParameters.dropout,
            context_size=HyperParameters.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=HyperParameters.hidden_size,
            n_head=HyperParameters.attention_head_size,
            dropout=HyperParameters.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(
            HyperParameters.hidden_size,
            dropout=HyperParameters.dropout,
            trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            HyperParameters.hidden_size,
            HyperParameters.hidden_size,
            HyperParameters.hidden_size,
            dropout=HyperParameters.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(HyperParameters.hidden_size, dropout=None,
                                                trainable_add=False)

        if HyperParameters.n_targets > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [nn.Linear(HyperParameters.hidden_size, output_size) for output_size in HyperParameters.output_size]
            )
        else:
            self.output_layer = nn.Linear(HyperParameters.hidden_size, HyperParameters.output_size)



    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        x.keys() -> ['encoder_cat', 'encoder_cont', 'encoder_target', 'encoder_lengths', 'decoder_cat', 'decoder_cont', 'decoder_target', 'decoder_lengths', 'decoder_time_idx', 'groups', 'target_scale']
        x['encoder_cat'].shape -> [batch, encoder_length, number_of_cat_features] e.g. [1, 30, 1]
        x['encoder_cont'].shape -> [1, 30, 16] -> real features normalized ['step', 'encoder_length', 'nrn_center', 'nrn_scale', 'time_idx', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'ly_n_visitors',
            'ly_nrn', 'ly_dayofweek_sin', 'ly_dayofweek_cos', 'ly_month_sin', 'ly_month_cos', 'relative_time_idx']
        x['encoder_target'].shape ->  [1, 30] -> target in the encoder part (e.g. nrn) -> e.g. tensor([[12.,  5.,  9.,  8.,  3.,  8.,  9., 10., 10.,  7.,  5.,  2.,  7.,  5.,
          5.,  6.,  3.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.]]) looks like un-normalized
        x['encoder_lengths'] -> tensor([30])
        x['decoder_cat'].shape -> batch, encoder_length, 1] e.g. [1, 45, 1], same as encoder_cat but for the length of the decoder e.g. tensor([[[89],[89],...]])
        x['decoder_cont'].shape -> [1, 45, 16] -> same as encoder_cont but for the length of the decoder
        x['decoder_target'].shape -> torch.Size([1, 45])
        x['decoder_lengths'] -> tensor([45])
        x['decoder_time_idx'] -> torch.Size([1, 45]) -> tensor([[30, 31, ..., 74]])
        x['groups'] -> tensor([[0]])
        x['target_scale'] -> tensor([[4.1467, 4.8231]])
        """

        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        # x_cat -> torch.Size([1, 75, 1])
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        # x_cont -> torch.Size([1, 75, 16])
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        # timesteps
        timesteps = x_cont.size(1)  # encode + decode length
        # max_encoder_length
        max_encoder_length = int(encoder_lengths.max())
        # input_vectors['market_id'].shape torch.Size([1, 75, 16])
        """
        uses nn.Embedding() it's 16 not 100 because of a variable `max_embedding_size` (probably based on hidden_size) 
        """
        # todo fix
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(HyperParameters.x_reals)
                if name in Utils.reals(HyperParameters)
            }
        )

        # Embedding and variable selection
        if len(Utils.static_variables(HyperParameters)) > 0:
            static_embedding = {name: input_vectors[name][:, 0] for name in Utils.static_variables(HyperParameters)}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), HyperParameters.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)
        static_context_variable_selection = Utils.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in Utils.encoder_variables(HyperParameters)
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in Utils.decoder_variables(HyperParameters)  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            HyperParameters.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(HyperParameters.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, Utils.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=Utils.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_length=timesteps - max_encoder_length, device=self.device
            ),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if HyperParameters.n_targets > 1:  # if to use multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)

        return dict(
            prediction=output,
            attention=attn_output_weights,
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
            groups=x["groups"],
            decoder_time_idx=x["decoder_time_idx"],
            target_scale=x["target_scale"],
        )


def preprocess(pdf: pd.DataFrame):
    pdf = pdf.drop(["step_raw", "demand_date"], axis=1)
    pdf = pdf.rename(columns={"time_index": "time_idx"})
    pdf = pdf.assign(**{'market_id': pdf['market_id'].astype("str")})
    pdf = pdf.fillna(0)
    return pdf


data_location = "/home/damianos/Documents/projects/tft_walkthrough/pytorch-forecasting/fe2_small_sample_sdf.parquet"
data_pdf = pd.read_parquet(data_location)
data_pdf = data_pdf.query("nrn!=-999")
ratio_train = 0.7
train_length = 30
test_length = 45

demand_dates_list = list(set(data_pdf['demand_date'].to_list()))
demand_pydates_list = [i.to_pydatetime() for i in demand_dates_list]
demand_pydates_list.sort()
middleIndex = int((len(demand_pydates_list) - 1) * ratio_train)
middle_date = demand_pydates_list[middleIndex]

train_data_pdf = preprocess(data_pdf[data_pdf['demand_date'] <= middle_date])
test_data_pdf = preprocess(data_pdf[data_pdf['demand_date'] > middle_date])

del data_pdf
gc.collect()
training = TimeSeriesDataSet(
    train_data_pdf,
    time_idx="time_idx",
    target="nrn",
    group_ids=["group"],
    min_encoder_length=train_length,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=train_length,
    min_prediction_length=test_length,
    max_prediction_length=test_length,
    static_categoricals=["market_id"],
    static_reals=["step"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos", "ly_n_visitors", "ly_nrn", "ly_dayofweek_sin",
                              "ly_dayofweek_cos", "ly_month_sin", "ly_month_cos"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[],
    target_normalizer=GroupNormalizer(
        groups=["group"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

prediction = TimeSeriesDataSet(
    test_data_pdf,
    time_idx="time_idx",
    target="nrn",
    group_ids=["group"],
    min_encoder_length=train_length,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=train_length,
    min_prediction_length=test_length,
    max_prediction_length=test_length,
    static_categoricals=["market_id"],
    static_reals=["step"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos", "ly_n_visitors", "ly_nrn", "ly_dayofweek_sin",
                              "ly_dayofweek_cos", "ly_month_sin", "ly_month_cos"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[],
    target_normalizer=GroupNormalizer(
        groups=["group"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    predict_mode=True
)

# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=1, num_workers=0)
val_dataloader = prediction.to_dataloader(train=False, batch_size=1, num_workers=0)


for batch in train_dataloader:
    x, y = batch
    y = y[0]
    break

model = TemporalFusionTransformer(device=torch.device('cpu'))
out=model(x)
a = 1

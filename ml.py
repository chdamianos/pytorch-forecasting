import numba
import gc
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


def preprocess(pdf: pd.DataFrame):
    pdf = pdf.drop(["step_raw", "demand_date"], axis=1)
    pdf = pdf.rename(columns={"time_index": "time_idx"})
    pdf = pdf.assign(**{'market_id': pdf['market_id'].astype("str")})
    pdf = pdf.fillna(0)
    return pdf


# data_location = "/home/damianos/Documents/projects/tft_walkthrough/pytorch-forecasting/fe2_small_sample_sdf.parquet"
data_location = "/Users/dchristophides/OneDrive - Expedia Group/brain/renewal/V2/data/fe2_small_sample_sdf/fe2_small_sample_sdf.parquet"
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
# del data_sdf
gc.collect()
# (c) filling missing values and/or (d) optionally adding a variable indicating filled values
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
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = prediction.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=1,  # 10
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=1,  # 30 coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

# fit network
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)
tft.predict(val_dataloader)
a = 1

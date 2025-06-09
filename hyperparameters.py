from collections import namedtuple
import utils

testcase = 2
do_rolling_window = False

if testcase == 1:
    # small testcase
    start_year, start_month = 2018, 10
    end_year, end_month = 2018, 12
    split_dict = {
        "train_start": 20181019,
        "train_end": 20181119,
        "validate_start": 20181120,
        "validate_end": 20181131,
        "test_start": 20181201,
        "test_end": 20181231,
    }
elif testcase == 2:
    # full training testcase
    start_year, start_month = 2018, 10
    end_year, end_month = 2021, 12
    split_dict = {
        "train_start": 20181019,
        "train_end": 20201018,
        "validate_start": 20201019,
        "validate_end": 20211018,
        "test_start": 20211019,
        "test_end": 20211231,
    }
elif testcase == 3:
    # full testing testcase
    start_year, start_month = 2021, 10
    end_year, end_month = 2023, 10
    split_dict = {
        "train_start": 20211001,
        "train_end": 20211015,
        "validate_start": 20211016,
        "validate_end": 20211018,
        "test_start": 20211019,
        "test_end": 20231018,
    }
elif testcase == 4:
    # full testcase
    start_year, start_month = 2018, 10
    end_year, end_month = 2023, 10
    split_dict = {
        "train_start": 20181019,
        "train_end": 20201018,
        "validate_start": 20201019,
        "validate_end": 20211018,
        "test_start": 20211019,
        "test_end": 20231018,
    }
elif testcase == 5:
    # full tmp testing testcase
    start_year, start_month = 2021, 10
    end_year, end_month = 2023, 10
    split_dict = {
        "train_start": 20211001,
        "train_end": 20211015,
        "validate_start": 20211016,
        "validate_end": 20211018,
        "test_start": 20211019,
        "test_end": 20231018,
    }
else:
    raise ValueError("wrong testcase number!!!")


PathRoot = r"C:\Users\nycu_dev1\Desktop\YEN-HSU Code\\"
preprocess_data_path = PathRoot + "preprocess_data/" + f'preprocess_data_{start_year}_{start_month:02d}_to_{end_year}_{end_month:02d}.pickle'
current_time = utils.get_time()

# Market parameters
Tax = 0.0000278
open_delete = 16
close_delete = 0
formation_period = 150
trading_period = 224



# model parameters
batch_size = 256
lr = 1e-4
norm_clip = 1e-3
optim = "Adam"


# training parameters
num_epochs = 1000
early_stop_count = 50
save_freq = 1
device = "cuda"
num_workers = 0
loss_set = ["GaussCopGumLoss", "GaussCopNormLoss", "GumIndLoss", "NormIndLoss"]

loss_type = loss_set[0]
if loss_type in ["GaussCopGumLoss", "GaussCopNormLoss"]:
    do_copula_model = True
else:
    do_copula_model = False

# model save path
save_root = f"{PathRoot}Train_ckpt/"
# pre-train model path
model_path = None#r'train_ckpt\2024-08-04 19-44-41\BestEpoch=12_ValLoss=4.5620.pth'#None
# train, or test
mode = "train"







# trading behavior
must_open = False
open_prob_threshold = 0.65
do_dynamic_sl = True
if do_dynamic_sl is True:
    sl_offset = None
    sl_sigma = 0.999
else:
    sl_offset = 100000
    sl_sigma = None

do_discard_expect_prof_threshold = False
if do_discard_expect_prof_threshold:
    expect_prof_threshold = 0.5
else:
    expect_prof_threshold = None

use_adj_expect_prof = False

# constant threshold
do_constant_output = False

# Global parameters:
# 2016~201710
global_paras = dict()
global_paras["rtop"] = [2.4693460030466663, 1.0608914863005185]
global_paras["top"] = [3.70266662086769, 1.6990610342437076]
global_paras["close"] = [3.0172715942390242, 3.0833261860426373]
global_paras["cormat"] = [0.1790868, -0.10757489, 0.75402701]

arg_define = namedtuple("arguments",
                 'batch_size, lr, norm_clip, num_epochs, optim, early_stop_count, save_root, '
                 'device, save_freq, num_workers, do_constant_output, global_paras, must_open,'
                 'Loss, sl_offset, sl_sigma, model_path, mode, do_discard_expect_prof_threshold, expect_prof_threshold,'
                 'do_dynamic_sl, use_adj_expect_prof, do_copula_model, open_prob_threshold')
args = arg_define(batch_size, lr, norm_clip, num_epochs, optim, early_stop_count, save_root,
           device, save_freq, num_workers, do_constant_output, global_paras, must_open,
           loss_type, sl_offset, sl_sigma, model_path, mode, do_discard_expect_prof_threshold, expect_prof_threshold,
           do_dynamic_sl, use_adj_expect_prof, do_copula_model, open_prob_threshold)
split_info_define = namedtuple("arguments",
                 'split_dict, start_year, start_month, end_year, end_month')
split_info = split_info_define(split_dict, start_year, start_month, end_year, end_month)


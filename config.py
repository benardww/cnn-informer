import torch

DATA_ROOT = "autodl-tmp/chb-mit-scalp-eeg-database-1.0.0"

ALL_PATIENTS = [
    "chb01", "chb02", "chb03", "chb04", "chb05",
    "chb07", "chb08", "chb09", "chb10",
    "chb11", "chb12", "chb13", "chb14", "chb15",
    "chb17", "chb18", "chb19", "chb20", "chb21",
    "chb22", "chb23", "chb24",
]  # 22 patients; chb06 and chb16 excluded

CHANNELS = [
    "FP1-F7", "F7-T7",  "T7-P7",  "P7-O1",
    "FP1-F3", "F3-C3",  "C3-P3",  "P3-O1",
    "FP2-F4", "F4-C4",  "C4-P4",  "P4-O2",
    "FP2-F8", "F8-T8",  "T8-P8",  "P8-O2",
    "FZ-CZ",  "CZ-PZ",
]

SFREQ              = 256
WINDOW_SAMPLES     = 1024   # 4 s × 256 Hz
SEIZURE_STRIDE     = 512    # 50% overlap for seizure segments
NON_SEIZURE_STRIDE = 1024   # no overlap for non-seizure segments
WAVELET            = "db4"
DWT_LEVEL          = 5

D_MODEL         = 64
N_HEADS         = 8
INFORMER_LAYERS = 3
D_FF            = 256
PROB_FACTOR     = 3      # c: u = c * ceil(ln(L))
DROPOUT         = 0.3

BATCH_SIZE  = 32
LR          = 1e-4
EPOCHS      = 50
TRAIN_RATIO = 0.8   # chronological split within each patient

MAF_WINDOW = 10    # 10 × 4 s windows = 40 s smoothing
THRESHOLD  = 0.5
COLLAR_K   = 5.0   # seconds, default; can be overridden per patient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

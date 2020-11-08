# Database
MONGODB_SERVER = '127.0.0.1'
MONGODB_PORT   =  27017
MONGODB_DB     = 'nlc' # neural_loop_combiner

MONGO_USERNAME = ''
MONGO_PASSWORD = ''

MONGODB_TRACK_COL   = 'tracks'
MONGODB_LOOP_COL    = 'loops'
MONGODB_TAG_COL     = 'tags'
MONGODB_DATASET_COL = 'datasets'
MONGODB_MODEL_COL   = 'models'

# Directory
INT_DIR = 'files/inputs' # put tracks you want to extract here
OUT_DIR = 'files/outputs'

# Others
DUR   = 2
SR    = 44100
CACHE = True 
LOG   = True

# Threshold
HASH_TYPE         = 'ahash'
HASH_THRESHOLD    = 5
EXISTED_THRESHOLD = 0.2

# Datasets
TEST_SIZE   = 100
SPLIT_RATIO = 0.8
NG_TYPES    = {
    'shift'    : 1,
    'reverse'  : 1,
    'rearrange': 1,
    'random'   : 1,
    'selected' : 1
}


# Models (default settings)
LR           = 0.001
MARGIN       = 2
EPOCHS       = 2
BATCH_SIZE   = 128
LOG_INTERVAL = 10 
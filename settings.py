import pathlib
from datetime import datetime

INPUT_DIR = pathlib.Path.cwd().joinpath("input")
OUTPUT_DIR = pathlib.Path.cwd().joinpath("output")
FILETYPE_EXT = '.csv'
INPUT_FILETYPES = ['.tsv', '.csv']
DELIMITER = '\t'

TIMESTAMP = int(datetime.utcnow().timestamp())

IDENTIFIER_COLUMN = 'name'
REQUEST_COLUMN = 'request'
COHORT_COLUMN_STEM = 'cohort'
TARGET_COLUMN_STEM = 'target'
COLUMN_JOIN_CHAR = '-'

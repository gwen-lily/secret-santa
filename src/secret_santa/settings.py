"""Settings for secret-santa.

###############################################################################
# package:  secret-santa                                                      #
# website:  github.com/noahgill409/secret-santa                               #
# email:    noahgill409@gmail.com                                             #
###############################################################################

"""

import pathlib
from datetime import datetime

INPUT_DIR = pathlib.Path.cwd().joinpath("input")
OUTPUT_DIR = pathlib.Path.cwd().joinpath("output")
FILETYPE_EXT = '.csv'
INPUT_FILETYPES = ['.tsv', '.csv']
CSV_SEP = '\t'
MULT_ENTRY_SEP = r'|'

GIFTS = 2

TIMESTAMP = int(datetime.utcnow().timestamp())

NAME_COLUMN = 'name'
REQUEST_COLUMN = 'request'
COHORT_COLUMN = 'cohort'
TARGET_COLUMN = 'target'
FORCED_GIFTEE_COLUMN = 'forced_giftee'
INVALID_GIFTEE_COLUMN = 'invalid_giftee'
COLUMN_JOIN_CHAR = '-'

NO_ID = -1

import pathlib
from random import sample
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse
import pandas as pd
import unicodedata
import re
import settings


class Anon:
    def __init__(self, user_id: int, gift_limit: int = 1, identifier: str = None, request: str = None):
        self.user_id = user_id
        self.gift_limit = gift_limit
        self._identifier = identifier if identifier else user_id
        self._request = request if request else ""

        self._targets = []
        self._targeted_by = []

    def get_identifier(self):
        return str(self._identifier)

    def get_request(self):
        return self._request

    def get_targets(self):
        return self._targets

    def get_targeted_by(self):
        return self._targeted_by

    def target_count(self) -> int:
        m = len(self._targets)
        n = len(set(self._targets))
        assert m == n
        return m

    def targeted_by_count(self) -> int:
        m = len(self._targeted_by)
        n = len(set(self._targeted_by))
        assert m == n
        return m

    def targets_needed(self) -> int:
        assert self.can_pick()
        return self.gift_limit - self.target_count()

    def can_pick(self) -> bool:
        return self.target_count() < self.gift_limit

    def can_be_picked(self) -> bool:
        return self.targeted_by_count() < self.gift_limit

    def pick(self, other):
        assert other not in self._targets and isinstance(other, Anon)
        self._targets.append(other)
        other.get_picked_by(self)

    def get_picked_by(self, other):
        self._targeted_by.append(other)

    def targets(self, other):
        assert isinstance(other, Anon)
        return other in self.get_targets()

    def is_targeted_by(self, other):
        assert isinstance(other, Anon)
        return self in other.get_targets()

    def is_cohort(self, other):
        assert isinstance(other, Anon)
        return any(ti == tj for ti in self.get_targets() for tj in other.get_targets()) and not self == other

    def __str__(self):
        s = "Anon (id: {:})".format(self.user_id)
        return s

    def __eq__(self, other):
        return self.user_id == other.user_id

    def __hash__(self):
        return hash(self.user_id)

    def __lt__(self, other):
        return self.user_id < other.user_id


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def make_assignments(args, df):
    while True:
        try:
            anons = []
            for i in df.index.values:
                try:
                    identifier = df.loc[i, settings.IDENTIFIER_COLUMN]
                except KeyError as exc:
                    identifier = None

                try:
                    request = df.loc[i, settings.REQUEST_COLUMN]
                except KeyError as exc:
                    request = None

                anons.append(Anon(
                    user_id=i,
                    gift_limit=args.gifts,
                    identifier=identifier,
                    request=request
                ))

            for anon in anons:
                if not anon.can_pick():
                    continue

                potential_targets = set(a for a in anons if a.can_be_picked() and not a.user_id == anon.user_id)
                targets = sample(potential_targets, anon.targets_needed())

                for t in targets:
                    anon.pick(t)

                    if args.verbose:
                        s = "{:} >>> {:}".format(anon.user_id, t.user_id)
                        print(s)

            return anons

        # Can occur with multiple gift assignments per person, creates impossible situations to resolve, just try again!
        except ValueError:
            print('Unsolvable, dumping previous assignments and restarting...')
            continue


def target_index(n: int):
    t_stem = settings.TARGET_COLUMN_STEM
    jc = settings.COLUMN_JOIN_CHAR

    return jc.join([t_stem, str(n)])


def request_index(n: int):
    r_stem = settings.REQUEST_COLUMN
    jc = settings.COLUMN_JOIN_CHAR

    return jc.join([r_stem, str(n)])


def cohort_index(m: int, n: int):
    c_stem = settings.COHORT_COLUMN_STEM
    jc = settings.COLUMN_JOIN_CHAR

    return jc.join([target_index(m), c_stem, str(n)])


def main(args, df):
    assigned_anons = make_assignments(args, df)
    headers = df.columns.to_list()

    for i in range(1, args.gifts+1):
        headers.extend([target_index(i), request_index(i)])

        for j in range(1, args.gifts):
            headers.append(cohort_index(i, j))

    df.reindex(columns=headers)

    for a in assigned_anons:
        targets = a.get_targets()

        for i, t in enumerate(targets):
            t_col = target_index(i+1)
            r_col = request_index(i+1)
            df.at[a.user_id, t_col] = t.get_identifier()
            df.at[a.user_id, r_col] = t.get_request()
            cohorts = set(x for x in assigned_anons if x.targets(t) and not a == x)

            for j, c in enumerate(cohorts):
                c_col = cohort_index(i+1, j+1)
                df.at[a.user_id, c_col] = c.get_identifier()

    if args.log:
        if not settings.OUTPUT_DIR.exists():
            settings.OUTPUT_DIR.mkdir()

        dir_stem = slugify('-'.join([str(settings.TIMESTAMP), args.label]))
        master_filename = slugify(dir_stem) + settings.FILETYPE_EXT
        simulation_dir = settings.OUTPUT_DIR.joinpath(dir_stem)
        simulation_dir.mkdir()
        master_fp = simulation_dir.joinpath(master_filename)

        with open(master_fp, 'w') as f:
            df.to_csv(f, sep=settings.DELIMITER)

        for a in assigned_anons:
            anon_slice = df.loc[a.user_id]
            anon_filename = slugify(a.get_identifier()) + settings.FILETYPE_EXT
            anon_fp = simulation_dir.joinpath(anon_filename)

            with open(anon_fp, 'w') as f:
                anon_slice.to_csv(f, sep=settings.DELIMITER)

    if args.graph:
        plt.axes()

        # arrow properties
        margin = 1.1
        width = 0.01
        head_width = 2.5*width
        head_length = 2*width
        shape = 'left'
        length_includes_head = False
        squeeze = 0.1
        center_offset = 0.01
        basis_0 = np.asarray([0 + squeeze, 0, 1])   # basis start and endpoints to transform later, in the z = 1 plane
        basis_1 = np.asarray([1 - squeeze, 0, 1])   #
        plt.xlim(-margin, margin)
        plt.ylim(-margin, margin)

        n = len(assigned_anons)
        THETA = np.linspace(0, 2*math.pi, n + 1)[:-1]   # last value is equal to the zeroth, remove it
        X0 = np.cos(THETA)
        Y0 = np.sin(THETA)
        anon_coords = {}
        anon_ids = [a.get_identifier() for a in assigned_anons]

        for anon in assigned_anons:
            x0 = X0[anon.user_id]
            y0 = Y0[anon.user_id]
            aid = anon.get_identifier()
            plt.text(x0, y0, aid, horizontalalignment='center', verticalalignment='center')

            for targ in anon.get_targets():
                x1 = X0[targ.user_id]
                y1 = Y0[targ.user_id]
                dx = x1 - x0
                dy = y1 - y0

                try:
                    slope = dy / dx
                except ZeroDivisionError as exc:
                    if dy > 0:
                        slope = np.inf
                    elif dy < 0:
                        slope = -1 * np.inf
                    else:
                        raise exc

                theta = np.arctan(slope)    # angle associated with line
                phi = theta - np.pi/4       # angle associated with normal offset

                x_offset = center_offset * np.cos(phi)
                y_offset = center_offset * np.sin(phi)

                ROTATE = np.asarray([[np.cos(theta),    np.sin(theta),  0],
                                     [-np.sin(theta),   np.cos(theta),  0],
                                     [0,                0,              1]])

                base_length = math.sqrt(dx**2 + dy**2)
                SCALE = np.asarray([[base_length,   0,              0],
                                    [0,             base_length,    0],
                                    [0,             0,              1]])

                TRANSLATION = np.asarray([[1,               0,              0],
                                          [0,               1,              0],
                                          [x0 + x_offset,   y0 + y_offset,  1]])

                TRANSFORMATION = np.matmul(np.matmul(ROTATE, SCALE), TRANSLATION)   # full transformation in one matrix

                x0_prime, y0_prime = np.matmul(basis_0, TRANSFORMATION)[:-1]     # Project back to 2D plane
                x1_prime, y1_prime = np.matmul(basis_1, TRANSFORMATION)[:-1]     #

                dx_prime = x1_prime - x0_prime
                dy_prime = y1_prime - y0_prime

                plt.arrow(x0_prime, y0_prime, dx_prime, dy_prime, head_width=head_width, head_length=head_length,
                          width=width, shape=shape, length_includes_head=length_includes_head)

        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''Bedevere's blind secret santa assignment program! Here's how to use it.''',
        epilog='''Example: python ss.py /path/to/file.csv -gifts 2 --verbose --graph --log''',
    )
    parser.add_argument('file',
                        type=str,
                        help='The delimited file with each person delimited by tabs and column headers given.')
    parser.add_argument('-gifts',
                        type=int, default=1,
                        help='How many gifts does each participant give & receive? Default is 1.')
    parser.add_argument('-label', '-l',
                        type=str,
                        help='A label to append to logs.')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Write the assignments to the console.')
    parser.add_argument('--graph',
                        action='store_true',
                        help='Display the assignments graphically.')
    parser.add_argument('--log',
                        action='store_true',
                        help="Write assignments to a sub-directory.")
    input_args = parser.parse_args()

    try:
        participants_filepath = pathlib.Path(input_args.file)
        assert participants_filepath.exists()
        assert participants_filepath.suffix in settings.INPUT_FILETYPES

    except AssertionError as E:
        print(''.join(['The file: ', input_args.file, ' does not exist in this directory or it\'s not a csv']))
        raise E

    input_df = pd.read_csv(filepath_or_buffer=participants_filepath, sep=settings.DELIMITER)

    main(input_args, input_df)

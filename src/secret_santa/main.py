"""Secret Santa with a wicked twist. You won't wanna miss this Harjmas.

###############################################################################
# package:  secret-santa                                                      #
# website:  github.com/noahgill409/secret-santa                               #
# email:    noahgill409@gmail.com                                             #
###############################################################################

"""

from __future__ import annotations
from dataclasses import dataclass, field
from argparse import ArgumentParser
from unicodedata import normalize

import logging
import re
import random
import pandas as pd

from secret_santa import settings


logging.basicConfig()
stream_logger = logging.StreamHandler()

###############################################################################
# exceptions                                                                  #
###############################################################################


class SecretSantaError(Exception):
    """Base package exception from which others inherit."""


class AssignmentError(SecretSantaError):
    """Raised if assignment fails, as in the case of a logical inconsistency.
    """

###############################################################################
# main classes                                                                #
###############################################################################


@dataclass
class Gift:
    """A gift which describes the giver and receiver of a transaction.

    Attributes
    ----------
    giver : Person
        The person giving the gift (ie the gifter)
    receiver : Person
        The person receiving the gift (ie the giftee)
    """
    giver: Person = field(default_factory=None)
    receiver: Person = field(default_factory=None)

    def anonymous_str(self) -> str:
        """Return an anonymous representation of gift.

        An anonymous representation only shows transactions between unique id
        numbers, not names.

        Returns
        -------
        str
        """
        giv = self.giver
        rec = self.receiver
        assert isinstance(giv.id, int)
        assert isinstance(rec.id, int)
        _s = f"{self.__class__.__name__}({giv.id} -> {rec.id})"
        return _s

    def __str__(self) -> str:
        _s = f'{self.__class__.__name__}({self.giver} -> {self.receiver})'
        return _s


@ dataclass(kw_only=True)
class Person:
    """A person that has identifying information and a log of transactions.

    Attributes
    ----------
    name : str
        The person's name, or unique identifier.
    id : int
        The person's unique id.
    request : str
        The person's request for secret santa, defaults to empty string.
    gifts_given : list[Gift]
        A list of the gifts a person has given.
    gifts_received : list[Gift]
        A list of the gifts a person has received.
    """
    name: str
    id: int = field(default=settings.NO_ID)
    request: str = field(default_factory=str)
    gifts_given: list[Gift] = field(default_factory=list)
    gifts_received: list[Gift] = field(default_factory=list)

    def assign_gift(self, gift: Gift) -> bool:
        """Read the provided gift and append it to appropriate list.

        The gift is read to determine whether self is the giver or
        receiver.

        Parameters
        ----------
        gift : Gift
            A gift between self and another Person.

        Returns
        -------
        bool
            True if the assignment is successful, False otherwise.
        """
        # always handle from the giver's instance.
        if gift.receiver == self:
            return gift.giver.assign_gift(gift)

        assert self == gift.giver
        other = gift.receiver

        if self.can_give and other.can_receive:
            self.gifts_given.append(gift)
            other.gifts_received.append(gift)
            return True

        return False

    @ property
    def can_give(self) -> bool:
        """Return if the person is able to give more gifts.

        Returns
        -------
        bool
        """
        return len(self.gifts_given) < settings.GIFTS

    @ property
    def can_receive(self) -> bool:
        """Return if the person is able to receive more gifts.

        Returns
        -------
        bool
        """
        return len(self.gifts_received) < settings.GIFTS

    def __str__(self) -> str:
        return self.name


@dataclass
class SpecialPerson(Person):
    """A person who has additional gift assignment constraints.

    Attributes
    ----------
    forced_giftees : list[Person]
        A list of Person objects whom self must give a gift to.
    invalid_giftees : list[Person]
        A list of Person objects whom self cannot give a gift to.
    """
    forced_giftees: list[Person] = field(default_factory=list)
    invalid_giftees: list[Person] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure that SpecialPerson is in some way special."""
        if not (self.forced_giftees or self.invalid_giftees):
            raise TypeError(f"""{self.__class__.__name__} should be
                initialized as {Person.__name__}
                """)

        return

    def assign_gift(self, gift: Gift) -> bool:
        """Read the provided gift and append it to the appropriate list.

        If the provided receiver is in self's invalid giftees list, the
        assignment is stopped.

        Parameters
        ----------
        gift : Gift
            A gift from self to another Person.

        Returns
        -------
        bool
            True if the assignment is successful, otherwise False.
        """
        # always handle from the giver's instance.
        if gift.receiver == self:
            return gift.giver.assign_gift(gift)

        assert self == gift.giver
        other = gift.receiver

        if other in self.invalid_giftees:
            return False

        return super().assign_gift(gift)


@ dataclass
class SecretSantaOrganizer:
    """Performs gift assignment on its people.

    Attributes
    ----------
    people : list[Person]
        A list of Person objects.
    """
    people: list[Person]

    def __post_init__(self) -> None:
        """Ensure people are unique."""
        unique_attributes = ['name', 'id']

        for idx_i, p_i in enumerate(self.people[:]):
            for idx_j, p_j in enumerate(self.people[:]):
                if idx_j <= idx_i:
                    continue

                for _ua in unique_attributes:
                    if p_i.__getattribute__(_ua) == p_j.__getattribute__(_ua):
                        raise ValueError("names & ids must be unique.")

    def assign_gifts(self) -> bool:
        """Peform gift assignment.

        Returns
        -------
        bool
            True if the assignment is successful, otherwise False.
        """
        # pre-set values for special people
        for lad in (p for p in self.people if isinstance(p, SpecialPerson)):
            forced_gifts = [Gift(lad, receiver)
                            for receiver in lad.forced_giftees]

            for gift in forced_gifts:
                gift.giver.assign_gift(gift)

        # actual assignment
        generous_lads = [p for p in self.people if p.can_give]
        while generous_lads:

            for lad in generous_lads:
                receiving_lads = [p for p in self.people
                                  if p.can_receive and p != lad]
                try:
                    choice = random.sample(receiving_lads, 1)[0]
                    lad.assign_gift(Gift(lad, choice))
                except ValueError as exc:
                    raise AssignmentError from exc

            generous_lads = [p for p in self.people if p.can_give]

        return True

    @ classmethod
    def from_csv(cls, filename: str):
        """Read a csv of information on People.

        In order to import any amount of SpecialPerson, special people MUST be
        at the end of the list. This is because they reference other people. If
        SpecialPerson needs to reference another SpecialPerson, I will have to
        re-write this code.

        Parameters
        ----------
        filename : str
            The filename or pathlike object to read.
        """
        df = pd.read_csv(filename, sep=settings.CSV_SEP)
        people: list[Person] = []
        person_count = 0

        for index in df.index.values:
            try:
                name = df.loc[index, settings.NAME_COLUMN]
            except KeyError as exc:
                raise exc

            try:
                request = df.loc[index, settings.REQUEST_COLUMN]
            except KeyError:
                request = ''

            try:
                forced_giftees = df.loc[index, settings.FORCED_GIFTEE_COLUMN]
                forced_giftees = forced_giftees.split(settings.MULT_ENTRY_SEP)
                assert len(forced_giftees) <= settings.GIFTS

            except (KeyError, AttributeError):
                forced_giftees = []

            try:
                invalid_giftees = df.loc[index, settings.INVALID_GIFTEE_COLUMN]
                invalid_giftees = invalid_giftees.split(
                    settings.MULT_ENTRY_SEP)
            except (KeyError, AttributeError):
                invalid_giftees = []

            if forced_giftees or invalid_giftees:

                for fg_idx, forced_giftee in enumerate(forced_giftees[:]):
                    # match by name
                    p = [p for p in people if p.name == forced_giftee]
                    assert len(p) == 1

                    # replace inplace with Person object
                    forced_giftees[fg_idx] = p[0]

                for ig_idx, invalid_giftee in enumerate(invalid_giftees[:]):
                    # match by name
                    p = [p for p in people if p.name == invalid_giftee]
                    assert len(p) == 1

                    # replace inplace with Person object
                    invalid_giftees[ig_idx] = p[0]

                person = SpecialPerson(
                    name=name,
                    id=person_count,
                    request=request,
                    forced_giftees=forced_giftees,
                    invalid_giftees=invalid_giftees
                )

            else:
                person = Person(name=name, id=person_count, request=request)

            people.append(person)
            person_count += 1

        return cls(people)

    def __str__(self) -> str:
        s = f"{self.__class__.__name__}({', '.join(self.people)})"
        return s


###############################################################################
# helper functions & execution                                                #
###############################################################################

def slugify(val, allow_unicode=False) -> str:
    """Sanitize a value and return its string.

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.


    This procedure was Taken from:
    https://github.com/django/django/blob/master/django/utils/text.py

    """
    val = str(val)
    if allow_unicode:
        val = normalize('NFKC', val)
    else:
        val = normalize('NFKD', val)
        val = val.encode('ascii', 'ignore').decode('ascii')

    val = re.sub(r'[^\w\s-]', '', val.lower())
    val = re.sub(r'[-\s]+', '-', val).strip('-_')
    return val


def main():
    """Run secret santa logic."""

    filepath = settings.INPUT_DIR.joinpath(main_args.file)

    while True:
        try:
            organizer = SecretSantaOrganizer.from_csv(filepath)
            organizer.assign_gifts()
            break
        except AssignmentError:
            pass

    out_subdir = settings.OUTPUT_DIR.joinpath(main_args.label)
    out_subdir.mkdir(exist_ok=False)

    for person in organizer.people:
        out_filepath = out_subdir.joinpath(f"{slugify(person.name)}.txt")

        with open(out_filepath, 'w', encoding='UTF-8') as wfp:
            for gift in person.gifts_given:
                cohorts: list[Person] = []
                # add all who share this recipient to a cohorts list
                cohorts.extend([g.giver for g in gift.receiver.gifts_received])
                cohorts.remove(person)

                if cohorts:
                    s = f"You and {', '.join(str(c) for c in cohorts)} " + \
                        f"have {gift.receiver}.\n"
                else:
                    s = f"You have {gift.receiver}.\n"

                if main_args.display_anonymous is True:
                    print(gift.anonymous_str())
                else:
                    print(gift)
                wfp.write(s)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="bedevere's blind secret santa assignment program",
        epilog="""Example: python ./main.py /path/to/file.csv -file
        input/example.csv -gifts 2 -label test-label.
        """
    )

    parser.add_argument('-file', '-f', type=str,
                        help="""The delimited file with each person delimited
                        by tabs and column headers given.
                        """)
    parser.add_argument('-gifts', type=int, default=1,
                        help="""How many gifts does each participant give &
                        receive? Default is 1.
                        """)
    parser.add_argument('-label', '-l', type=str,
                        help="""A label to append to logs.
                        """)
    parser.add_argument('--display-anonymous', action='store_true',
                        help="""Add this flag to print out anonymous data.""")
    main_args = parser.parse_args()

    main()

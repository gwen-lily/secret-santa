import tkinter.filedialog as fd
import os
from random import sample
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import argparse


class Person:

	def __init__(self, name: str, dirname: str, gifts: int, code_length: int):
		self.name = name
		self.filepath = os.path.join(os.getcwd(), dirname, ''.join(['ss_', self.name, '.txt']))
		self.targets = []
		self.targeted = 0
		self.gifts = gifts
		self.code = self.name[:code_length]

	def can_pick(self) -> bool:
		return len(self.targets) < self.gifts

	def can_be_picked(self) -> bool:
		return self.targeted < self.gifts

	def get_picked(self):
		self.targeted += 1


def main(participants: List[str], args: argparse.Namespace):
	now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	dirname = ''.join(['ss_', now])
	log = not args.no_log

	if log and not os.path.isdir(dirname):
		os.makedirs(dirname)

	code_length = 3
	participants_dict = {}

	while True:
		try:
			for p in participants:
				participants_dict[p] = Person(p, dirname, args.gifts, code_length)

			for participant_name in participants:
				P = participants_dict[participant_name]

				if not P.can_pick():
					continue

				targets_needed = P.gifts - len(P.targets)
				potential_targets = set(filter(lambda u: participants_dict[u].can_be_picked(), participants)) - {P.name}
				targets = sample(potential_targets, targets_needed)

				for t in targets:
					P.targets.append(t)
					participants_dict[t].get_picked()

					if args.verbose:
						print(P.name, ">>>", t)

			break

		# Can occur with multiple gift assignments per person, creates impossible situations to resolve, just try again!
		except ValueError:
			continue

	if log:
		for participant_name in participants:
			P = participants_dict[participant_name]

			with open(P.filepath, 'w') as f:
				for t in P.targets:
					cohorts = []

					for cohort_name in participants:

						if not cohort_name == participant_name and t in participants_dict[cohort_name].targets:
							cohorts.append(cohort_name)

					f.write(''.join(['You & ', ', '.join(cohorts), ' have ', t, '\n']))

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

		n = len(participants)
		THETA = np.linspace(0, 2*math.pi, n + 1)[:-1]   # last value is equal to the zeroth, remove it
		X = list(map(lambda t: math.cos(t), THETA))
		Y = list(map(lambda t: math.sin(t), THETA))

		participant_coordinates = {}

		for x, y, pn in zip(X, Y, participants):
			P = participants_dict[pn]
			participant_coordinates[P.name] = (x, y)
			plt.text(x, y, P.code, horizontalalignment='center', verticalalignment='center')

		for participant_name in participants:
			P = participants_dict[participant_name]
			x0, y0 = participant_coordinates[P.name]

			for t in P.targets:
				x1, y1 = participant_coordinates[t]
				dx = x1 - x0
				dy = y1 - y0

				slope = dy / dx
				theta = math.atan(slope) + (dx <= 0)*math.pi    # angle associated with line
				phi = theta - math.pi/4                         # angle associated with normal offset
				x_offset = center_offset * math.cos(phi)
				y_offset = center_offset * math.sin(phi)

				ROTATE = np.asarray([[math.cos(theta),  math.sin(theta),    0],
				                     [-math.sin(theta), math.cos(theta),    0],
				                     [0,                0,                  1]])

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
		epilog='''Example: python ss.py "list of people.txt" 2 --verbose''',
	)
	parser.add_argument('file',
	                    type=str,
	                    help='The file with each person delimited by tabs or newlines.')
	parser.add_argument('-gifts',
	                    type=int, default=1,
	                    help='How many gifts does each participant give & receive? Default is 1.')
	parser.add_argument('--verbose',
	                    action='store_true',
	                    help='Write the assignments to the console.')
	parser.add_argument('--graph',
	                    action='store_true',
	                    help='Display the assignments graphically.')
	parser.add_argument('--no-log',
	                    action='store_true',
	                    help="Used for testing; Don't log assignments to a sub-directory.")
	args = parser.parse_args()

	try:
		assert os.path.exists(args.file)

	except AssertionError as E:
		print(''.join(['The file: ', args.file, ' does not exist in this directory']))
		raise E

	participants = []

	with open(args.file, 'r') as f:
		sanitized_text = f.read().strip().replace('\t', '\n').splitlines()

		for p in sanitized_text:
			participants.append(p)

	main(participants, args)



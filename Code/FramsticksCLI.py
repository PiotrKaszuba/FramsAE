from subprocess import Popen, PIPE, check_output
from enum import Enum
from typing import List
import json
import sys, os
import argparse
import numpy as np


class FramsticksCLI:
	"""Runs Framsticks CLI (command-line) executable and communicates with it using standard input and output.
	You can perform basic operations like mutation, crossover, and evaluation of genotypes.
	This way you can perform evolution controlled by python, access and manipulate genotypes.
	You can even design and use in evolution your own genetic representation implemented entirely in python.

	You need to provide one or two parameters when you run this class: the path to Framsticks CLI
	and the name of the Framsticks CLI executable (if it is non-standard). See::
		FramsticksCLI.py -h"""

	PRINT_FRAMSTICKS_OUTPUT: bool = False  # set to True for debugging
	DETERMINISTIC: bool = False  # set to True to have the same results on each run

	GENO_SAVE_FILE_FORMAT = Enum('GENO_SAVE_FILE_FORMAT', 'NATIVEFRAMS RAWGENO')  # how to save genotypes
	OUTPUT_DIR = "scripts_output"
	STDOUT_ENDOPER_MARKER = "FileObject.write"  # we look for this message on Framsticks CLI stdout to detect when Framsticks created a file with the result we expect

	FILE_PREFIX = 'framspy_'

	RANDOMIZE_CMD = "rnd" + "\n"
	SETEXPEDEF_CMD = "expdef standard-eval" + "\n"
	GETSIMPLEST_CMD = "getsimplest"
	GETSIMPLEST_FILE = FILE_PREFIX + "simplest.gen"
	EVALUATE_CMD = "evaluate eval-allcriteria.sim "
	EVALUATE_FILE = "genos_eval.json"
	CROSSOVER_CMD = "crossover"
	CROSSOVER_FILE = FILE_PREFIX + "child.gen"
	DISSIMIL_CMD = "dissimil"
	DISSIMIL_FILE = FILE_PREFIX + "dissimilarity_matrix.gen"
	ISVALID_CMD = "isvalid"
	ISVALID_FILE = FILE_PREFIX + "validity.gen"
	MUTATE_CMD = "mutate"
	MUTATE_FILE = FILE_PREFIX + "mutant.gen"

	CLI_INPUT_FILE = FILE_PREFIX + "genotypes.gen"


	def __init__(self, framspath, framsexe):
		self.frams_path = framspath
		self.frams_exe = framsexe if framsexe is not None else 'frams.exe' if os.name == "nt" else 'frams.linux'
		self.writing_path = None
		mainpath = os.path.join(self.frams_path, self.frams_exe)
		exe_call = [mainpath, '-Q', '-s', '-c', '-icliutils.ini']  # -c will be ignored in Windows Framsticks (this option is meaningless because the Windows version does not support color console, so no need to deactivate this feature using -c)
		exe_call_to_get_version = [mainpath, '-V']
		exe_call_to_get_path = [mainpath, '-?']
		try:
			print("\n".join(self.__readAllOutput(" ".join(exe_call_to_get_version))))
			help = self.__readAllOutput(" ".join(exe_call_to_get_path))
			for helpline in help:
				if 'dDIRECTORY' in helpline:
					self.writing_path = helpline.split("'")[1]
		except FileNotFoundError:
			print("Could not find Framsticks executable ('%s') in the given location ('%s')." % (self.frams_exe, self.frams_path))
			sys.exit(1)
		print("Temporary files with results will be saved in detected writable working directory '%s'" % self.writing_path)
		self.__spawnFramsticksCLI(exe_call)


	def __readAllOutput(self, command):
		frams_process = Popen(command, stdout=PIPE, stderr=PIPE, stdin=PIPE)
		return [line.decode('utf-8').rstrip() for line in iter(frams_process.stdout.readlines())]


	def __spawnFramsticksCLI(self, args):
		# the child app (Framsticks CLI) should not buffer outputs and we need to immediately read its stdout, hence we use pexpect/wexpect
		print('Spawning Framsticks CLI for continuous stdin/stdout communication... ', end='')
		if os.name == "nt":  # Windows:
			import wexpect  # https://pypi.org/project/wexpect/
			# https://github.com/raczben/wexpect/tree/master/examples
			self.child = wexpect.spawn(' '.join(args))
		else:
			import pexpect  # https://pexpect.readthedocs.io/en/stable/
			self.child = pexpect.spawn(' '.join(args))
			self.child.setecho(False)  # linux only
		print('OK.')

		self.__readFromFramsCLIUntil("UserScripts.autoload")
		print('Performing a basic test 1/3... ', end='')
		assert self.getSimplest("1") == "X"
		print('OK.')
		print('Performing a basic test 2/3... ', end='')
		assert self.isValid("X[0:0]") == True
		print('OK.')
		print('Performing a basic test 3/3... ', end='')
		assert self.isValid("X[0:0],") == False
		print('OK.')
		if not self.DETERMINISTIC:
			self.child.sendline(self.RANDOMIZE_CMD)
		self.child.sendline(self.SETEXPEDEF_CMD)


	def closeFramsticksCLI(self):
		# End gracefully by sending end-of-file character: ^Z or ^D
		# Without -Q argument ("quiet mode"), Framsticks CLI would print "Shell closed." for goodbye.
		self.child.sendline(chr(26 if os.name == "nt" else 4))


	def __saveGenotypeToFile(self, genotype, name, mode):
		outpath = os.path.join(self.writing_path, name)
		outfile = open(outpath, mode)
		outfile.write("org:\n")
		outfile.write("genotype:~\n")
		outfile.write(genotype + "~\n\n")  # TODO proper quoting of genotype...
		outfile.close()
		return name


	def __saveToFile(self, genotype, name, mode):
		outpath = os.path.join(self.writing_path, name)
		outfile = open(outpath, mode)
		outfile.write(genotype)
		outfile.close()
		return name


	def __removeFile(self, path):
		filepath = os.path.join(self.writing_path, path)
		if os.path.exists(filepath):
			os.remove(filepath)


	def __readFromFramsCLIUntil(self, until_marker: str):
		while True:
			self.child.expect('\n')
			msg = str(self.child.before)
			if self.PRINT_FRAMSTICKS_OUTPUT or msg.startswith("[ERROR]"):
				print(msg)
			if until_marker in msg:
				break


	def __runCommand(self, command, genotypes, result_file_name, saveformat) -> List[str]:
		filenames = []  # list of file names with input data for the command
		if saveformat == self.GENO_SAVE_FILE_FORMAT["RAWGENO"]:
			for i in range(len(genotypes)):
				filenames.append(self.__saveToFile(genotypes[i], "genotype" + str(i) + ".gen", "w"))  # plain text format = must have a separate file for each genotype
		elif saveformat == self.GENO_SAVE_FILE_FORMAT["NATIVEFRAMS"]:
			self.__removeFile(self.CLI_INPUT_FILE)  # ensure there is nothing left from the last run of the program because we "a"ppend to file in the loop below
			for i in range(len(genotypes)):
				outfilename = self.__saveGenotypeToFile(genotypes[i], self.CLI_INPUT_FILE, "a")
			filenames.append(outfilename)  # since we use the same file in the loop above, add this file only once (i.e., outside of the loop)

		if result_file_name != self.EVALUATE_FILE:  # all functions except for evaluate provide frams with the file name to write to
			self.child.sendline(command + " " + " ".join(filenames) + " " + result_file_name + "\n")
		else:
			self.child.sendline(command + " " + " ".join(filenames) + "\n")
		self.__readFromFramsCLIUntil(self.STDOUT_ENDOPER_MARKER)
		filenames.append(os.path.join(self.writing_path, self.OUTPUT_DIR, result_file_name))
		return filenames  # last element is a path to the file containing results


	def __cleanUpCommandResults(self, filepaths):
		"""Deletes files with results created by the command."""
		for i in filepaths:
			if i == filepaths[-1]:
				os.remove(i)  # the result is written with its full path and we have used it before so the file surely exists
			else:
				self.__removeFile(i)


	def getSimplest(self, genetic_format) -> str:
		assert len(genetic_format) == 1, "Genetic format should be a single character"
		files = self.__runCommand(self.GETSIMPLEST_CMD + " " + genetic_format + " ", [], self.GETSIMPLEST_FILE, self.GENO_SAVE_FILE_FORMAT["RAWGENO"])
		with open(files[-1]) as f:
			genotype = "".join(f.readlines())
		self.__cleanUpCommandResults(files)
		return genotype


	def evaluate(self, genotype: str):
		"""
		Returns:
			Dictionary -- genotype evaluated with self.EVALUATE_COMMAND. Note that for whatever reason (e.g. incorrect genotype),
			the dictionary you will get may be empty or partially empty and may not have the fields you expected, so handle such cases properly.
		"""
		files = self.__runCommand(self.EVALUATE_CMD, [genotype], self.EVALUATE_FILE, self.GENO_SAVE_FILE_FORMAT["NATIVEFRAMS"])
		with open(files[-1]) as f:
			data = json.load(f)
		if len(data) > 0:
			self.__cleanUpCommandResults(files)
			return data
		else:
			print("Evaluating genotype: no performance data was returned in", self.EVALUATE_FILE)  # we do not delete files here
			return None


	def mutate(self, genotype: str) -> str:
		files = self.__runCommand(self.MUTATE_CMD, [genotype], self.MUTATE_FILE, self.GENO_SAVE_FILE_FORMAT["RAWGENO"])
		with open(files[-1]) as f:
			newgenotype = "".join(f.readlines())
		self.__cleanUpCommandResults(files)
		return newgenotype


	def crossOver(self, genotype1: str, genotype2: str) -> str:
		files = self.__runCommand(self.CROSSOVER_CMD, [genotype1, genotype2], self.CROSSOVER_FILE, self.GENO_SAVE_FILE_FORMAT["RAWGENO"])
		with open(files[-1]) as f:
			child_genotype = "".join(f.readlines())
		self.__cleanUpCommandResults(files)
		return child_genotype


	def dissimilarity(self, genotype1: str, genotype2: str) -> float:
		files = self.__runCommand(self.DISSIMIL_CMD, [genotype1, genotype2], self.DISSIMIL_FILE, self.GENO_SAVE_FILE_FORMAT["NATIVEFRAMS"])
		with open(files[-1]) as f:
			dissimilarity_matrix = np.genfromtxt(f, dtype=np.float64, comments='#', encoding=None, delimiter='\t')
		# We would like to skip column #1 while reading and read everything else, but... https://stackoverflow.com/questions/36091686/exclude-columns-from-genfromtxt-with-numpy
		# This would be too complicated, so strings (names) in column #1 become NaN as floats (unless they accidentally are valid numbers) - not great, not terrible
		EXPECTED_SHAPE = (2, 4)
		assert dissimilarity_matrix.shape == EXPECTED_SHAPE, f"Not a correct dissimilarity matrix, expected {EXPECTED_SHAPE} "
		for i in range(len(dissimilarity_matrix)):
			assert dissimilarity_matrix[i][i + 2] == 0, "Not a correct dissimilarity matrix, diagonal expected to be 0"
		assert dissimilarity_matrix[0][3] == dissimilarity_matrix[1][2], "Probably not a correct dissimilarity matrix, expecting symmetry, verify this"
		self.__cleanUpCommandResults(files)
		return dissimilarity_matrix[0][3]


	def isValid(self, genotype: str) -> bool:
		files = self.__runCommand(self.ISVALID_CMD, [genotype], self.ISVALID_FILE, self.GENO_SAVE_FILE_FORMAT["RAWGENO"])
		with open(files[-1]) as f:
			valid = f.readline() == "1"
		self.__cleanUpCommandResults(files)
		return valid


def parseArguments():
	parser = argparse.ArgumentParser(description='Train this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
	parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks CLI without trailing slash.')
	parser.add_argument('-exe', required=False, help='Executable name. If not given, "frams.exe" or "frams.linux" is assumed.')
	parser.add_argument('-genformat', required=False, help='Genetic format for the demo run, for example 4, 9, or B. If not given, f1 is assumed.')
	return parser.parse_args()


def ensureDir(string):
	if os.path.isdir(string):
		return string
	else:
		raise NotADirectoryError(string)


if __name__ == "__main__":
	# A demo run.

	# TODO ideas:
	# - check_validity with three levels (invalid, corrected, valid)
	# - "vectorize" some operations (isvalid, evaluate) so that a number of genotypes is handled in one call
	# - use threads for non-blocking reading from frams' stdout and thus not relying on specific strings printed by frams
	# - a pool of binaries run at the same time, balance load - in particular evaluation
	# - if we read genotypes in "org:" format anywhere: import https://pypi.org/project/framsreader/0.1.2/ and use it if successful,
	#    if not then print a message "framsreader not available, using simple internal method to save a genotype" and proceed as it is now.
	#    So far we don't read, but we should use the proper writer to handle all special cases like quoting etc.

	# A demo run.
	# parsed_args = parseArguments()
	# framsCLI = FramsticksCLI(parsed_args.path, parsed_args.exe)
	representation = '1'
	framsCLI = FramsticksCLI('C:/Users/Piotr/Desktop/Framsticks50rc14', None)

	# simplest = framsCLI.getSimplest('1' if parsed_args.genformat is None else parsed_args.genformat)
	simplest = framsCLI.getSimplest(representation)
	print("\tSimplest genotype:", simplest)
	parent1 = framsCLI.mutate(simplest)
	parent2 = parent1
	MUTATE_COUNT = 40
	for x in range(MUTATE_COUNT):  # example of a chain of 20 mutations
		parent2 = framsCLI.mutate(parent2)
	print("\tParent1 (mutated simplest):", parent1)
	print("\tParent2 (Parent1 mutated %d times):" % MUTATE_COUNT, parent2)
	# offspring = framsCLI.crossOver(parent1, parent2)
	offspring = parent2.replace(' ', '')
	print("\tCrossover (Offspring):", offspring)
	print('\tDissimilarity of Parent1 and Offspring:', framsCLI.dissimilarity(offspring, parent1))
	print('\tPerformance of Offspring:', framsCLI.evaluate(offspring))
	print('\tValidity of Offspring:', framsCLI.isValid('/*4*/r<qFqX>X'))

	framsCLI.closeFramsticksCLI()


	# simplest = framsCLI.getSimplest('1')
	# parent1 = framsCLI.mutate(simplest)
	# parent2 = parent1
	# MUTATE_COUNT = 10
	# for x in range(MUTATE_COUNT):  # example of a chain of 20 mutations
	# 	parent2 = framsCLI.mutate(parent2)
	# print("\tParent1 (mutated simplest):", parent1)
	# print("\tParent2 (Parent1 mutated %d times):" % MUTATE_COUNT, parent2)
	# child = framsCLI.crossOver(parent1, parent2)
	# print("\tCrossover (Child):", child)
	# print('\tDissimilarity of Parent1 and Child:', framsCLI.dissimilarity(child, parent1))
	# print('\tPerformance:', framsCLI.evaluate(child))
	# framsCLI.endCommunication()

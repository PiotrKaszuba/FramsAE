from typing import List  # to be able to specify a type hint of list(something)
import json
import sys, os
import argparse
import numpy as np
import frams


class FramsticksLib:
	"""Communicates directly with Framsticks DLL/SO.
	You can perform basic operations like mutation, crossover, and evaluation of genotypes.
	This way you can perform evolution controlled by python as well as access and manipulate genotypes.
	You can even design and use in evolution your own genetic representation implemented entirely in python,
	or access and control the simulation and simulated creatures step by step.

	You need to provide one or two parameters when you run this class: the path to Framsticks CLI where .dll/.so resides
	and the name of the Framsticks dll/so (if it is non-standard). See::
		FramsticksLib.py -h"""

	PRINT_FRAMSTICKS_OUTPUT: bool = False  # set to True for debugging
	DETERMINISTIC: bool = False  # set to True to have the same results in each run

	GENOTYPE_INVALID = "/*invalid*/"  # this is how genotype invalidity is represented in Framsticks
	EVALUATION_SETTINGS_FILE = "eval-allcriteria.sim"  # MUST be compatible with standard-eval expdef


	def __init__(self, frams_path, frams_lib_name):
		frams.init(frams_path, frams_lib_name, "-Ddata")  # "-D"+os.path.join(frams_path,"data")) # not possible (maybe python windows issue) because of the need for os.chdir(). So we assume "data" is where the dll/so is

		print('Available objects:', dir(frams))
		print()

		print('Performing a basic test 1/2... ', end='')
		simplest = self.getSimplest("1")
		assert simplest == "X" and type(simplest) is str
		print('OK.')
		print('Performing a basic test 2/2... ', end='')
		assert self.isValid(["X[0:0],", "X[0:0]", "X[1:0]"]) == [False, True, False]
		print('OK.')
		if not self.DETERMINISTIC:
			frams.Math.randomize();
		frams.Simulator.expdef = "standard-eval"  # this expdef must be used by EVALUATION_SETTINGS_FILE


	def getSimplest(self, genetic_format) -> str:
		return frams.GenMan.getSimplest(genetic_format).genotype._string()


	def evaluate(self, genotype_list: List[str]):
		"""
		Returns:
			List of dictionaries containing the performance of genotypes evaluated using self.EVALUATION_SETTINGS_FILE.
			Note that for whatever reason (e.g. incorrect genotype), the dictionaries you will get may be empty or
			partially empty and may not have the fields you expected, so handle such cases properly.
		"""
		assert isinstance(genotype_list, list)  # because in python str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity

		if not self.PRINT_FRAMSTICKS_OUTPUT:
			ec = frams.MessageCatcher.new()  # mute potential errors, warnings, messages

		frams.GenePools[0].clear()
		frams.Simulator.ximport(self.EVALUATION_SETTINGS_FILE, 2 + 4 + 8 + 16)
		for g in genotype_list:
			frams.GenePools[0].add(g)
		frams.ExpProperties.evalsavefile = ""  # no need to store results in a file - we will get evaluations directly from Genotype's "data" field
		frams.Simulator.init()
		frams.Simulator.start()
		step = frams.Simulator.step  # cache reference to avoid repeated lookup in the loop
		while frams.Simulator.running._int():  # standard-eval.expdef sets running to 0 when the evaluation is complete
			step()

		if not self.PRINT_FRAMSTICKS_OUTPUT:
			if ec.error_count._value() > 0:  # errors are important and should not be ignored, at least display how many
				print("[ERROR]", ec.error_count, "error(s) and", ec.warning_count, "warning(s) while evaluating", len(genotype_list), "genotype(s)")
			ec.close()

		results = []
		for g in frams.GenePools[0]:
			serialized_dict = frams.String.serialize(g.data[frams.ExpProperties.evalsavedata._value()])
			evaluations = json.loads(serialized_dict._string())
			# now, for consistency with FramsticksCLI.py, add "num" and "name" keys that are missing because we got data directly from Genotype, not from the file produced by standard-eval.expdef's function printStats(). What we do below is what printStats() does.
			result = {"num": g.num._value(), "name": g.name._value(), "evaluations": evaluations}
			results.append(result)

		return results


	def mutate(self, genotype_list: List[str]) -> List[str]:
		"""
		Returns:
			The genotype(s) of the mutated source genotype(s). self.GENOTYPE_INVALID for genotypes whose mutation failed (for example because the source genotype was invalid).
		"""
		assert isinstance(genotype_list, list)  # because in python str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity

		mutated = []
		for g in genotype_list:
			mutated.append(frams.GenMan.mutate(frams.Geno.newFromString(g)).genotype._string())
		assert len(genotype_list) == len(mutated), "Submitted %d genotypes, received %d validity values" % (len(genotype_list), len(mutated))
		return mutated


	def crossOver(self, genotype_parent1: str, genotype_parent2: str) -> str:
		"""
		Returns:
			The genotype of the offspring. self.GENOTYPE_INVALID if the crossing over failed.
		"""
		return frams.GenMan.crossOver(frams.Geno.newFromString(genotype_parent1), frams.Geno.newFromString(genotype_parent2)).genotype._string()


	def dissimilarity(self, genotype_list: List[str]) -> np.ndarray:
		"""
		Returns:
			A square array with dissimilarities of each pair of genotypes.
		"""
		assert isinstance(genotype_list, list)  # because in python str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity

		frams.SimilMeasure.type = 1  # adjust to your needs. Set here because loading EVALUATION_SETTINGS_FILE during evaluation may overwrite these parameters
		frams.SimilMeasureHungarian.simil_weightedMDS = 1
		frams.SimilMeasureHungarian.simil_partgeom = 1

		n = len(genotype_list)
		square_matrix = np.zeros((n, n))
		genos = []  # prepare an array of Geno objects so we don't need to convert raw strings to Geno objects all the time
		for g in genotype_list:
			genos.append(frams.Geno.newFromString(g))
		for i in range(n):
			for j in range(n):  # maybe calculate only one triangle if you really need a 2x speedup
				square_matrix[i][j] = frams.SimilMeasure.evaluateDistance(genos[i], genos[j])._double()

		for i in range(n):
			assert square_matrix[i][i] == 0, "Not a correct dissimilarity matrix, diagonal expected to be 0"
		assert (square_matrix == square_matrix.T).all(), "Probably not a correct dissimilarity matrix, expecting symmetry, verify this"  # could introduce tolerance in comparison (e.g. class field DISSIMIL_DIFF_TOLERANCE=10^-5) so that miniscule differences do not fail here
		return square_matrix


	def isValid(self, genotype_list: List[str]) -> List[bool]:
		assert isinstance(genotype_list, list)  # because in python str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity
		valid = []
		for g in genotype_list:
			valid.append(frams.Geno.newFromString(g).is_valid._int() == 1)
		assert len(genotype_list) == len(valid), "Submitted %d genotypes, received %d validity values" % (len(genotype_list), len(valid))
		return valid


def parseArguments():
	parser = argparse.ArgumentParser(description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
	parser.add_argument('-path', type=ensureDir, required=True, help='Path to the Framsticks library (.dll or .so) without trailing slash.')
	parser.add_argument('-lib', required=False, help='Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.')
	parser.add_argument('-genformat', required=False, help='Genetic format for the demo run, for example 4, 9, or S. If not given, f1 is assumed.')
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
	# - a pool of binaries running simultaneously, balance load - in particular evaluation

	parsed_args = parseArguments()
	framsDLL = FramsticksLib(parsed_args.path, parsed_args.lib)

	print("Sending a direct command to Framsticks CLI that calculates \"4\"+2 yields", frams.Simulator.eval("return \"4\"+2;"))

	simplest = framsDLL.getSimplest('1' if parsed_args.genformat is None else parsed_args.genformat)
	print("\tSimplest genotype:", simplest)
	parent1 = framsDLL.mutate([simplest])[0]
	parent2 = parent1
	MUTATE_COUNT = 10
	for x in range(MUTATE_COUNT):  # example of a chain of 10 mutations
		parent2 = framsDLL.mutate([parent2])[0]
	print("\tParent1 (mutated simplest):", parent1)
	print("\tParent2 (Parent1 mutated %d times):" % MUTATE_COUNT, parent2)
	offspring = framsDLL.crossOver(parent1, parent2)
	print("\tCrossover (Offspring):", offspring)
	print('\tDissimilarity of Parent1 and Offspring:', framsDLL.dissimilarity([parent1, offspring])[0, 1])
	print('\tPerformance of Offspring:', framsDLL.evaluate([offspring]))
	print('\tValidity of Parent1, Parent 2, and Offspring:', framsDLL.isValid([parent1, parent2, offspring]))

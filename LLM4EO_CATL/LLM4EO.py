import copy
import math
import random
import numpy as np
from scipy.stats import norm
from DataProcess.Fitness import Fitness
from DataProcess.Translate import Translate
from LLM.Strategy import Strategy
fitness = Fitness()
translate = Translate()
strategy = Strategy()

class LLM4EO:
    def __init__(self, data, solutionSpace,paras):
        self.data = data
        self.solutionSpace = solutionSpace
        self.chromosome = []
        self.paras = paras
        self.GroupSize = paras.GroupSize
        self.iterNum = paras.iterations
        self.nowIterNum = 0
        self.scoreArray = []
        self.lastScoreArray = []
        self.calAlgorithm = {}
        self.calAlgorithmList = []
        self.character = {}
        self.characterList = []

    def main(self):
        bestChromosome, bestScore = self.evolve()
        return bestChromosome, bestScore

    def evolve(self):
        deltaT = 1
        self.scoreArray = []
        geneDataArray = []
        for i in range(len(self.solutionSpace)):
            chromosome = self.solutionSpace[i]
            geneData = translate.main(self.data, chromosome)
            self.scoreArray.append(geneData['score'])
            geneDataArray.append(geneData)
        self.lastScoreArray = copy.deepcopy(self.scoreArray)
        print("Initial best makespan：" + str(min(self.scoreArray)))


        # Initial operator population
        algorithmNum = self.paras.initCreateNum
        for i in range(algorithmNum):
            print('Generate initial operator population：' + str(i + 1) + '/' + str(algorithmNum))
            self.calAlgorithm = self.strategy_init()
            self.calAlgorithmList.append(self.calAlgorithm)
        ruleUpdateNum = 0

        # Iteration
        pop = copy.deepcopy(self.solutionSpace)
        for i in range(self.iterNum):
            self.nowIterNum = i

            bestMakespan = copy.deepcopy(min(self.scoreArray))
            meanMakespan = copy.deepcopy(np.average(self.scoreArray))
            print('Iteration: ' + str(i) + '/' + str(self.iterNum) + '; ' + 'Best Makespan：' + str(bestMakespan) + '; ' + 'Mean Makespan: ' + str(meanMakespan))

            # crossover
            for crs in range(0, self.GroupSize, 2):
                if random.random() > self.paras.crossRate:
                    continue

                # solution selection
                if self.paras.selection == 'roulette':  # roulette
                    fitnessArray = fitness.main(self.scoreArray)
                    parents, parentsIndex = self.roulette_selection(fitnessArray, pop, 2)
                else: # binary
                    fitnessArray = copy.deepcopy(self.scoreArray)
                    parents, parentsIndex = self.binary_tournament(fitnessArray, pop, 2)

                parentsChorosome = copy.deepcopy(parents)
                scoreList = [self.scoreArray[parentsIndex[0]], self.scoreArray[parentsIndex[1]]]
                geneDataList = [geneDataArray[parentsIndex[0]], geneDataArray[parentsIndex[1]]]
                childList, scoreList, geneDataList = self.crossover_MS(parents, scoreList, geneDataList)
                childList, scoreList, geneDataList = self.crossover_OS(childList, scoreList, geneDataList)
                for j in range(len(parentsIndex)):
                    score = scoreList[j]
                    if score < self.scoreArray[parentsIndex[j]]:
                        self.scoreArray[parentsIndex[j]] = copy.deepcopy(score)
                        pop[parentsIndex[j]] = copy.deepcopy(childList[j])
                        geneDataArray[parentsIndex[j]] = copy.deepcopy(geneDataList[j])
                    else:
                        pop[parentsIndex[j]] = copy.deepcopy(parentsChorosome[j])

            # mutation
            for mut in range(self.GroupSize):
                if random.random() > self.paras.muteRate:
                    continue

                # solution selection
                if self.paras.selection == 'roulette':  # roulette
                    fitnessArray = fitness.main(self.scoreArray)
                    parents, parentsIndex = self.roulette_selection(fitnessArray, pop, 1)
                else: # binary
                    fitnessArray = copy.deepcopy(self.scoreArray)
                    parents, parentsIndex = self.binary_tournament(fitnessArray, pop, 1)

                chorosome = copy.deepcopy(parents[0])
                geneData = copy.deepcopy(geneDataArray[parentsIndex[0]])
                muteChromosome, geneData = self.mutation_MS(chorosome, geneData)
                muteChromosome, geneData = self.mutation_OS(muteChromosome, geneData)
                score = geneData["score"]
                if score < self.scoreArray[parentsIndex[0]]:
                    self.scoreArray[parentsIndex[0]] = copy.deepcopy(score)
                    pop[parentsIndex[0]] = copy.deepcopy(muteChromosome)
                    geneDataArray[parentsIndex[0]] = copy.deepcopy(geneData)
                else:
                    pop[parentsIndex[0]] = copy.deepcopy(chorosome)

            if min(self.scoreArray) < bestMakespan:
                deltaT = 1
            else:
                deltaT = deltaT + 1

            f3 = 1 / (deltaT * 0.05)
            if random.random() > f3:
                ruleUpdateNum = ruleUpdateNum + 1
                self.calAlgorithm = self.strategy_improve(self.calAlgorithmList)
                minRate = 1
                aloIndex = -1
                for j in range(len(self.calAlgorithmList)):
                    rate = self.calAlgorithmList[j]['success'] / self.calAlgorithmList[j]['run']
                    if minRate > rate:
                        minRate = rate
                        aloIndex = j
                    self.calAlgorithmList[j]['success'] = 1
                    self.calAlgorithmList[j]['run'] = 1
                self.calAlgorithmList[aloIndex] = copy.deepcopy(self.calAlgorithm)
        bestScore = min(self.scoreArray)
        bestChromosome = pop[self.scoreArray.index(bestScore)]
        return bestChromosome, bestScore

    def roulette_selection(self, fitnessArray, pop, num):
        parentsIndex = []
        parents = []
        while len(parentsIndex) < num:
            probability = random.random()
            left = 0
            right = len(fitnessArray) - 1
            out = 0
            while left <= right and out == 0:
                mid = (left + right) // 2
                if fitnessArray[mid][0] <= probability and fitnessArray[mid][1] >= probability:
                    if mid not in parentsIndex:
                        parentsIndex.append(mid)
                        parents.append(pop[mid])
                        out = 1
                    else:
                        out = 1
                elif fitnessArray[mid][1] < probability:
                    left = mid + 1
                else:
                    right = mid - 1
        return parents, parentsIndex

    def binary_tournament(self, fitness_values, pop, num_selected):
        population_size = len(fitness_values)
        selected = []
        parents = []

        for _ in range(num_selected):
            candidates = random.sample(range(population_size), 2)
            f1 = fitness_values[candidates[0]]
            f2 = fitness_values[candidates[1]]

            if f1 < f2:
                selected.append(candidates[0])
                parents.append(pop[candidates[0]])
            else:
                selected.append(candidates[1])
                parents.append(pop[candidates[1]])
        return parents, selected

    def crossover_MS(self, parents, scoreList, geneDataList):
        chromosomeList = copy.deepcopy(parents)
        chromosomeLength = len(chromosomeList[0]["job"])
        mateGeneList = []
        for i in range(len(parents)):
            geneData = geneDataList[i]
            chromosome = parents[i]

            # operator seleciton
            successRate = []
            for n in range(len(self.calAlgorithmList)):
                rate = self.calAlgorithmList[n]['run'] / self.calAlgorithmList[n]['success']
                successRate.append(rate)
            successRate = fitness.main(successRate)
            _, aloIndex = self.roulette_selection(successRate, self.calAlgorithmList, 1)
            aloIndex = aloIndex[0]

            self.calAlgorithmList[aloIndex]['run'] += 1
            job_probability, machine_probability = self.getGenePerturbate(aloIndex, geneData, chromosome)
            chromosome["perturbate_rate"] = machine_probability
            normalized_probabilities = self.gerCrossGene(chromosome)

            # Gene selection
            genes = {'job': [], 'operation': [], 'index': []}
            for j in range(chromosomeLength):
                if random.random() < normalized_probabilities[j]:
                    genes['job'].append(chromosome['job'][j])
                    genes['operation'].append(chromosome['operation'][j])
                    genes['index'].append(j)
            mateGeneList.append(genes)

        childList = copy.deepcopy(chromosomeList)
        childMateLine = [[], []]
        for muteIndex in [0, 1]:
            childIndex = abs(1 - muteIndex)
            for i in range(len(mateGeneList[muteIndex]["job"])):
                for j in range(chromosomeLength):
                    if childList[childIndex]["job"][j] == mateGeneList[muteIndex]["job"][i] \
                            and childList[childIndex]["operation"][j] == mateGeneList[muteIndex]["operation"][i]:
                        childMateLine[muteIndex].append(childList[childIndex]["machine"][j])
                        break

        # Generate solutions
        for i in range(len(childMateLine[0])):
            index = mateGeneList[0]["index"][i]
            line = childMateLine[0][i]
            childList[0]["machine"][index] = line
        for i in range(len(childMateLine[1])):
            index = mateGeneList[1]["index"][i]
            line = childMateLine[1][i]
            childList[1]["machine"][index] = line

        if len(mateGeneList[1]["job"]) > 0:
            geneData_1 = translate.main(self.data, childList[0])
            newScre_1 = geneData_1['score']
            if scoreList[0] > newScre_1:
                self.calAlgorithmList[aloIndex]['success'] += 1
                chromosomeList[0] = childList[0]
                scoreList[0] = newScre_1
                geneDataList[0] = geneData_1
        if len(mateGeneList[0]["job"]) > 0:
            geneData_2 = translate.main(self.data, childList[1])
            newScre_2 = geneData_2['score']
            if scoreList[1] > newScre_2:
                self.calAlgorithmList[aloIndex]['success'] += 1
                chromosomeList[1] = childList[1]
                scoreList[1] = newScre_2
                geneDataList[1] = geneData_2

        return chromosomeList, scoreList, geneDataList

    def crossover_OS(self, parents, scoreList, geneDataList):
        chromosomeList = copy.deepcopy(parents)
        chromosomeLength = len(chromosomeList[0]["job"])

        mateGeneList = []
        for i in range(len(parents)):
            geneData = geneDataList[i]
            chromosome = parents[i]

            # operator seleciton
            successRate = []
            for n in range(len(self.calAlgorithmList)):
                rate = self.calAlgorithmList[n]['run'] / self.calAlgorithmList[n]['success']
                successRate.append(rate)
            successRate = fitness.main(successRate)
            _, aloIndex = self.roulette_selection(successRate, self.calAlgorithmList, 1)
            aloIndex = aloIndex[0]
            self.calAlgorithmList[aloIndex]['run'] += 1
            job_probability, machine_probability = self.getGenePerturbate(aloIndex, geneData, chromosome)
            chromosome["perturbate_rate"] = job_probability

            # Gene selection
            normalized_probabilities = self.gerCrossGene(chromosome)
            genJobs = []
            for j in range(len(geneData['jobList'])):
                job_id = geneData['jobList'][j]
                if random.random() < normalized_probabilities[j]:
                    genJobs.append(job_id)
            crossGens = {'job': [], 'operation': [], 'machine': [], 'index': []}
            for j in range(chromosomeLength):
                if chromosome['job'][j] in genJobs:
                    crossGens["job"].append(chromosome["job"][j])
                    crossGens["operation"].append(chromosome["operation"][j])
                    crossGens["machine"].append(chromosome["machine"][j])
                    crossGens["index"].append(j)
            mateGeneList.append(crossGens)

        # Generate solutions
        childList = copy.deepcopy(chromosomeList)
        for muteIndex in [0, 1]:
            jobList = []
            operationList = []
            machineList = []
            childIndex = abs(1 - muteIndex)
            mIndex = -1
            if len(np.unique(mateGeneList[muteIndex]['job'])) < 2:
                continue
            for j in range(len(chromosomeList[childIndex]["job"])):
                if chromosomeList[childIndex]["job"][j] not in mateGeneList[muteIndex]['job']:
                    jobList.append(chromosomeList[childIndex]["job"][j])
                    operationList.append(chromosomeList[childIndex]["operation"][j])
                    machineList.append(chromosomeList[childIndex]["machine"][j])
                if j in mateGeneList[muteIndex]['index']:
                    mIndex += 1
                    jobList.append(mateGeneList[muteIndex]["job"][mIndex])
                    operationList.append(mateGeneList[muteIndex]["operation"][mIndex])
                    machineList.append(mateGeneList[muteIndex]["machine"][mIndex])

            childList[childIndex]['job'] = jobList
            childList[childIndex]['operation'] = operationList
            childList[childIndex]['machine'] = machineList

        if len(mateGeneList[1]["job"]) > 0:
            geneData_1 = translate.main(self.data, childList[0])
            newScre_1 = geneData_1['score']
            if scoreList[0] > newScre_1:
                self.calAlgorithmList[aloIndex]['success'] += 1
                chromosomeList[0] = childList[0]
                scoreList[0] = newScre_1
                geneDataList[0] = geneData_1

        if len(mateGeneList[0]["job"]) > 0:
            geneData_2 = translate.main(self.data, childList[1])
            newScre_2 = geneData_2['score']
            if scoreList[1] > newScre_2:
                self.calAlgorithmList[aloIndex]['success'] += 1
                chromosomeList[1] = childList[1]
                scoreList[1] = newScre_2
                geneDataList[1] = geneData_2

        return chromosomeList, scoreList, geneDataList

    def mutation_OS(self, chromosome, geneData):
        # operator selection
        successRate = []
        for n in range(len(self.calAlgorithmList)):
            rate = self.calAlgorithmList[n]['run'] / self.calAlgorithmList[n]['success']
            successRate.append(rate)
        successRate = fitness.main(successRate)
        _, aloIndex = self.roulette_selection(successRate, self.calAlgorithmList, 1)
        aloIndex = aloIndex[0]
        self.calAlgorithmList[aloIndex]['run'] += 1
        job_probability, machine_probability = self.getGenePerturbate(aloIndex, geneData, chromosome)
        chromosome["perturbate_rate"] = machine_probability

        # gene selection
        chromosomeLength = len(chromosome["perturbate_rate"])
        indices = list(range(chromosomeLength))
        mute_index = random.choice(indices)
        muteGene = {'job': chromosome['job'][mute_index],
                    'operation': chromosome['operation'][mute_index], 'machine': chromosome['machine'][mute_index]}

        # generate solutions
        front_index = 0
        back_index = chromosomeLength
        if muteGene['operation'] > 0:
            for i in range(0, mute_index):
                if chromosome['job'][mute_index - i - 1] == muteGene['job']:
                    front_index = mute_index - i - 1
                    break
        if muteGene['operation'] < len(
                self.data.jobProcess[muteGene['job']]['operations']) - 1:
            for i in range(mute_index + 1, chromosomeLength):
                if chromosome['job'][i] == muteGene['job']:
                    back_index = i
                    break
        insertList = []
        for i in range(front_index + 1, back_index):
            if chromosome['machine'][i] == muteGene['machine'] and i != mute_index:
                insertList.append(i)
        if len(insertList) > 0:
            insertIndex = random.choice(insertList)
            oldChromosome = copy.deepcopy(chromosome)
            newChromosome = copy.deepcopy(chromosome)
            if insertIndex < mute_index:
                newChromosome['job'] = oldChromosome['job'][0:front_index] + oldChromosome['job'][
                                                                             front_index:insertIndex] + [
                                           muteGene['job']] + oldChromosome['job'][insertIndex:mute_index] + \
                                       oldChromosome['job'][mute_index + 1:chromosomeLength]
                newChromosome['operation'] = oldChromosome['operation'][0:front_index] + oldChromosome['operation'][
                                                                                         front_index:insertIndex] + [
                                                 muteGene['operation']] + oldChromosome['operation'][
                                                                          insertIndex:mute_index] + oldChromosome[
                                                                                                        'operation'][
                                                                                                    mute_index + 1:chromosomeLength]
                newChromosome['machine'] = oldChromosome['machine'][0:front_index] + oldChromosome['machine'][
                                                                                     front_index:insertIndex] + [
                                               muteGene['machine']] + oldChromosome['machine'][insertIndex:mute_index] + \
                                           oldChromosome['machine'][mute_index + 1:chromosomeLength]
            else:
                newChromosome['job'] = oldChromosome['job'][0:mute_index] + oldChromosome['job'][
                                                                            mute_index + 1:insertIndex + 1] + [
                                           muteGene['job']] + oldChromosome['job'][insertIndex + 1:chromosomeLength]
                newChromosome['operation'] = oldChromosome['operation'][0:mute_index] + oldChromosome['operation'][
                                                                                        mute_index + 1:insertIndex + 1] + [
                                                 muteGene['operation']] + oldChromosome['operation'][
                                                                          insertIndex + 1:chromosomeLength]
                newChromosome['machine'] = oldChromosome['machine'][0:mute_index] + oldChromosome['machine'][
                                                                                    mute_index + 1:insertIndex + 1] + [
                                               muteGene['machine']] + oldChromosome['machine'][
                                                                      insertIndex + 1:chromosomeLength]
            newGeneData = translate.main(self.data, newChromosome)
            if newGeneData['score'] < geneData['score']:
                geneData = copy.deepcopy(newGeneData)
                chromosome = copy.deepcopy(newChromosome)
                self.calAlgorithmList[aloIndex]['success'] += 1

        return chromosome, geneData

    def mutation_MS(self, chromosome, geneData):
        newChromosome = copy.deepcopy(chromosome)

        # operator selection
        successRate = []
        for n in range(len(self.calAlgorithmList)):
            rate = self.calAlgorithmList[n]['run'] / self.calAlgorithmList[n]['success']
            successRate.append(rate)
        successRate = fitness.main(successRate)
        _, aloIndex = self.roulette_selection(successRate, self.calAlgorithmList, 1)
        aloIndex = aloIndex[0]
        self.calAlgorithmList[aloIndex]['run'] += 1
        job_probability, machine_probability = self.getGenePerturbate(aloIndex, geneData, chromosome)
        newChromosome["perturbate_rate"] = machine_probability

        # gene selection
        chromosomeLength = len(newChromosome["job"])
        normalized_probabilities = self.gerCrossGene(newChromosome)
        genes = {'job': [], 'operation': [], 'index': []}
        for j in range(chromosomeLength):
            if random.random() < normalized_probabilities[j]:
                genes['job'].append(chromosome['job'][j])
                genes['operation'].append(chromosome['operation'][j])
                genes['index'].append(j)

        # generate solutions
        for i in genes["index"]:
            job_id = newChromosome['job'][i]
            operation_id = newChromosome['operation'][i]
            machineList = list(
                self.data.jobProcess[job_id]['operations'][operation_id]['machines'].keys())
            workTimeList = []
            for machine_id in machineList:
                workTimeList.append(geneData['machine_utilize'][machine_id])
            newChromosome["machine"][i] = random.choice(machineList)
        newGeneData = translate.main(self.data, newChromosome)
        if newGeneData['score'] < geneData['score']:
            self.calAlgorithmList[aloIndex]['success'] += 1
            geneData = copy.deepcopy(newGeneData)
            chromosome = copy.deepcopy(newChromosome)

        return chromosome, geneData

    def gerCrossGene(self, chromosome):
        gene_probabilities = chromosome['perturbate_rate']
        mu = np.mean(gene_probabilities)
        sigma = np.std(gene_probabilities)
        normalized_probabilities = [self.normalize_probability(p, mu, sigma) for p in gene_probabilities]
        return normalized_probabilities

    def normalize_probability(self, p, mu, sigma):
        return norm.cdf(p, loc=mu, scale=sigma)

    def getGenePerturbate(self, aloIndex, geneData, chromosome):
        check = True
        while check:
            calAlgorithm = self.calAlgorithmList[aloIndex]
            exec(calAlgorithm['code'], globals())
            try:
                job_probability, machine_probability = globals()['calculate_priority'](geneData["job_totalTime"],
                                                                                       geneData["job_shortestTime"],
                                                                                       geneData["job_operationNumber"],
                                                                                       geneData["operation_startTime"],
                                                                                       geneData["operation_earliestStartTime"],
                                                                                       geneData[
                                                                                           "operation_processTime"],
                                                                                       geneData["operation_machineNum"])
                if len(machine_probability) == len(chromosome['operation']) and len(job_probability) == len(
                        np.unique(chromosome['job'])):
                    check = False
            except Exception as e:
                self.calAlgorithm = self.strategy_improve(self.calAlgorithmList)
                self.calAlgorithmList[aloIndex] = copy.deepcopy(self.calAlgorithm)
        return job_probability, machine_probability

    def strategy_init(self):
        calAlgorithm = {}
        [calAlgorithm['code'], calAlgorithm['algorithm']] = strategy.init(self.paras)
        calAlgorithm['success'] = 1
        calAlgorithm['run'] = 1
        return calAlgorithm

    def strategy_improve(self, parents):
        calAlgorithm = {}
        characterList = self.getPopCharacter_improve()
        [calAlgorithm['code'], calAlgorithm['algorithm']] = strategy.improve(parents, characterList,self.paras)
        calAlgorithm['success'] = 1
        calAlgorithm['run'] = 1
        return calAlgorithm

    def getPopCharacter(self):
        character = {'nowIterNum': self.nowIterNum, 'iterNum': self.iterNum, "meanFitness": {}, "optimalFitness": {},
                     "deviation": {}}
        if len(self.characterList) == 0:
            character["meanFitness"]["gener_range"] = [0, self.nowIterNum]
            character["meanFitness"]["change_rate"] = round(
                (sum(self.lastScoreArray) - sum(self.scoreArray)) / sum(self.scoreArray), 2)
        else:
            character["meanFitness"]["gener_range"] = [self.characterList[-1]["meanFitness"]["gener_range"][1],
                                                       self.nowIterNum]
            character["meanFitness"]["change_rate"] = round(
                (sum(self.lastScoreArray) - sum(self.scoreArray)) / sum(self.scoreArray), 2)

        character["meanFitness"]["value"] = round(sum(self.scoreArray) / len(self.scoreArray), 2)
        character["optimalFitness"]["value"] = round(min(self.scoreArray), 2)
        character["optimalFitness"]["change_rate"] = round(
            (min(self.lastScoreArray) - min(self.scoreArray)) / min(self.lastScoreArray), 2)

        avgFitness = np.average(self.lastScoreArray)
        squared_diffs = [(x - avgFitness) ** 2 for x in self.lastScoreArray]
        population_variance = sum(squared_diffs) / len(self.lastScoreArray)
        initial_dev = math.sqrt(population_variance)
        character["deviation"]["initial_dev"] = round(initial_dev, 2)

        avgFitness = np.average(self.scoreArray)
        squared_diffs = [(x - avgFitness) ** 2 for x in self.scoreArray]
        population_variance = sum(squared_diffs) / len(self.scoreArray)
        character["deviation"]["change_rate"] = round((math.sqrt(population_variance) - initial_dev) / initial_dev, 2)

        self.lastScoreArray = copy.deepcopy(self.scoreArray)
        return character

    def getPopCharacter_improve(self):
        character = self.getPopCharacter()
        if character not in self.characterList:
            self.characterList.append(character)
        return self.characterList

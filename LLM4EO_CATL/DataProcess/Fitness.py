class Fitness:
    def main(self, scoreArray):
        fitnessArray = []
        fitnessTemp = []
        scoreArrayTemp = []
        for i in range(len(scoreArray)):
            gap = scoreArray[i] - min(scoreArray)
            scoreArrayTemp.append(gap)
        scoreArray = scoreArrayTemp
        for score in scoreArray:
            if sum(scoreArray) == 0:
                temp = 1
            else:
                temp = 1 - score / sum(scoreArray)
            fitnessTemp.append(temp)
        for temp in fitnessTemp:
            if sum(fitnessTemp) != 0:
                fitness = temp / sum(fitnessTemp)
            else:
                fitness = 1
            if len(fitnessArray) == 0:
                fitnessArray.append([0, fitness])
            else:
                fitness = fitness + fitnessArray[-1][1]
                fitnessArray.append([fitnessArray[-1][1], fitness])
        return fitnessArray


Fitness()

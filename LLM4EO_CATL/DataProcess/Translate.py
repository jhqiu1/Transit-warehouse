import numpy as np
from Parameter import Paras
paras = Paras()
class Translate:
    def main(self, data, chromosome):
        bigM = 10000000000000000000
        recordTime = {}
        machineTime = {}
        finishTime = 0
        jobMachine = {}
        operation_startTime = []
        operation_earlyStart = []
        operation_processTime = []
        operation_minProcessTime = []
        operation_machine = []
        select_utilization = []
        mean_utilization = []
        minimum_utilization = []
        machine_utilize = {}
        machine_processTime = {}
        operation_machineNum = []
        jobNum = len(np.unique(chromosome['job']))
        job_startTime = list(10000000 for i in range(jobNum))
        job_endTime = list(0 for i in range(jobNum))
        job_totalTime = list(0 for i in range(jobNum))
        job_shortestTime = list(0 for i in range(jobNum))
        job_operationNumber = list(0 for i in range(jobNum))
        jobList = []

        for machine_id in range(data.proMachineNum):
            machineTime[machine_id] = []
            machine_utilize[machine_id] = 0
            machine_processTime[machine_id] = 0

        for i in range(len(chromosome['job'])):
            job_id = chromosome['job'][i]
            operation_id = chromosome['operation'][i]
            machine_id = chromosome['machine'][i]
            operation_code = chromosome['operation_code'][i]
            processTime = data.jobProcess[job_id]['operations'][operation_id]['machines'][
                machine_id]

            if job_id not in recordTime.keys():
                recordTime[job_id] = {}
            if operation_id not in recordTime[job_id].keys():
                recordTime[job_id][operation_id] = [0, 0]

            preJobEndTime = 0
            if operation_id > 0:
                preJobEndTime = recordTime[job_id][operation_id - 1][1]

            availableTime = []
            for j in range(len(machineTime[machine_id])):
                avail_startTime = machineTime[machine_id][j][1]
                if len(machineTime[machine_id]) - 1 > j:
                    avail_endTime = machineTime[machine_id][j + 1][0]
                else:
                    avail_endTime = bigM
                availableTime.append([avail_startTime, avail_endTime])

            if len(availableTime) == 0:
                availableTime.append([0, bigM])
            machineBestTime = self.chooseLineTimeLeft(availableTime, preJobEndTime, processTime)

            if len(machineTime[machine_id]) == 0:
                machineTime[machine_id].append(machineBestTime)
            else:
                timeList_1 = []
                timeList_2 = []
                for j in range(0, len(machineTime[machine_id])):
                    if machineTime[machine_id][j][1] <= machineBestTime[0]:
                        timeList_1.append(machineTime[machine_id][j])
                    else:
                        timeList_2.append(machineTime[machine_id][j])
                machineTime[machine_id] = timeList_1 + [[machineBestTime[0], machineBestTime[1]]] + timeList_2

            recordTime[job_id][operation_id] = machineBestTime
            jobMachine[(job_id, operation_id, machine_id, operation_code)] = (
            machineBestTime[0], machineBestTime[1], processTime)
            if finishTime < machineBestTime[1]:
                finishTime = machineBestTime[1]

            startTime = recordTime[job_id][operation_id][0]
            endTime = recordTime[job_id][operation_id][1]
            processTime = endTime - startTime

            operation_startTime.append(startTime)
            operation_processTime.append(processTime)
            operation_machine.append(machine_id)
            machine_processTime[machine_id] = machine_processTime[machine_id] + processTime
            machine_utilize[machine_id] = round(machine_processTime[machine_id] / endTime, 2)
            select_utilization.append(machine_utilize[machine_id])
            machineDict = data.jobProcess[job_id]['operations'][operation_id]['machines']
            min_key = min(machineDict, key=machineDict.get)
            operation_minProcessTime.append(machineDict[min_key])
            job_shortestTime[job_id] += processTime
            job_operationNumber[job_id] += 1
            if startTime < job_startTime[job_id]:
                job_startTime[job_id] = startTime
            if endTime > job_endTime[job_id]:
                job_endTime[job_id] = endTime
            job_totalTime[job_id] = job_endTime[job_id] - job_startTime[job_id]

            if operation_id == 0:
                operation_earlyStart.append(0)
            else:
                operation_earlyStart.append(recordTime[job_id][operation_id - 1][1])
            temp = []
            operation_machineNum.append(0)
            for machine_id in data.jobProcess[job_id]['operations'][operation_id][
                'machines'].keys():
                temp.append(machine_utilize[machine_id])
                operation_machineNum[-1] += 1
            mean_utilization.append(round(sum(temp) / len(temp), 2))
            minimum_utilization.append(min(temp))
            if job_id not in jobList:
                jobList.append(job_id)

        geneData = {}
        geneData["makespan"] = finishTime
        geneData["recordTime"] = recordTime
        geneData["machineTime"] = machineTime
        geneData["jobMachine"] = jobMachine
        geneData["score"] = geneData[paras.object]
        geneData["operation_startTime"] = operation_startTime
        geneData["operation_earliestStartTime"] = operation_earlyStart
        geneData["operation_processTime"] = operation_processTime
        geneData["operation_machine"] = operation_machine
        geneData["select_utilization"] = select_utilization
        geneData["mean_utilization"] = mean_utilization
        geneData["minimum_utilization"] = minimum_utilization
        geneData["machine_utilize"] = machine_utilize
        geneData["machine_processTime"] = machine_processTime
        geneData["operation_minProcessTime"] = operation_minProcessTime
        geneData['operation_machineNum'] = operation_machineNum
        geneData['job_totalTime'] = job_totalTime
        geneData['job_shortestTime'] = job_shortestTime
        geneData['job_operationNumber'] = job_operationNumber
        geneData['jobList'] = jobList
        return geneData

    def chooseLineTimeLeft(self, availableTime, last_endTime, processTime):
        startTimeList = []
        endTimeList = []
        lineBestTime = []
        for num in range(len(availableTime)):
            startTime = availableTime[num][0]
            endTime = availableTime[num][1]

            if startTime <= last_endTime and last_endTime + processTime < endTime:
                startTimeList.append(last_endTime)
                endTimeList.append(last_endTime + processTime)
            elif num == len(availableTime) - 1:
                if last_endTime >= startTime:
                    startTimeList.append(last_endTime)
                    endTimeList.append(last_endTime + processTime)
                else:
                    startTimeList.append(startTime)
                    endTimeList.append(startTime + processTime)
        index = endTimeList.index(min(endTimeList))
        lineBestTime.append(startTimeList[index])
        lineBestTime.append(endTimeList[index])
        return lineBestTime


Translate()

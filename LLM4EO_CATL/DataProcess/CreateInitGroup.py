import random
from Parameter import Paras
paras = Paras()
from DataProcess.Translate import Translate
translate = Translate()
stan_date = "%Y-%m-%d %H:%M:%S"


class CreateInitGroup:
    def __init__(self, data, GroupSize):
        self.jobNum = data.jobNum
        self.operationNum = data.operationNum
        self.proMachineNum = data.proMachineNum
        self.jobProcess = data.jobProcess
        self.data = data
        self.waite_process = {}
        self.T = 0
        self.jobStartEnd = {}
        self.workTime = []
        self.oMPstatus = {}
        self.solutionSpace = {}
        self.GroupSize = GroupSize
        self.colors = []
        self.chromosome = {}
        self.recordTime = {}
        self.machineTime = {}

    def main(self):
        self.init()
        self.createGroup()

    def init(self):
        workTime = {}
        for i in range(self.proMachineNum):
            workTime[i] = 0
        self.workTime = workTime

        oMPstatus = {}
        for i in range(len(self.jobProcess)):
            job_id = self.jobProcess[i]['job_id']
            oMPstatus[job_id] = {}
            for k in range(len(self.jobProcess[i]['operations'])):
                operation_id = self.jobProcess[i]['operations'][k]['operation_id']
                oMPstatus[job_id][operation_id] = 0
        self.oMPstatus = oMPstatus
        self.jobStartEnd = {}
        self.recordTime = {}
        self.machineTime = {}
        self.chromosome = {'job': [], 'operation': [], 'machine': [], "operation_code": []}

    def randomLine(self, waitJobs, product_id):
        if len(waitJobs) > 0:
            job = random.choice(waitJobs)
            job_id = job['job_id']
            operation_id = job['operation_id']
            operation_code = job['operation_code']
            machine_id = random.choice(
                list(self.jobProcess[product_id]['process_job'][job_id]['operations'][operation_id]['machines'].keys()))
            self.createGene(product_id, job_id, operation_id, machine_id, operation_code)
            self.oMPstatus[product_id][job_id][operation_id] = 2
        else:
            print('no process')
            return

    def chooseBestLine(self, waitJobs, product_id, type):
        bigM = 100000000000000000
        workTimeList_init = {}
        for job in waitJobs:
            job_id = job['job_id']
            operation_id = job['operation_id']
            operation_code = job['operation_code']
            if product_id not in self.recordTime.keys():
                self.recordTime[product_id] = {}
            if job_id not in self.recordTime[product_id].keys():
                self.recordTime[product_id][job_id] = {}
            if operation_id not in self.recordTime[product_id][job_id].keys():
                self.recordTime[product_id][job_id][operation_id] = [0, 0]

            preJobEndTime = 0
            if operation_id > 0:
                preJobEndTime = self.recordTime[product_id][job_id][operation_id - 1][1]

            workTimeList = []
            workTimeList_LS = []
            workTimeList_GS = []
            machineList = []
            startTimeList = []
            endTimeList = []
            lineBestTimeList = []
            processTimeList = []
            for machine_id in self.jobProcess[product_id]["process_job"][job_id]['operations'][operation_id][
                'machines'].keys():
                processTime = \
                self.jobProcess[product_id]["process_job"][job_id]['operations'][operation_id]['machines'][machine_id]
                if machine_id not in self.machineTime.keys():
                    self.machineTime[machine_id] = []
                if machine_id not in workTimeList_init.keys():
                    workTimeList_init[machine_id] = 0
                availableTime = []

                if len(self.machineTime[machine_id]) == 0:
                    avail_startTime = 0
                    avail_endTime = bigM
                    availableTime.append([avail_startTime, avail_endTime])
                else:
                    for num in range(len(self.machineTime[machine_id])):
                        avail_startTime = self.machineTime[machine_id][num][1]
                        if num == len(self.machineTime[machine_id]) - 1:
                            avail_endTime = bigM
                        else:
                            avail_endTime = self.machineTime[machine_id][num + 1][0]
                        availableTime.append([avail_startTime, avail_endTime])

                        if num == len(self.machineTime[machine_id]) - 2:
                            availableTime.append([self.machineTime[machine_id][num + 1][1], bigM])
                lineBestTime = self.chooseLineBestTime(availableTime, preJobEndTime, processTime)
                machineList.append(machine_id)
                startTimeList.append(lineBestTime[0])
                endTimeList.append(lineBestTime[1])
                lineBestTimeList.append(lineBestTime)
                processTimeList.append(processTime)
                workTimeList.append(self.workTime[machine_id])
                workTimeList_GS.append(self.workTime[machine_id] + processTime)
                workTimeList_LS.append(workTimeList_init[machine_id] + processTime)

            if type == 0:
                machineIndex = endTimeList.index(min(endTimeList))
            elif type == 1:
                machineIndex = processTimeList.index(min(processTimeList))
            elif type == 2:
                machineIndex = startTimeList.index(min(startTimeList))
            elif type == 3:
                machineIndex = workTimeList.index(min(workTimeList))
            elif type == 4:
                machineIndex = random.randint(0, len(machineList) - 1)
            elif type == 5:
                machineIndex = workTimeList_GS.index(min(workTimeList_GS))
            elif type == 6:
                machineIndex = workTimeList_LS.index(min(workTimeList_LS))
            else:
                print("fail")

            machine_id = machineList[machineIndex]
            machineBestTime = lineBestTimeList[machineIndex]
            self.recordTime[product_id][job_id][operation_id] = machineBestTime

            self.workTime[machine_id] = self.workTime[machine_id] + processTimeList[machineIndex]
            workTimeList_init[machine_id] += processTimeList[machineIndex]
            if len(self.machineTime[machine_id]) == 0:
                self.machineTime[machine_id].append(machineBestTime)
            else:
                timeList_1 = []
                timeList_2 = []
                for j in range(0, len(self.machineTime[machine_id])):
                    if self.machineTime[machine_id][j][1] <= machineBestTime[0]:
                        timeList_1.append(self.machineTime[machine_id][j])
                    else:
                        timeList_2.append(self.machineTime[machine_id][j])
                self.machineTime[machine_id] = timeList_1 + [[machineBestTime[0], machineBestTime[1]]] + timeList_2
            self.createGene(product_id, job_id, operation_id, machine_id, operation_code)
            self.oMPstatus[product_id][job_id][operation_id] = 2
            a = 0

    def chooseLine_GS(self):
        num = 0
        workTimeList_init = {}
        jobMachine = {}
        while num < self.operationNum:
            waitJobs = []
            jobList = list(range(self.jobNum))
            for job_id in jobList:
                for operation_id in self.oMPstatus[job_id].keys():
                    if self.oMPstatus[job_id][operation_id] == 0:
                        operation_code = self.jobProcess[job_id]['operations'][operation_id][
                            'operation_code']
                        waitJobs.append(
                            {'job_id': job_id, 'operation_id': operation_id, 'operation_code': operation_code})
                        break

            jobID = []
            operationID = []
            operationCode = []
            machineID = []
            shortTime = []
            for job in waitJobs:
                job_id = job['job_id']
                operation_id = job['operation_id']
                operation_code = job['operation_code']
                if job_id not in jobMachine.keys():
                    jobMachine[job_id] = {}
                jobMachine[job_id][operation_id] = {'machine': None, 'processTime': None, 'finish': False}

                workTimeList = []
                machineList = []
                processTimeList = []
                for machine_id in self.jobProcess[job_id]['operations'][operation_id][
                    'machines'].keys():
                    processTime = \
                    self.jobProcess[job_id]['operations'][operation_id]['machines'][
                        machine_id]
                    if machine_id not in workTimeList_init.keys():
                        workTimeList_init[machine_id] = 0

                    machineList.append(machine_id)
                    processTimeList.append(processTime)
                    workTimeList.append(self.workTime[machine_id] + processTime)

                indexList = [index for index, value in enumerate(workTimeList) if value == min(workTimeList)]
                machineIndex = random.choice(indexList)

                machine_id = machineList[machineIndex]
                jobID.append(job_id)
                operationID.append(operation_id)
                machineID.append(machine_id)
                shortTime.append(min(workTimeList))
                operationCode.append(operation_code)

            indexList = [index for index, value in enumerate(shortTime) if value == min(shortTime)]
            index = random.choice(indexList)
            job_id = jobID[index]
            operation_id = operationID[index]
            machine_id = machineID[index]
            operation_code = operationCode[index]
            processTime = self.jobProcess[job_id]['operations'][operation_id]['machines'][
                machine_id]

            jobMachine[job_id][operation_id]['operation_code'] = operation_code
            jobMachine[job_id][operation_id]['machine'] = machine_id
            jobMachine[job_id][operation_id]['processTime'] = processTime

            self.workTime[machine_id] = self.workTime[machine_id] + processTime
            workTimeList_init[machine_id] += processTime
            self.oMPstatus[job_id][operation_id] = 2
            num += 1

        return jobMachine

    def chooseLine_random(self):
        num = 0
        jobMachine = {}
        while num < self.operationNum:
            waitJobs = []
            jobList = list(range(self.jobNum))
            while len(jobList) > 0:
                job_id = random.choice(jobList)
                for operation_id in self.oMPstatus[job_id].keys():
                    if self.oMPstatus[job_id][operation_id] == 0:
                        operation_code = self.jobProcess[job_id]['operations'][operation_id][
                            'operation_code']
                        waitJobs.append(
                            {'job_id': job_id, 'operation_id': operation_id, 'operation_code': operation_code})
                        if operation_id == len(self.jobProcess[job_id]['operations']) - 1:
                            jobList.remove(job_id)
                        self.oMPstatus[job_id][operation_id] = 2
                        break

            for job in waitJobs:
                job_id = job['job_id']
                operation_id = job['operation_id']
                operation_code = job['operation_code']
                if job_id not in jobMachine.keys():
                    jobMachine[job_id] = {}

                jobMachine[job_id][operation_id] = {'machine': None, 'processTime': None, 'finish': False}

                workTimeList = []
                machineList = []
                processTimeList = []
                for machine_id in self.jobProcess[job_id]['operations'][operation_id][
                    'machines'].keys():
                    processTime = \
                    self.jobProcess[job_id]['operations'][operation_id]['machines'][
                        machine_id]

                    machineList.append(machine_id)
                    processTimeList.append(processTime)
                    workTimeList.append(self.workTime[machine_id] + processTime)

                indexList = [index for index, value in enumerate(workTimeList) if value == min(workTimeList)]
                machineIndex = random.choice(indexList)
                machine_id = machineList[machineIndex]
                processTime = \
                self.jobProcess[job_id]['operations'][operation_id]['machines'][machine_id]

                jobMachine[job_id][operation_id]['operation_code'] = operation_code
                jobMachine[job_id][operation_id]['machine'] = machine_id
                jobMachine[job_id][operation_id]['processTime'] = processTime

                self.workTime[machine_id] = self.workTime[machine_id] + processTime
                num += 1

        return jobMachine

    def jobSequence(self, jobMachine, sequenceType):
        num = 0
        while num < self.operationNum:
            jobList = []
            for job_id in range(self.jobNum):
                for operation_id in jobMachine[job_id].keys():
                    if jobMachine[job_id][operation_id]['finish'] == False:
                        job = {'job_id': job_id, 'operation_id': operation_id,
                               'machine_id': jobMachine[job_id][operation_id]['machine'],
                               'operation_code': jobMachine[job_id][operation_id]['operation_code']}
                        if job not in jobList:
                            jobList.append(job)
                            break

            jobTime = [0 for _ in range(len(jobList))]

            operationNum = [0 for _ in range(len(jobList))]
            for i in range(len(jobList)):
                job_id = jobList[i]['job_id']
                for operation_id in jobMachine[job_id].keys():
                    if jobMachine[job_id][operation_id]['finish'] == False:
                        jobTime[i] += jobMachine[job_id][operation_id]['processTime']
                        operationNum[i] += 1

            if sequenceType == 0:
                job = random.choice(jobList)
            elif sequenceType == 1:
                indexList = [index for index, value in enumerate(jobTime) if value == max(jobTime)]
                jobIndex = random.choice(indexList)
                job = jobList[jobIndex]
            else:
                indexList = [index for index, value in enumerate(operationNum) if value == max(operationNum)]
                jobIndex = random.choice(indexList)
                job = jobList[jobIndex]

            self.createGene(job['job_id'], job['operation_id'], job['machine_id'], job['operation_code'])
            jobMachine[job['job_id']][job['operation_id']]['finish'] = True
            num += 1

    def chooseLineBestTime(self, availableTime, last_endTime, processTime):
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

    def judge(self, product_id):
        over = 1
        for job_id in self.oMPstatus[product_id].keys():
            for operation_id in self.oMPstatus[product_id][job_id].keys():
                if self.oMPstatus[product_id][job_id][operation_id] == 0:
                    over = 0
                    break
            if over == 0:
                break
        return over

    def createGroup(self):
        R1_num = paras.R_random * self.GroupSize
        R2_num = paras.R_GS * self.GroupSize

        for i in range(0, self.GroupSize):
            self.init()
            if i < R1_num:
                machineType = 0
                if i < R1_num * paras.S_random:
                    sequenceType = 0
                elif i < R1_num * (paras.S_random + paras.S_MOR):
                    sequenceType = 1
                else:
                    sequenceType = 2
            else:
                machineType = 1
                if i < R2_num * paras.S_random:
                    sequenceType = 0
                elif i < R2_num * (paras.S_random + paras.S_MOR):
                    sequenceType = 1
                else:
                    sequenceType = 2

            if machineType == 0:
                machineJobs = self.chooseLine_random()
            else:
                machineJobs = self.chooseLine_GS()
            self.jobSequence(machineJobs, sequenceType)
            self.solutionSpace[i] = self.chromosome

    def createGene(self, job_id, operation_id, machine_id, operation_code):
        self.chromosome["job"].append(job_id)
        self.chromosome["operation"].append(operation_id)
        self.chromosome["machine"].append(machine_id)
        self.chromosome["operation_code"].append(operation_code)


class Data:
    def __init__(self):
        self.jobNum = 0
        self.operationNum = 0
        self.proMachineNum = 0
        self.jobProcess = []
        self.xNum = 0

    def createJobs(self, content):
        contentList = content.strip().split('\n')
        baseData = contentList[0].split()
        jobNum = int(baseData[0])
        proMachineNum = int(baseData[1])
        self.jobNum = self.jobNum + jobNum
        self.proMachineNum = self.proMachineNum + proMachineNum

        operation_code = 0
        jobs = []
        for line in contentList[1:]:
            job_info = {}
            numbers = list(map(int, line.split()))
            job_id = len(jobs)
            job_info['job_id'] = job_id
            job_info['operations'] = []
            num_operations = numbers[0]
            index = 1

            for op_num in range(num_operations):
                operation = {}
                operation['operation_id'] = len(job_info['operations'])
                operation['operation_code'] = operation_code
                operation_code = operation_code + 1
                operation['machines'] = {}
                num_machines = numbers[index]
                index += 1
                for mc_no in range(num_machines):
                    machine_id = numbers[index]
                    process_time = numbers[index + 1]
                    operation['machines'][machine_id] = process_time
                    index += 2
                    self.xNum = self.xNum + 1
                job_info['operations'].append(operation)
                self.operationNum = self.operationNum + 1
            jobs.append(job_info)
        return jobs

    def readFile(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    def main(self, path):
        content = self.readFile(path)
        self.jobProcess = self.createJobs(content)

Data()

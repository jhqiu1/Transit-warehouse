from Parameter import Paras

paras = Paras()


class Prompts:
    def __init__(self):
        self.prompt_title = "Given information about jobs and machines, you need to determine the jobs for which it is necessary to change the sequence and the machine in order to minimize the makespan. "
        self.prompt_task = "In order to minimize the makespan, please design an algorithm to calculate the sequential priority and machine replacement priority for all jobs: "
        self.prompt_taskImprove = "In order to minimize the makespan, please design a new algorithm totally different from the above algorithms to calculate the sequential priority and machine replacement priority for all jobs: "
        self.prompt_heuristics = "Refer to some heuristics for solving flexible job-shop scheduling problem,such as 'Shortest Processing Time'. "
        self.prompt_template = "\
        1. First, describe your new algorithm and main steps in one sentence,The description must be inside within boxed {}. \n \
        2. Next, implement the following Python function: \
        def calculate_priority(job_totalTime: list, job_shortestTime: list, job_operationNumber: list, operation_preEndTime: list, operation_startTime: list,operation_processTime: list, operation_machineNumber: list)-> (list,list): \n \
        Job Args:\n \
        process_span: Current total time taken for each job. \n \
        minimal_process_span: Shortest processing time for each job. The value of job_shortestTime is less than or equal to the value of job_totalTime.\n \
        operation_number: Number of operations for each job. \n \
        Operation Args:\n \
        operation_startTime: Processing start time on the current machine for each operation. The value of operation_startTime is greater than or equal to the value of operation_preEndTime.\n \
        operation_earliestStartTime: Completion time of the previous operation for each operation. \n \
        operation_processTime: Processing time on the current machine for each operation.\n \
        operation_machineNumber: The number of optional machines for each operation.\n \
        Return: The types of all return variables must be a list, not an array.\n \
        jobPriority: Priority of processing sequence adjustment for each job. Length of jobPriority is job number.\n \
        operationPriority: Priority of the current machine being replaced by other machines for each operation. Length of operationPriority is operation number.\n \
        Note: Different list lengths for Job Args and Operation Args. The code cannot have a division by zero error. \n \
        Do not give additional explanations."
        self.prompt_improveTemplate = "\
        In order to minimize the makespan, please design a new algorithm totally different from the above algorithms to calculate the sequential priority and machine replacement priority for all jobs:"

    def prompt_init(self):
        prompt_content = (
            self.prompt_heuristics + self.prompt_task + "\n" + self.prompt_template
        )
        return prompt_content

    def prompt_improve(self, prompt_evaluation, prompt_title, popCharacter):
        prompt_evaluation = (
            "Given population evaluation, algorithm limitations, and suggestions for algorithm design are presented below: \n"
            + prompt_evaluation
        )
        prompt_content = (
            popCharacter
            + "\n"
            + prompt_evaluation
            + "\n"
            + "Your task:"
            + "\n"
            + prompt_title
            + "\n"
            + self.prompt_template
        )
        return prompt_content

    def prompt_popCharacter(self, parents, prompt_popCharacter):
        prompt = (
            prompt_popCharacter + "\n" + "Given " + str(len(parents)) + " algorithms: "
        )
        for i in range(len(parents)):
            prompt = (
                prompt
                + "\n"
                + str(i + 1)
                + ". "
                + "Algorithm: "
                + "(Successful evolution rate: "
                + str(round(parents[i]["success"] / parents[i]["run"], 7))
                + ")"
                + " "
                + parents[i]["algorithm"]
            )
        return prompt

    def prompt_indicate(self, populationCharacter):
        indicate = populationCharacter[-1]
        prompt_iterNum = (
            "1. The current iteration progress is "
            + str(indicate["nowIterNum"])
            + "/"
            + str(indicate["iterNum"])
            + "."
        )
        prompt_fitness = (
            "2. The minimum makespan is "
            + str(indicate["optimalFitness"]["value"])
            + " and the average makespan is "
            + str(indicate["meanFitness"]["value"])
            + "."
        )
        prompt_meanFitness = (
            "3. The average makespan change rate is "
            + str(indicate["meanFitness"]["change_rate"])
            + "."
        )
        prompt_optimalFitness = (
            "4. The minimum makespan change rate is "
            + str(indicate["optimalFitness"]["change_rate"])
            + "."
        )

        prompt = (
            "The characteristics of the population are as follows:"
            + "\n"
            + prompt_iterNum
            + "\n"
            + prompt_fitness
            + "\n"
            + prompt_meanFitness
            + "\n"
            + prompt_optimalFitness
        )
        return prompt

    def prompt_evaluation(self, popCharacter):
        prompt_task = "Please describe the optimization dilemma, limitations of existing algorithms, and give suggestions for designing a new algorithm. \n \
        1. First, characterize the evolution of populations in one sentence. \n \
        2. Describe the limitations of each algorithm in one sentence respectively. For example: First algorithm, <Limitation of the algorithm>. \n \
        3. Finally, a suggestion for designing a new calculate algorithm is given in one sentences. The algorithm calculates sequential priority and machine replacement priority for all jobs to minimize the makespan, according to processing order, start time and processing time for all jobs.\n \
        Do not give additional explanations."
        prompt = popCharacter + "\n" + prompt_task
        return prompt

    def prompt_newPrompt(self):
        prompt = (
            "Rewrite the following sentence without changing its original meaning and do not give additional explanations:"
            + "\n"
            + self.prompt_improveTemplate
        )
        return prompt

    def prompt_newTemplate(self, new_alg, template):
        """
        Generate a refined prompt template for multi-objective optimization algorithms.

        Args:
            new_alg (str): The advanced algorithm code for reference.
            template (str): The original prompt template to improve.

        Returns:
            str: A complete prompt template string with clear instructions.
        """
        # Core instruction with explicit requirements
        description = (
            "You are an AI Python expert specializing in multi-objective optimization. "
            "Your task is to refine the following prompt template to generate more advanced Python algorithms "
            "aiming to maximize Hypervolume (HV).\n\n"
            "REFERENCE ALGORITHM (Analyze its advantages for inspiration):\n"
            f"{new_alg}\n\n"
            "ORIGINAL TEMPLATE TO IMPROVE:\n"
            f"{template}\n\n"
        )

        # Clear formatting rules based on PEP 8 [1,3](@ref)
        formatting_rules = (
            "IMPROVEMENT REQUIREMENTS:\n"
            "1. Return a COMPLETE, ready-to-use Python function including:\n"
            "   - All necessary import statements\n"
            "   - Preserved function name, input parameters (with types), and return type\n"
            "   - Full function body implementation\n"
            "2. Apply PEP 8 formatting [1,3](@ref):\n"
            "   - 4-space indentation (no tabs)\n"
            "   - Maximum 79 characters per line\n"
            "   - Proper spacing around operators\n"
            "   - Snake_case for functions/variables, CamelCase for classes\n"
            "3. Correct any syntax errors in the original template\n"
            "4. Enhance the docstring under 'Args' with 2-3 specific improvement suggestions:\n"
            "   - Focus on HV optimization techniques\n"
            "   - Reference the reference algorithm's strengths\n"
        )

        # Output specification to prevent incomplete responses
        output_spec = (
            "OUTPUT FORMAT:\n"
            "Return ONLY the final improved template as a single string, formatted as:\n"
            "```python\n"
            "import necessary_libraries\n"
            "\n"
            "def function_name(param: type) -> return_type:\n"
            '    """\n'
            "    Enhanced function description.\n"
            "    \n"
            "    Args:\n"
            "        param (type): Description.\n"
            "        \n"
            "    Improvement Suggestions:\n"
            "        - Suggestion 1 based on reference algorithm\n"
            "        - Suggestion 2 for HV optimization\n"
            "        \n"
            "    Returns:\n"
            "        return_type: Description.\n"
            '    """\n'
            "    # Implemented function body\n"
            "    return result\n"
            "```\n"
            "No additional explanations or text outside the code block."
        )

        prompt_content = description + formatting_rules + output_spec
        return prompt_content

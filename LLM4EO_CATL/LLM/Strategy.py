import LLM.ApiGenerate as Generate
import LLM.Prompts as Prompts

class Strategy():

    def init(self,paras):
        prompts = Prompts.Prompts()

        prompt_content = prompts.prompt_init()

        state = False
        while state == False:
            generator = Generate.ApiGenerate(prompt_content,paras)
            [code_all, algorithm, state] = self.codeTest(generator)

        return [code_all, algorithm]

    def improve(self, parents, populationCharacter,paras):

        prompts = Prompts.Prompts()
        prompt_popCharacter = prompts.prompt_indicate(populationCharacter)
        popCharacter = prompts.prompt_popCharacter(parents, prompt_popCharacter)
        prompt_evaluation = prompts.prompt_evaluation(popCharacter)
        generator = Generate.ApiGenerate(prompt_evaluation,paras)
        evaluationRest = generator.runLLM()

        state = False
        while state == False:
            prompt_templateTask = prompts.prompt_newPrompt()
            generator = Generate.ApiGenerate(prompt_templateTask,paras)
            prompt_template = generator.runLLM()

            prompt_content = prompts.prompt_improve(evaluationRest, prompt_template, popCharacter)
            generator = Generate.ApiGenerate(prompt_content,paras)

            [code_all, algorithm, state] = self.codeTest(generator)

        return [code_all, algorithm]

    def codeTest(self, generator):
        state = True
        [code_all, algorithm] = generator.getStrategy()
        try:
            exec(code_all, globals())
            job_probability, machine_probability = globals()['calculate_priority']([1,1], [1,1], [1,1], [0,0], [1,1], [1,1], [1,1])

            if isinstance(job_probability, list) and isinstance(machine_probability, list) and len(
                    job_probability) > 0 and len(machine_probability) > 0:
                print("operator generate success")
            else:
                print("operator generate fail")
                state = False
        except Exception as e:
            print(f"operator generate fail: {e}")
            state = False
        return [code_all, algorithm, state]

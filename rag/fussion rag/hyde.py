from llm.llm_factory import get_llm


class HyDE:
    #Hypothetical Document Embeddings

    def __init__(self):

        self.llm = get_llm()

    def generate_hypothetical_answer(
        self,
        question
    ):

        prompt = f"""
Generate a short hypothetical answer
for the following question.

Question:
{question}
Hypothetical Answer:
"""
        
        return self.llm.generate(prompt)
    
#""""What is Clause 6.2(e)?" 
#    I don't know what this specific PDF says, but in 90% of corporate, legal, or medical documents,
#    a clause numbered like '6.2(e)' usually talks about policies, terms, conditions, or emergency protocols 
#    (hypothetical) answer:-
#      Clause 6.2(e) outlines the medical emergency policy, detailing the steps an employee must take during a health crisis.
#     @HyDE Query (LLM Guess)	@"Clause 6.2(e) outlines the medical emergency policy..."	
#     @ Succeeds often: The mathematical vector for this sentence is highly similar to the structure of the real clause in your hemant_story.pdf, even if the exact words differ slightly."""
        
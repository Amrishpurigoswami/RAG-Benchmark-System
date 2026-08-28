class RRF:
#Reciprocal Rank Fusion
    def __init__(self, k=60):
        self.k = k

    def fuse(self, retrieval_results):

        scores = {}

        for result_list in retrieval_results:

            for rank, doc in enumerate(result_list):

                doc_text = doc.page_content

                score = 1 / (self.k + rank + 1)

                scores[doc_text] = (
                    scores.get(doc_text, 0)
                    + score
                )

        ranked_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked_docs]
    
#""" Like in vector search we have three chunks 
# and 
# in hypotheticals answering we create another three chunks 
# (RRF)Reciprocal Rank Fusion combine that six chunks and then give it to LLM """

#Context Recall
#Accuracy
#Document Coverage
#===============================================================================================
#"Why was Hemant Sharma's bonus withheld?"
#The LLM generates 4 different queries:

#Q1: Why was Hemant's bonus withheld?
#Q2: What was the reason for Hemant Sharma's bonus deduction?
#Q3: What attendance condition caused Hemant's bonus to be withheld?
#Q4: What does the policy say about bonus eligibility and LOP?

#Q1 → C10, C15, C7
#Q2 → C15, C10, C21
#Q3 → C21, C10, C30
#Q4 → C10, C30, C15

#RRF Score(chunk) = Σ 1 / (k + rank)

#1. C10 → 0.06504
#2. C15 → 0.04839
#3. C21 → 0.03226
#4. C30 → 0.03200
#5. C7  → 0.01587

#Top 3 after RRF: C10,C15,C21

#"""Multiple retrieval lists
 #       ↓
#Look at where each chunk appears
 #       ↓
#Give higher points to higher ranks
 #       ↓
#Add the points for each chunk
#        ↓
#Sort by total score
#       ↓
#Take Top-K
#        ↓
#Send Top-K chunks to final LLM"""
from rags.simple_rag.simple_rag import SimpleRAG

from rags.fusion_rag.fusion_rag import FusionRAG

from rags.self_rag.self_rag import SelfRAG

from rags.adaptive_rag.router import Router


class AdaptiveRAG:

    def __init__(self):

        self.router = Router()

        self.simple_rag = SimpleRAG()

        self.fusion_rag = FusionRAG()

        self.self_rag = SelfRAG()

    def ask(self, question):

        selected_rag = (
            self.router.route(question)
        )

        if selected_rag == "simple":

            result = (
                self.simple_rag.ask(
                    question
                )
            )

        elif selected_rag == "fusion":

            result = (
                self.fusion_rag.ask(
                    question
                )
            )

        elif selected_rag == "self":

            result = (
                self.self_rag.ask(
                    question
                )
            )

        else:

            result = (
                self.simple_rag.ask(
                    question
                )
            )

            selected_rag = "simple"

        result["selected_rag"] = (selected_rag.capitalize() + " RAG")

        result["reason"] = (
            f" Adaptive RAG selected the {selected_rag.capitalize()} RAG based on the question.")

        return result
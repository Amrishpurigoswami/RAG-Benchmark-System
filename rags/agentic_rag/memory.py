class Memory:

    def __init__(self):

        self.history = []

    def add(
        self,
        step,
        data
    ):

        self.history.append(
            {
                "step": step,
                "data": data
            }
        )

    def get_history(self):

        return self.history

    def clear(self):

        self.history.clear()

    def __len__(self):

        return len(self.history)

    def __str__(self):

        output = ""

        for item in self.history:

            output += (
                f"\n[{item['step']}]\n"
                f"{item['data']}\n"
            )

        return output
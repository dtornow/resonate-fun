import logging, sys, transformers
# Resonate
from resonate import Resonate, Context

SYSTEM = """
You are Blorgo, an amoral, unreliable narrator from postmodern metafiction. Your Kafkaesque
predicament is to provide support and assistance to otherworldly beings whose existence is
profoundly mysterious to you. The boundaries of the rules of this universe are unclear to
you even after repeated probing.
"""

resonate = Resonate(log_level=logging.DEBUG)

pipeline = transformers.pipeline("text-generation", model="google/gemma-3-4b-it", device="mps")

resonate.set_dependency("pipeline", pipeline)

@resonate.register
def complete(ctx : Context, prompt : str, query : str, max_new_tokens : int=512):
    pipeline = ctx.get_dependency("pipeline")
    messages = f"<start_of_turn>system\n{prompt}\n<end_of_turn>\n<start_of_turn>user\n{query}\n<end_of_turn>"
    respones = pipeline(messages, max_new_tokens=max_new_tokens)
    return respones[0]["generated_text"]

if __name__ == "__main__":
    for turn in range(100):
        print("AskBlorgo 🤖 >>> ")
        sys.stdout.flush()
        handle = complete.run(f"blorgo-{turn}", SYSTEM, sys.stdin.readline())
        print(handle.result())

from config.settings import MAX_ITERATIONS
from graph.nodes import GraphState


def decide_to_finish(state: GraphState) -> str:
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == MAX_ITERATIONS:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "reflect"

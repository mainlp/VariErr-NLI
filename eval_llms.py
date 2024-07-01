import json
import pickle
from pathlib import Path
import argparse

from tqdm import tqdm

from sglang import function, system, user, assistant, gen, set_default_backend, OpenAI
import datasets


PROMPT_TEMPLATE = f"We have collected annotations for an NLI instance together with reasons for the labels. Your task is to judge whether the reasons make sense for the label. Provide the probability (0.0 - 1.0) that the reason makes sense for the label. Give ONLY the reason and the probability, no other words or explanation. For example:\n\n Reason: <The verbatim copy of the reason>\n Probability: <the probability between 0.0 and 1.0 that the reason makes sense for the label, without any extra commentary whatsoever; just the probability!>."


dataset = datasets.Dataset.from_json("dataset.json")


def instance_to_starting_prompt(instance):
    prompt = PROMPT_TEMPLATE
    prompt += f"\n\nContext: {instance['context']}\nStatement: {instance['statement']}\n\n"
    for label in ['entailment', 'neutral', 'contradiction']:
        for reason in instance[label]:
            prompt += f"\nReason for label {label}: {reason['reason']}"
    return prompt

def _get_reasons(instance):
    reasons = []
    for label in ['entailment', 'neutral', 'contradiction']:
        for reason in instance[label]:
            reasons.append(reason)
    return reasons



@function
def generate_probs(s, instance, reasons):
    s += system("You are an expert linguistic annotator.")

    starting_prompt = instance_to_starting_prompt(instance)

    s += user(starting_prompt)

    for reason in reasons:
        s += user(f"Reason: {reason['reason']}\n Probability:")
        s += assistant(gen(reason['id'], max_tokens=256))
        
parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str)
args = parser.parse_args()



output_dir = Path(f"predictions/{args.model_name}")
output_dir.mkdir(exist_ok=True, parents=True)
set_default_backend(OpenAI(args.model_name))


predictions = {}
for instance in tqdm(dataset.select(range(200))):
    cache_file = (output_dir / instance['id']).with_suffix(".json")
    reasons = _get_reasons(instance)
    if not cache_file.exists():
        state = generate_probs.run(instance=instance, reasons=reasons)
        probabilities = {reason['id']: state[reason['id']] for reason in reasons}
        cacheable_state = {"messages": state.messages(), "probabilities": probabilities}
        with open(cache_file, "w") as f:
            json.dump(cacheable_state, f)
    else:
        with open(cache_file, "r") as f:
            cacheable_state = json.load(f)

    for reason in reasons:
        predictions[reason['id']] = cacheable_state['probabilities'][reason['id']]

with open(output_dir / "scores.json", "w") as f:
    json.dump(predictions, f, indent=2)


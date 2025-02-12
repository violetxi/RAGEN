import re
import datasets
import numpy as np
import pandas as pd
from typing import List

STEP_TOKEN = " \n\n"

def merge_datasets(dataset_list):
    ds_list = [datasets.load_dataset(dataset_name, split="train") for dataset_name in dataset_list]    
    return datasets.concatenate_datasets(ds_list)


def remove_content_after_boxed_solutions(sample):
    clean_rollouts = []
    for rollout in sample["tir_seed_rollouts"]:
        text = ""
        sol_steps = rollout.split(STEP_TOKEN)
        for sol_step in sol_steps:
            text += sol_step.strip() + STEP_TOKEN
            if "boxed{" in sol_step:
                break
        clean_rollouts.append(text)
    sample["tir_seed_rollouts"] = clean_rollouts
    return sample


def add_num_python_blocks(sample):
    pattern = r"```python\s+.*?```"
    output_pattern = r"```output\s+.*?```"
    rollouts = []
    is_correct = []
    num_code_blocks = []
    num_output_blocks = []

    for text, correct in zip(sample["tir_seed_rollouts"], sample["is_correct"]):
        code_blocks = re.findall(pattern, text, re.DOTALL)
        output_blocks = re.findall(output_pattern, text, re.DOTALL)
        # model does random stuff sometimes.. let's filter out the insane behaviors
        if len(code_blocks) == len(output_blocks):
            rollouts.append(text)
            is_correct.append(correct)
            num_code_blocks.append(len(code_blocks))
            num_output_blocks.append(len(output_blocks))

    sample["is_correct"] = is_correct
    sample["tir_seed_rollouts"] = rollouts    
    sample["num_python_blocks"] = num_code_blocks
    sample["num_output_blocks"] = num_output_blocks
    return sample


positive_feedbacks  = [
    "This answer is correct!",    
    "Looks good, this answer is correct!",
    "This is the correct answer!",
    "Let's verify the answer, this is correct!",
]
negative_feedbacks = [
    "This answer is incorrect, let me try again.",
    "Wait, this is not correct, let me try again.",
    "This is not the correct answer, let me try again.",
    "Something is wrong, this is not the correct answer.",
]

def segment_text(text, is_solution=False):
    segments = []
    pattern = re.compile(r'(```python\n.*?\n```|```output.*?```)', re.DOTALL)
    matches = pattern.split(text)    
    
    for segment in matches:
        if segment.startswith('```python'):
            segments.append({'type': 'python', 'content': segment[9:-3].strip()})            
        elif segment.startswith('```output'):
            segments.append({'type': 'output', 'content': segment[9:-3].strip()})
        else:
            clean_segment = segment.strip()
            if clean_segment:
                if "boxed{" in clean_segment and is_solution:
                    segments.append({'type': 'solution', 'content': clean_segment})
                else:
                    segments.append({'type': 'text', 'content': clean_segment})
    
    return segments


def add_tags_to_trajectory(trajectory: List):
    tagged_trajectory = ""
    i = 0
    while i < len(trajectory):
        content = trajectory[i]["content"]
        traj_type = trajectory[i]["type"]
        if traj_type == "text":
            if "boxed{" not in content:
                tagged_trajectory += f"<think> {content} </think> \n\n"
                i += 1
            else:    # this is when it proposed a solution but not correct, need to add critic text
                try:
                    assert trajectory[i+1]["type"] == "text"
                except:
                    ttype = trajectory[i+1]["type"]
                    print(f"Critic content is {ttype}")
                critic_content =  trajectory[i+1]["content"]
                tagged_trajectory += f"<think> {content} {critic_content} </think> \n\n"                
                i += 2
        elif traj_type == "python":
            tagged_trajectory += f"<act> {content} </act> \n\n"
            i += 1
        elif traj_type == "output":
            tagged_trajectory += f"<obs> {content} </obs> \n\n"
            i += 1
        elif traj_type == "solution":
            tagged_trajectory += f"<ans> {content} </ans> \n\n"
            i += 1
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
        
    return tagged_trajectory


def create_trajectory_data(sample, max_iter):
    rollouts = sample["tir_seed_rollouts"]
    is_correct = sample["is_correct"]
    num_python_blocks = sample["num_python_blocks"]
    num_output_blocks = sample["num_output_blocks"]
    assert len(rollouts) == len(is_correct) == len(num_python_blocks) == len(num_output_blocks), "Lengths do not match"
    
    inds = np.arange(len(rollouts))    
    chosen_inds = np.random.choice(inds, max_iter, replace=False)
    
    # generate trajectory with critique     
    trajectory = []
    for i, ind in enumerate(chosen_inds):
        correct = is_correct[ind]
        rollout = rollouts[ind]        
        # n_python = num_python_blocks[ind]
        # n_output = num_output_blocks[ind]

        if i == max_iter - 1:    # solution needs to be correct at the last iteration
            if correct:
                # trajectory.append(rollout)
                segments = segment_text(rollout, is_solution=True)
                trajectory.extend(segments)                
            else:    # if incorrect at the last iteration, find a correct one
                correct_inds = [item[0] for item in np.where(is_correct)]
                correct_ind = np.random.choice(correct_inds, 1)[0]
                rollout = rollouts[correct_ind]

                segments = segment_text(rollout, is_solution=True)  
                trajectory.extend(segments)                              
        else:
            if correct:    # positive feedback
                segments = segment_text(rollout, is_solution=True)
                trajectory.extend(segments)
                break
            else:    # negative feedback
                segments = segment_text(rollout)
                trajectory.extend(segments)
                if "boxed{" in segments[-1]["content"]:
                    feedback = np.random.choice(negative_feedbacks)
                else:
                    feedback = "Hmmm.. I'm not sure about this output, let me try again."
                trajectory.append({"type": "text", "content": feedback})
    
    tagged_trajectory = add_tags_to_trajectory(trajectory)    
    sample["full_trajectory"] = tagged_trajectory
    sample["num_attempts"] = i + 1    
    return sample
    


def main():
    max_length = 8000
    dataset_list = ["RLAIF/CS-PRM-Seed-Rollouts-10K", "RLAIF/CS-PRM-Seed-Rollouts-20K"]
    ds = merge_datasets(dataset_list)    
    ds = ds.map(remove_content_after_boxed_solutions, num_proc=128)    
    ds = ds.map(add_num_python_blocks, num_proc=128)
    ds = ds.filter(lambda x: any(x["is_correct"]) and not all(x["is_correct"]))
    num_unique_problems = len(ds.unique("problem"))
    print(f"A total of {num_unique_problems} samples are in the dataset with at least one correct and one incorrect sample.")

    # create trajectory data
    max_iter = 2
    ds = ds.map(create_trajectory_data, num_proc=1, fn_kwargs={"max_iter": max_iter})
    ds = ds.filter(lambda x: len(x["full_trajectory"]) <= max_length)
    num_unique_problems = len(ds.unique("problem"))
    print(f"A total of {num_unique_problems} samples are in the dataset")    
    ds.push_to_hub(f"RLAIF/Tool-SFT-iter{max_iter}-Raw", private=True)


if __name__ == "__main__":
    main()
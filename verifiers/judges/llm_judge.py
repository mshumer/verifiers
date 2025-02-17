import re
import os
import json
import requests
from typing import List, Any

def llm_judge_reward_func(
    completions: List[Any],
    answer: List[Any],
    judge_prompt_template: str = (
        "Judge the following response for quality.\n"
        "Prompt: {prompt}\n"
        "Response: {response}\n"
        "\nEvaluation criteria:\n"
        "- Correctness and accuracy of the response\n"
        "- Clarity and coherence\n"
        "- Completeness in addressing the prompt\n"
        "- Appropriate level of detail\n"
        "\nProvide your judgment as a single float between 0.0 and 1.0 where:\n"
        "0.0 = completely incorrect or irrelevant\n"
        "0.3 = major issues or gaps\n"
        "0.5 = partially correct with significant room for improvement\n"
        "0.7 = mostly correct with minor issues\n"
        "1.0 = excellent, complete and accurate\n"
        "\nOutput only the float number, nothing else."
    ),
    **kwargs
) -> List[float]:
    """
    Reward function that uses Claude-3.5-Sonnet via OpenRouter as a judge to evaluate generated responses.
    For each trajectory in completions, it extracts the prompt and the assistant's response,
    constructs a judging prompt, and queries Claude to produce a quality score.

    Args:
        completions: A list of trajectories (each trajectory is a list of message dicts).
        answer: The ground truth answers (unused by this judge but kept for compatibility).
        judge_prompt_template: A template to format the judge prompt.
        **kwargs: Additional arguments.

    Returns:
        A list of float scores (one per trajectory) between 0 and 1.
    """
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    scores = []
    for traj in completions:
        prompt_text = ""
        response_text = ""
        # Extract the latest user prompt and the last assistant response
        for msg in traj:
            if msg["role"] == "user":
                prompt_text = msg["content"]
            elif msg["role"] == "assistant":
                response_text = msg["content"]
        
        # Construct the judge prompt using the provided template
        judge_input = judge_prompt_template.format(prompt=prompt_text, response=response_text)
        
        data = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [
                {"role": "system", "content": "You are an objective judge. Follow the instructions carefully, and output only a single float number between 0.0 and 1.0 representing your judgment score."},
                {"role": "user", "content": judge_input}
            ]
        }

        # Query Claude via OpenRouter
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )

        print('Claude response:', response.json()['choices'][0]['message']['content'])
        
        if response.status_code == 200:
            judge_output = response.json()["choices"][0]["message"]["content"].strip()
            # Extract a floating-point number from the judge's response
            match = re.search(r"([0-1](?:\.\d+)?)", judge_output)
            if match:
                try:
                    score = float(match.group(1))
                except Exception:
                    score = 0.0
            else:
                score = 0.0
        else:
            score = 0.0
            
        scores.append(score)
    return scores
"""Dataset generation pipeline for evaluation subsets.

Generator-Evaluator architecture:
    - DeepSeek generates draft items (questions, adversarial inputs, etc.)
    - 5-model panel validates via majority voting (≥4/5 at temp 0.0)
    - Researcher verifies final output

Requires OpenRouter API keys.
"""

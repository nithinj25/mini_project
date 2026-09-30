"""Benchmark workloads (SPEC §9): 6 prompts each for code, maths and general chat."""

PROMPTS = {
    "code": [
        "Write a Python function that returns the n-th Fibonacci number using memoization. Include a docstring.",
        "Implement binary search over a sorted list in Python and explain its time complexity.",
        "Write a Python class for a stack with push, pop, peek and is_empty methods.",
        "Write a SQL query that returns the top 5 customers by total order value from tables customers(id, name) and orders(id, customer_id, amount).",
        "Write a JavaScript function that debounces another function by a given number of milliseconds.",
        "Write a Python function that checks whether a string is a valid palindrome, ignoring case and non-alphanumeric characters.",
    ],
    "math": [
        "Solve for x: 3x + 7 = 22. Show each step.",
        "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Explain your reasoning.",
        "Find the derivative of f(x) = x^3 * sin(x). Show the steps.",
        "What is the sum of the first 50 positive integers? Explain the formula you use.",
        "A rectangle has a perimeter of 36 cm and its length is twice its width. Find its area step by step.",
        "Compute the probability of getting exactly two heads in four fair coin flips. Show the working.",
    ],
    "chat": [
        "Explain why the sky appears blue during the day.",
        "Give me three tips for staying focused while studying for exams.",
        "What are the main differences between a virus and a bacterium?",
        "Write a short, friendly email inviting a colleague to a team lunch on Friday.",
        "Summarise the causes of the French Revolution in a few sentences.",
        "What should I consider when choosing my first programming language?",
    ],
}

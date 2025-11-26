Bigram Language Model — Simple Explanation

This project builds a very basic character-level language model using PyTorch and the Tiny Shakespeare dataset.

📌 What This Model Does

A Bigram Language Model looks at one character and predicts the next character — nothing more.

It does not remember earlier characters

It does not understand words or meaning

It only learns patterns of character pairs (e.g., th, he, er, etc.)

For example, if the current character is 'h', the model tries to guess which character usually appears after 'h' in the training text.

🎯 Why This Model Is Useful

Even though it's simple, it helps explain:

How text is turned into numbers

How a model can learn small patterns

How language models generate text step by step

This is the easiest starting point for understanding how bigger language models work internally.

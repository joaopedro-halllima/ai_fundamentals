from collections import Counter

text = "the cat sat on the mat the cat ate the fish"
tokens = text.split()

# Count what follows the token "the"
followers = Counter()

for prev, nxt in zip(tokens, tokens[1:]):
    if prev == "the":
        followers[nxt] += 1

total = sum(followers.values())
for token, count in followers.most_common():
    print(f"P({token} | the) = {count}/{total} = {count / total:.2f}")
    
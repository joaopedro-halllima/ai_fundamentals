from collections import Counter, defaultdict

CORPUS = (
    "the client asked for a summary of the contract . "
    "the client asked for a summary of the report . "
    "the client asked for a refund . "
    "the model wrote a summary of the contract . "
    "the model wrote a poem ."
)

def tokenize(text: str) -> list[str]:
    # Splitting the text into lowercase word tokens, "the" and "The" count as the same token
    return text.lower().split()
    '''
    print("The Client".lower())        # the client
    print("the client asked".split())  # ['the', 'client', 'asked']
    '''

def train(tokens: list[str]) -> dict[str, Counter]:
    counts = defaultdict(Counter)
    for prev, next in zip(tokens, tokens[1:]):
            counts[prev][next] += 1
    return counts

'''
model = train(tokenize(CORPUS))
print(model["the"])
'''

def next_token_probs(model: dict[str, Counter], prev: str) -> dict[str, float]:
    # Turning the count into probability
    if prev not in model:
        return {} # If we never seen the word return nothing
    counts = model[prev] # Get a tally for a word
    total = sum(counts.values()) #
    probs = {}
    for token, count in counts.items():
         probs[token] = count / total # For each word in the tally, store the count divided by the total to get the prob
    return probs

'''
print(next_token_probs(model, "the"))
print(next_token_probs(model, "of"))
print(next_token_probs(model, "invoice"))
'''

def generate(model, start: str, max_tokens: int = 8) -> list[str]:
    # greedy loop; break ties alphabetically; stop after "." or if no prediction
    out = [start]
    for _ in range(max_tokens): # Repeat up to the max_tokens time
        probs = next_token_probs(model, out[-1])
        if not probs:
             break # Stop is menu is empty

        # We have to pick the best word
        best = None
        for token in sorted(probs): 
             # Best is the best so far, it asks if this word has a higher chance then my current winner
             if best is None or probs[token] > probs[best]: 
                  best = token

        out.append(best) # Then add it to the sentence and stop if it ends it
        if best == ".":
             break

    return out

if __name__ == "__main__":
    model = train(tokenize(CORPUS))
    for prev in ["the", "a", "of"]:
        probs = next_token_probs(model, prev)
        ranked = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
        print(prev, "->", [(t, round(p, 2)) for t, p in ranked])
    print(" ".join(generate(model, "the")))
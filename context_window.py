def estimate_tokens(text):
    # Rough guess that 1 token is about 4 characters, max(1, ...) makes sure even a tiny string costs at least 1 token.
    return max(1, len(text) // 4)

'''
def fit_to_window(messages, window, reserve_for_output):
    # As previously learned, input and output share the window, so setting aside room for reply first.
    budget = window - reserve_for_output
    kept, used = [], 0 # messages we keep, tokens spent so far
    for msg in reversed(messages): # From newest to oldest
        cost = estimate_tokens(msg)
        if used + cost > budget: # If the message would overflow the budget
            break
        kept.append(msg) # If it fits keep it
        used += cost # And pay for it

    # Kept is newest first, so flip it back to normal reading order
    return list(reversed(kept)), used
'''

def fit_to_window(messages, window, reserve_for_output):
    budget = window - reserve_for_output
    first = messages[0]                     # the message we always keep
    used = estimate_tokens(first)           # its cost is already spent
    kept = []
    for msg in reversed(messages[1:]):      # everything except the first, newest first
        cost = estimate_tokens(msg)
        if used + cost > budget:
            break
        kept.append(msg)
        used += cost
    kept.append(first)                      # oldest goes last in the newest-first list...
    return list(reversed(kept)), used       # ...so the flip puts it at the front

history = [
    "My order number is 48213 and it arrived damaged.",
    "Sorry to hear that. Can you describe the damage?",
    "The screen is cracked in the top left corner.",
    "Thanks. Would you like a refund or a replacement?",
    "A replacement please. What was my order number again?",
]

# Window of 60 tokens, 20 held bac for the answer, so 40 are left for the input
kept, used = fit_to_window(history, window=200, reserve_for_output=20)
print(f"kept {len(kept)} of {len(history)} messages, {used} tokens")
for m in kept:
    print("-", m)
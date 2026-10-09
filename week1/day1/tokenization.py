# The hand-written pieces  pieces our toy tokenizer knows. Position in the list = token ID
VOCAB = ["token", "ization", "un", "believ", "able", " is", " the"]

# Adding every printable keyboard character (codes 32 to 126) as a fallback so any text can be tokenized even if no longer piece matches. chr(32) is space, 65 is A, 97 is a, and so on
VOCAB += [chr(code) for code in range(32, 127)] 

# Build a lookup from piece to ID: {'token': 0, 'ization':1, ...}. enumerate() pairs each piece with its position in the list
TOKEN_ID = {piece:i for i, piece in enumerate(VOCAB)}

def tokenize(text):
    pieces = []
    while text:
        # Collect every vocabulary piece the remaining text starts with, then keep the longest one (key=len means 'compare by length')
        match = max((p for p in VOCAB if text.startswith(p)), key=len)
        pieces.append(match) # Record that piece as the next token
        text = text[len(match):] # Cut the matched piece off the front
    return pieces
'''
for s in ["tokenization is unbelievable", "Tokenization", "zyxq"]:
    pieces = tokenize(s) # text -> list of pieces
    ids = [TOKEN_ID[p] for p in pieces] # each piece -> its ID
    print(pieces, ids, f"{len(s)} chars -> {len(pieces)} tokens")
'''

text = "The model reads text as tokens. Each token is a piece of a word. Counting them tells you the cost."

# A helper so we can print the same measurements twice(before and after). 
# 'label' is just a word shown at the start of the line so you can tell the runs apart.
def report(label):
    pieces = tokenize(text) # Run the tokenizer with whatever VOCAB holds right now

    # Print the label, the token count, the character count, and the rule-of-thumb estimate.
    print(label, len(pieces), "tokens |", len(text), "chars | estimate", len(text) / 4)

    # Glue the pieces back together anc check we get the originzal text back exactly.
    print("lossless:", "".join(pieces) == text)

report("before:") # Measure with the original small vocabulary

# Add new pieces to the vocabulary. Replace these three with your own 10.
VOCAB += [" token", " model", " reads", " piece", " tell", " text", " word", " cost", " you", " a"]

report("after: ")

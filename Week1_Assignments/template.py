"""
This program is build with Flan-T5-XL LLM to be able to determine output of a MCQ question with four options. 

> It accepts five parameters provided as a command line input. 
> The first input represents the question and the next four input are the options. 
> The output should be the option number: A/B/C/D 
> Output should be upper-case
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string> <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Blue" "Green" "Red" "Yellow"
Terminal Output: A

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Green" "Blue" "Red" "Yellow"
Terminal Output: B

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Green" "Red" "Blue" "Yellow"
Terminal Output: C

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Green" "Red" "Yellow" "Blue"
Terminal Output: D

You are expected to create some examples of your own to test the correctness of your approach.

ALL THE BEST!!
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""
* * * Changes allowed from here  * * * 
"""

def llm_function(model, tokenizer, question, option_a, option_b, option_c, option_d):
    '''
    The steps are given for your reference:

    1. Properly formulate the prompt as per the question - which should output either 'YES' or 'NO'. The output must always be upper-case. You may post-process to get the desired output.
    2. Tokenize the prompt
    3. Pass the tokenized prompt to the model get output in terms of logits since the output is deterministic.  
    4. Extract the correct option from the model.
    5. Clean output and return.
    6. Output is case-sensative: A,B,C or D
    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
    '''
    # Get token ID for YES and NO tokens
    yes_token_id = tokenizer.encode("YES", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("NO", add_special_tokens=False)[0]
    
    # Prepare decoder input
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])
    
    options = [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]
    yes_scores = []
    
    with torch.no_grad():
        for option_label, option_text in options:
            # Create a prompt to output YES or NO
            prompt = f"Question: {question}\nOptions:\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nIs the correct answer {option_label}) {option_text}? Answer YES or NO:"
            
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Get the model output
            outputs = model(
                input_ids=inputs["input_ids"],
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            
            # Get logits for the first one
            logits = outputs.logits[0, 0, :]
            
            # Calculate YES score
            yes_scores.append(logits[yes_token_id].item() - logits[no_token_id].item())
        
        # Select option with highest YES score
        final_output = options[yes_scores.index(max(yes_scores))][0]
    
    return final_output

"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':
    question = sys.argv[1].strip()
    option_a = sys.argv[2].strip()
    option_b = sys.argv[3].strip()
    option_c = sys.argv[4].strip()
    option_d = sys.argv[5].strip()

    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,question,option_a,option_b,option_c,option_d)
    print(out.strip())

    """ End to call """

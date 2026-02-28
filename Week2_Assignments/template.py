"""
This program is build with Flan-T5-XL LLM to be able to answer the final question using the output from the previous questions as in-context learning/few-shot learning. 

Consider three related questions from a search session: Question 1, Question 2, Question 3
1. Answer to Question 1 needs to be generated. 
2. Answer to Question 2 needs to be generated with the answer to Question 1 as one-shot example / context. 
3. Answer to Question 3 needs to be generated with the answer to Question 2 as one-shot example / context.
4. Answer to Question 3 will be either YES or NO and nothing else.


> The program accepts three parameters provided as a command line input. 
> The three inputs represent the questions.
> The output of the first two question is Generation based whereas the last question output is deterministic i.e. its either YES or NO.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in America?"
Terminal Output: NO

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in India?"
Terminal Output: YES

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

def llm_function(model,tokenizer,questions):
    '''
    The steps are given for your reference:

    1. Generate answer for the first question.
    2. Generate answer for the second question use the answer for first question as context.
    3. Generate a deterministic output either 'YES' or 'NO' for the third question using the context from second question.  
    5. Clean output and return.
    6. Output is case-sensative: YES or NO
    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
    '''
    # Step 1: Generate answer for the first question
    q1 = questions[0]
    prompt_q1 = f"Question: {q1}\nAnswer:"
    inputs_q1 = tokenizer(prompt_q1, return_tensors="pt")
    
    with torch.no_grad():
        outputs_q1 = model.generate(
            inputs_q1["input_ids"],
            max_length=100
        )
    
    answer_q1 = tokenizer.decode(outputs_q1[0], skip_special_tokens=True).strip()
    
    # Step 2: Generate answer for the second question using Q1 answer as one-shot context
    q2 = questions[1]
    prompt_q2 = f"Question: {q1}\nAnswer: {answer_q1}\n\nQuestion: {q2}\nAnswer:"
    inputs_q2 = tokenizer(prompt_q2, return_tensors="pt")
    
    with torch.no_grad():
        outputs_q2 = model.generate(
            inputs_q2["input_ids"],
            max_length=100
        )
    
    answer_q2 = tokenizer.decode(outputs_q2[0], skip_special_tokens=True).strip()
    
    # Step 3: Generate YES/NO for the third question using Q2 answer as context
    q3 = questions[2]
    # Format prompt to encourage YES/NO answer with context from Q2
    prompt_q3 = f"Question: {q2}\nAnswer: {answer_q2}\n\nQuestion: {q3}\nAnswer:"
    inputs_q3 = tokenizer(prompt_q3, return_tensors="pt")
    
    with torch.no_grad():
        # Generate answer for Q3
        outputs_q3 = model.generate(
            inputs_q3["input_ids"],
            max_length=10
        )
    
    generated_text = tokenizer.decode(outputs_q3[0], skip_special_tokens=True).strip().upper()
    
    # Extract YES or NO from generated text
    # Use regex to find YES or NO in the output
    yes_match = re.search(r'\bYES\b', generated_text)
    no_match = re.search(r'\bNO\b', generated_text)
    
    if yes_match and no_match:
        # If both found, use the one that appears first
        if generated_text.find("YES") < generated_text.find("NO"):
            final_output = "YES"
        else:
            final_output = "NO"
    elif yes_match:
        final_output = "YES"
    elif no_match:
        final_output = "NO"
    else:
        # Fallback: use logits to determine YES/NO deterministically
        yes_token_ids = tokenizer.encode("YES", add_special_tokens=False)
        no_token_ids = tokenizer.encode("NO", add_special_tokens=False)
        
        decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])
        
        with torch.no_grad():
            outputs_logits = model(
                input_ids=inputs_q3["input_ids"],
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            
            logits = outputs_logits.logits[0, 0, :]
            
            # Calculate scores for YES and NO (handle multi-token case)
            yes_score = sum(logits[token_id].item() for token_id in yes_token_ids) / len(yes_token_ids)
            no_score = sum(logits[token_id].item() for token_id in no_token_ids) / len(no_token_ids)
            
            final_output = "YES" if yes_score > no_score else "NO"
    
    return final_output

"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':

    question_a = sys.argv[1].strip()
    question_b = sys.argv[2].strip()
    question_c = sys.argv[3].strip()

    questions = [question_a, question_b, question_c]
    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,questions)
    print(out.strip())

    """ End to call """

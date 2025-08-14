from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_open_cases():
    assert query_and_validate(
        question="How many cases are open?",
        expected_response="4"
    )

def test_disorderly_conduct():
    assert query_and_validate(
        question="How many disorderly conduct incidents and on which time were reported?",
        expected_response="3 and time: 15:19, 14:16, 11:40"
    )

def test_burglary_attempts():
    assert query_and_validate(
        question="How many burglary attempt(s) and where?",
        expected_response="FLOWER STREET STRUCTURE – on campus"
    )

def test_smoke_smell():
    assert query_and_validate(
        question="From where was the smell of smoke coming?",
        expected_response="the smell of the smoke was coming from ELAINE STEVELY HOFFMAN MEDICAL RESEARCH CENTER"
    )

def test_alcohol_cases():
    assert query_and_validate(
        question="Any alcohol cases?",
        expected_response="Yes, 1 case was reported."
    )

def test_alcohol_timing():
    assert query_and_validate(
        question="When was the alcohol incident reported?",
        expected_response="12/19/24 – THU at 21:08"
    )

def test_drugs_location():
    assert query_and_validate(
        question="What is the location for drugs incident?",
        expected_response="there is no information about drugs incidents"
    )

def test_stolen_truck():
    assert query_and_validate(
        question="Any truck stolen, if yes, where?",
        expected_response="there is no information about any truck stolen"
    )

def test_code_12():
    assert query_and_validate(
        question="When and where the CODE 12 reported?",
        expected_response="12/19/24 – THU at 14:16 and 2000 Block Of SOTO ST"
    )

def test_strong_arm_robbery():
    assert query_and_validate(
        question="Where was the strong-arm robbery reported and what is the disposition of it?",
        expected_response="600 Block Of EXPOSITION BL AND NO CRIME OCCURRED; NO REPORT TAKEN"
    )

def test_trespassing_time():
    assert query_and_validate(
        question="At what time the trespassing incident occurred?",
        expected_response="12/19/24 – THU at 22:05"
    )

def test_trespassing_location():
    assert query_and_validate(
        question="Where was the trespassing incident?",
        expected_response="32ND ST & SHRINE PL "
    )

def test_stolen_vehicle():
    assert query_and_validate(
        question="When and where was the motor vehicle stolen?",
        expected_response="12/19/24 – THU at 08:15 AND FLOWER STREET STRUCTURE"
    )

def test_fire_incident():
    assert query_and_validate(
        question="Where was the fire incident?",
        expected_response="there is no information about fire incident"
    )

def test_hospital_security():
    assert query_and_validate(
        question="Where was the hospital security incident?",
        expected_response="there is no information about hospital security incident"
    )

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3.2:latest")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
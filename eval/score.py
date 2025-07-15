import json
import re

def extract_option(text):
    """
    Extracts the first option letter (A, B, C, or D) from a given string.

    Args:
        text (str): The string containing the answer or option.

    Returns:
        str: The extracted letter (e.g., 'A') or None if no letter is found.
    """
    if not isinstance(text, str):
        return None
    
    # Use regular expression to find the first occurrence of A, B, C, or D.
    match = re.search(r'[A-D]', text)
    
    if match:
        return match.group(0)
    
    return None

def calculate_accuracy(json_file_path):
    """
    Calculates the accuracy of model predictions from a JSON file by comparing
    the extracted option letters (A, B, C, D).

    Args:
        json_file_path (str): The path to the JSON file.

    Returns:
        float: The accuracy percentage, or None if an error occurs.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file at {json_file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {json_file_path} is not a valid JSON file.")
        return None

    if not data:
        return 0.0

    correct_predictions = 0
    total_predictions = len(data)

    for item in data:
        # Extract the option letter from both the model's output and the correct answer
        model_answer_option = extract_option(item.get("model_output"))
        correct_answer_option = extract_option(item.get("correct_answer"))

        # Check if both options were successfully extracted and if they match
        if model_answer_option and model_answer_option == correct_answer_option:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

if __name__ == '__main__':
    # Replace '/path/to/your/file.json' with the actual path to your JSON file.
    file_path = '/path/to/your/file.json'
    
    accuracy = calculate_accuracy(file_path)
    
    if accuracy is not None:
        print(f"The accuracy is: {accuracy:.2f}%")

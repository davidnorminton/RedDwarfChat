import os
import json

def load_json_file(file_path):
    """
    Load JSON data from a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        exit(1)

def process_episodes(episodes_data):
    """
    Process episodes data into a list of passages.
    """
    passages = []
    for season in episodes_data:
        series_number = season.get('series', 'unknown')
        episodes = season.get('episodes', [])
        for episode in episodes:
            episode_number = episode.get('episode_number', 'unknown')
            title = episode.get('title', 'No Title')
            synopsis = episode.get('synopsis', 'No Synopsis')
            air_date = episode.get('air_date', 'Unknown Air Date')
            
            passage_text = f"Title: {title}\nSynopsis: {synopsis}\nAir Date: {air_date}"
            passage_id = f"episode_{series_number}_{episode_number}"
            
            passages.append({
                'id': passage_id,
                'text': passage_text
            })
    print(f"Processed {len(passages)} episodes into passages.")
    return passages

def process_qa_data(qa_data, qa_type):
    """
    Process QA data (transcript or wiki_qa) into a list of passages.
    """
    passages = []
    for idx, qa in enumerate(qa_data):
        question = qa.get('question', 'No Question')
        answer = qa.get('answer', 'No Answer')
        
        passage_text = f"Question: {question}\nAnswer: {answer}"
        passage_id = f"{qa_type}_{idx}"
        
        passages.append({
            'id': passage_id,
            'text': passage_text
        })
    print(f"Processed {len(passages)} {qa_type} entries into passages.")
    return passages

def save_to_jsonl(passages, output_path):
    """
    Save passages to a JSON Lines (.jsonl) file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for passage in passages:
                json.dump(passage, f)
                f.write('\n')
        print(f"Combined data saved to '{output_path}'")
    except Exception as e:
        print(f"Error saving to '{output_path}': {e}")
        exit(1)

def ensure_directory(path):
    """
    Ensure that a directory exists. Create it if it doesn't.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: '{path}'")
    else:
        print(f"Directory already exists: '{path}'")

def main():
    # Define file paths
    data_dir = 'data'
    ensure_directory(data_dir)
    
    episodes_file = os.path.join(data_dir, 'episodes.json')
    transcript_file = os.path.join(data_dir, 'transcript.json')
    wiki_qa_file = os.path.join(data_dir, 'wiki_qa.json')
    
    output_jsonl = os.path.join(data_dir, 'rag_documents.jsonl')
    dataset_path = os.path.join(data_dir, 'rag_dataset')  # Directory to save the Hugging Face Dataset
    
    # Load data from JSON files
    episodes_data = load_json_file(episodes_file)
    transcript_data = load_json_file(transcript_file)
    wiki_qa_data = load_json_file(wiki_qa_file)
    
    # Process data into passages
    passages = []
    passages.extend(process_episodes(episodes_data))
    passages.extend(process_qa_data(transcript_data, 'transcript'))
    passages.extend(process_qa_data(wiki_qa_data, 'wiki_qa'))
    
    # Save combined passages to a JSONL file
    save_to_jsonl(passages, output_jsonl)
    

if __name__ == "__main__":
    main()
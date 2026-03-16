# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import re
import glob
import logging
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

def analyze_quality_ratings(user_list: List, base_path: str = 'data/evaluation', 
                   model_id_asst: str = 'claude-3-sonnet-v1', 
                   model_id_eval: str = 'claude-3-5-sonnet-v2',
                   matching_string_score: str = 'Naturalness Score',
                   matching_string_justification: str = 'Justification',
                   dimension: str = 'naturalness',
                   file_ext: str = '_user',
                   multi_domain: bool = False) -> Dict:
    """
    Analyze coherence ratings across specified user directories.
    
    Args:
        user_list: List of user numbers to analyze
        base_path: Base directory path for evaluation files
        model_id_asst: Model ID for the assistant
        model_id_eval: Model ID for the evaluator
    
    Returns:
        Dictionary with comprehensive rating statistics
    """
    total_files = 0
    coherence_ratings = []
    coherence_ratings_by_domain = defaultdict(list)
    low_rating_files = []  # Track files with low ratings (≤ 3)
    justifications = []  # Store justifications for analysis
    
    def extract_score(content: str) -> Tuple[int, str]:
        """
        Extract coherence score and justification from content.
        
        Returns:
            Tuple of (score, justification)
        """
        # Extract coherence score
        score_pattern = rf"{matching_string_score}: (\d+)"
        score_match = re.search(score_pattern, content)
        score = int(score_match.group(1)) if score_match else None
        
        # Extract justification
        justification_pattern = rf"{matching_string_justification}: (.*?)(?=\n\n|$)"
        justification_match = re.search(justification_pattern, content, re.DOTALL)
        justification = justification_match.group(1).strip() if justification_match else None
        
        return score, justification

    # Loop through specified user range
    for i in user_list:
        # Create the path pattern to match user directory
        user_dir_pattern = os.path.join(base_path, f'user{i}', model_id_asst, dimension, model_id_eval)
        
        # Ensure the directory exists
        if not os.path.isdir(user_dir_pattern):
            print(f"Directory not found: {user_dir_pattern}")
            continue
        
        # Find all .txt files in the directory
        if multi_domain:
            matching_files = glob.glob(os.path.join(user_dir_pattern, f'MD*{file_ext}.txt'))
        else:
            matching_files = glob.glob(os.path.join(user_dir_pattern, f'SD*{file_ext}.txt'))
        
        # Process each file
        for file_path in matching_files:
            total_files += 1
            
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    
                    # Extract score and justification
                    score, justification = extract_score(content)
                    
                    if score is not None:
                        coherence_ratings.append(score)
                        justifications.append(justification)
                        file_name = file_path.split("/")[-1]
                        domain = file_name.split("-")[1]
                        coherence_ratings_by_domain[domain].append(score)
                        
                        # Track files with low ratings (≤ 3)
                        if score <= 2:
                            low_rating_files.append({
                                'file_path': file_path,
                                'score': score,
                                'justification': justification
                            })
                        elif score == 3:
                            # print(file_path)
                            pass
                    else:
                        print("Score is not parsed")
                    
            except IOError as e:
                print(f"Error reading file {file_path}: {e}")
    
    # Calculate statistics
    statistics = {
        'count': len(coherence_ratings),
        'average': sum(coherence_ratings) / len(coherence_ratings) if coherence_ratings else 0,
        'min': min(coherence_ratings) if coherence_ratings else 0,
        'max': max(coherence_ratings) if coherence_ratings else 0,
        'distribution': {
            score: coherence_ratings.count(score) 
            for score in range(min(coherence_ratings or [0]), max(coherence_ratings or [0]) + 1)
        }
    }

    stats_by_domain = {}
    # Compute max, min, and average for each key
    for key, values in coherence_ratings_by_domain.items():
        max_value = max(values)
        min_value = min(values)
        avg_value = sum(values) / len(values)
        stats_by_domain[key] = {
            "count": len(values), 
            "max": max_value, 
            "min": min_value, 
            "average": avg_value
            }

    
    return {
        'total_files': total_files,
        'statistics': statistics,
        'stats_by_domain': stats_by_domain,
        'low_rating_files': low_rating_files,
        'raw_data': {
            'coherence_ratings': coherence_ratings,
            'justifications': justifications
        }
    }

def analyze_task_completion_ratings(user_list: List, base_path: str = 'data/evaluation', 
                   model_id_asst: str = 'claude-3-sonnet-v1', 
                   model_id_eval: str = 'claude-3-5-sonnet-v2',
                   multi_domain: bool = False) -> Dict:
    """
    Analyze coherence ratings across specified user directories.
    
    Args:
        user_list: List of user numbers to analyze
        base_path: Base directory path for evaluation files
        model_id_asst: Model ID for the assistant
        model_id_eval: Model ID for the evaluator
    
    Returns:
        Dictionary with comprehensive rating statistics
    """
    total_files = 0
    true_verdicts = 0
    tc_ratings_by_domain = defaultdict(list)
    low_rating_files = []  # Track files with False
    
    def check_verdict(content: str) -> bool:
        """
        Check if "VERDICT: " is present (True or False) and extract justification from content.
        
        Returns:
            Tuple of (verdict, justification)
        """
        # Check for VERDICT
        verdict_pattern = r"VERDICT: (True|False)"
        verdict_match = re.search(verdict_pattern, content, re.IGNORECASE)
        verdict = verdict_match.group(1).lower() == "true" if verdict_match else None
       
        return verdict
    
    # Loop through specified user range
    for i in user_list:
        # Create the path pattern to match user directory
        user_dir_pattern = os.path.join(base_path, f'user{i}', model_id_asst, "task_completion", model_id_eval)
        
        # Ensure the directory exists
        if not os.path.isdir(user_dir_pattern):
            print(f"Directory not found: {user_dir_pattern}")
            continue
        
        # Find all .txt files in the directory
        if multi_domain:
            matching_files = glob.glob(os.path.join(user_dir_pattern, f'MD*.txt'))
        else:
            matching_files = glob.glob(os.path.join(user_dir_pattern, f'SD*.txt'))
        
        # Process each file
        for file_path in matching_files:
            total_files += 1
            
            try:
                 # Read file content
                with open(file_path, 'r') as file:
                    content = file.read()
                    file_name = file_path.split("/")[-1]
                    domain = file_name.split("-")[1]
                    # Check verdict using improved parsing
                    if check_verdict(content):
                        true_verdicts += 1
                        tc_ratings_by_domain[domain].append(1)
                    else:
                        tc_ratings_by_domain[domain].append(0)
                        low_rating_files.append({
                                'file_path': file_path
                            })
                    
            except IOError as e:
                print(f"Error reading file {file_path}: {e}")
     # Calculate true verdict rate
    verdict_rate = true_verdicts / total_files if total_files > 0 else 0

    stats_by_domain = {}
    # Compute max, min, and average for each key
    for key, values in tc_ratings_by_domain.items():
        avg_value = sum(values) / len(values)
        stats_by_domain[key] = {
            "count": len(values), 
            "tc_rate": avg_value
            }
        
    return {
        'total_files': total_files,
        'tc_rate': verdict_rate,
        'stats_by_domain': stats_by_domain,
        'low_rating_files': low_rating_files,
    }


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO, datefmt="%Y-%m-%d %H:%M")
    parser = argparse.ArgumentParser(description="Generate personas for user profiles.")

    # Define arguments for start and end indices
    parser.add_argument("-s", "--start_index", type=int, default=0, help="The starting index of the user profiles.")
    parser.add_argument("-e", "--end_index", type=int, default=1499, help="The ending index of the user profiles.")
    parser.add_argument("-m", "--model_id_asst", type=str, default='claude-3-sonnet-v1', help="The model id of the assistant used in the dialogue generation.")
    parser.add_argument("-d", "--eval_dimension", type=str, default='naturalness', help="The evaluation dimension for the dialogue.")
    parser.add_argument("-f", "--file_ext", type=str, default='', help="The file extension.")
    parser.add_argument("-md", "--multi_domain", action="store_true", help="Whether to ran eval on multi-domain tasks.")
    parser.add_argument("-s2", "--sample_20", action="store_true", help="Whether to use small sample of 20 users.")
    parser.add_argument("-s3", "--sample_30", action="store_true", help="Whether to use small sample of 30 users.")
    parser.add_argument("-s5", "--sample_50", action="store_true", help="Whether to use small sample of 50 users.")
    parser.add_argument("-s10", "--sample_100", action="store_true", help="Whether to use small sample of 100 users.")
    parser.add_argument("-i", "--icl", action="store_true", help="Whether to use in-context learning for path.")
    parser.add_argument("-p", "--icl_path", type=str, default="icl", help="Path for in-context learning experiment.")

    # Parse arguments
    args = parser.parse_args()

    sample_20_idxs = [66, 132, 139, 207, 230, 340, 386, 389, 415, 428, 597, 746, 774, 854, 900, 1111, 1197, 1231, 1322, 1458]
    sample_30_idxs = [7, 66, 132, 139, 207, 230, 251, 313, 340, 376, 386, 389, 415, 428, 597, 746, 774, 788, 822, 854, 900, 1111, 1142, 1150, 1197, 1231, 1322, 1458, 1492, 1495]
    sample_50_idxs = [7, 53, 66, 107, 132, 139, 167, 195, 207, 230, 251, 313, 340, 376, 386, 389, 415, 418, 428, 439, 517, 532, 597, 630, 641, 660, 674, 701, 746, 774, 788, 802, 822, 854, 900, 1111, 1142, 1145, 1150, 1197, 1231, 1322, 1335, 1362, 1377, 1439, 1458, 1492, 1493, 1495]
    sample_100_idxs = [7, 21, 53, 66, 86, 107, 132, 139, 157, 166, 167, 168, 195, 207, 230, 248, 251, 312, 313, 340, 352, 363, 365, 376, 386, 389, 394, 415, 418, 428, 431, 439, 470, 482, 517, 532, 597, 619, 630, 641, 657, 659, 660, 664, 674, 686, 689, 701, 744, 745, 746, 774, 788, 802, 813, 822, 838, 840, 842, 847, 854, 857, 870, 878, 880, 900, 913, 928, 942, 954, 997, 1069, 1111, 1114, 1118, 1120, 1142, 1145, 1150, 1151, 1167, 1184, 1197, 1231, 1246, 1322, 1330, 1335, 1345, 1362, 1377, 1384, 1434, 1439, 1458, 1461, 1492, 1493, 1495, 1496]

  
    if args.sample_30:
        sample_idxs = sample_30_idxs
    if args.sample_50:
        sample_idxs = sample_50_idxs
    elif args.sample_100:
        sample_idxs = sample_100_idxs
    elif args.sample_20:
        sample_idxs = sample_20_idxs
    elif args.start_index or args.end_index:
        sample_idxs = [i for i in range(args.start_index, args.end_index+1)]
    else:
        print("wrong sample index provided")
        exit()
    
    base_path = "output/evaluation"

    if args.eval_dimension == "task_completion":
        results = analyze_task_completion_ratings(
            sample_idxs, 
            base_path=base_path, 
            model_id_asst=args.model_id_asst, 
            model_id_eval="claude-3-sonnet-v1",
            multi_domain=args.multi_domain)
        
        # Print summary
        print(f"\nAnalyzed {results['total_files']} files")
        
        print("\nOverall Statistics:")
        print(f"Average Task Completion Rate: {results['tc_rate'] * 100:.2f}%")

        for key, stats in sorted(results['stats_by_domain'].items()):
            print(f"\n{key} Statistics:")
            print(f"Count {stats['count']}")
            print(f"Average Task Completion Rate: {stats['tc_rate'] * 100:.2f}%")
    else:
        if args.eval_dimension == "naturalness":
            matching_string_score = "Naturalness Score"
        elif args.eval_dimension == "coherence":
            matching_string_score = "Coherence Score"
        elif args.eval_dimension == "personalization":
            matching_string_score = "Personalization Score"
        else:
            print(f"dimension entered did not match: {args.eval_dimension}")
            exit()
        
        results = analyze_quality_ratings(
            sample_idxs, 
            base_path=base_path, 
            model_id_asst=args.model_id_asst, 
            model_id_eval="claude-3-5-sonnet-v2",
            matching_string_score=matching_string_score, # Naturalness Score, Coherence Score, Personalization Score
            dimension=args.eval_dimension, # naturalness, coherence, personalization
            file_ext=args.file_ext,
            multi_domain=args.multi_domain) # _user, _asst
        
        # Print summary
        print(f"\nAnalyzed {results['total_files']} files")

        print("\nOverall Statistics:")
        print(f"Average {args.eval_dimension} rating: {results['statistics']['average']:.2f}")
        print(f"Range: {results['statistics']['min']} - {results['statistics']['max']}")
        
   

if __name__ == "__main__":
    main()


    
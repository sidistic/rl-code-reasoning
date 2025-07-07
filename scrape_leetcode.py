"""
Leetcode Problem Scraper
Scrapes coding problems and solutions for RLAIF training
"""

import requests
import json
from typing import List, Dict
import time
import argparse

class LeetcodeScraper:
    """Simple Leetcode problem scraper using public APIs"""
    
    def __init__(self):
        self.base_url = "https://leetcode.com/api/problems/all/"
        self.problems = []
    
    def get_problem_list(self, difficulty="easy", limit=20):
        """Get list of problems (using mock data for simplicity)"""
        # Note: Real implementation would use Leetcode's GraphQL API
        # For demo, using structured mock problems
        
        mock_problems = [
            {
                "title": "Two Sum",
                "prompt": "Given an array of integers nums and an integer target, return indices of two numbers that add up to target.",
                "solution": "def twoSum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        if target - num in seen:\n            return [seen[target - num], i]\n        seen[num] = i",
                "test_cases": [
                    {"input": "twoSum([2,7,11,15], 9)", "output": "[0,1]"},
                    {"input": "twoSum([3,2,4], 6)", "output": "[1,2]"}
                ]
            },
            {
                "title": "Reverse Integer", 
                "prompt": "Given a signed 32-bit integer x, return x with its digits reversed.",
                "solution": "def reverseInteger(x):\n    sign = -1 if x < 0 else 1\n    x = abs(x)\n    rev = int(str(x)[::-1])\n    return sign * rev if rev < 2**31 else 0",
                "test_cases": [
                    {"input": "reverseInteger(123)", "output": "321"},
                    {"input": "reverseInteger(-123)", "output": "-321"}
                ]
            },
            {
                "title": "Palindrome Number",
                "prompt": "Given an integer x, return true if x is a palindrome.",
                "solution": "def isPalindrome(x):\n    if x < 0:\n        return False\n    return str(x) == str(x)[::-1]",
                "test_cases": [
                    {"input": "isPalindrome(121)", "output": "True"},
                    {"input": "isPalindrome(-121)", "output": "False"}
                ]
            },
            {
                "title": "Valid Parentheses",
                "prompt": "Given a string s containing just '()', '{}', '[]', determine if input is valid.",
                "solution": "def isValid(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n        else:\n            stack.append(char)\n    return not stack",
                "test_cases": [
                    {"input": "isValid('()')", "output": "True"},
                    {"input": "isValid('()[]{}')", "output": "True"}
                ]
            },
            {
                "title": "Merge Two Sorted Lists",
                "prompt": "Merge two sorted linked lists and return it as a sorted list.",
                "solution": "def mergeTwoLists(l1, l2):\n    dummy = ListNode(0)\n    curr = dummy\n    while l1 and l2:\n        if l1.val <= l2.val:\n            curr.next = l1\n            l1 = l1.next\n        else:\n            curr.next = l2\n            l2 = l2.next\n        curr = curr.next\n    curr.next = l1 or l2\n    return dummy.next",
                "test_cases": [
                    {"input": "mergeTwoLists([1,2,4], [1,3,4])", "output": "[1,1,2,3,4,4]"},
                ]
            }
        ]
        
        return mock_problems[:limit]
    
    def format_for_rlaif(self, problems: List[Dict]) -> List[Dict]:
        """Format problems for RLAIF training"""
        formatted = []
        
        for p in problems:
            # Create multiple training examples from each problem
            base_prompt = f"Write a Python function to solve: {p['prompt']}"
            
            # Add the main problem
            formatted.append({
                "prompt": base_prompt,
                "solution": p['solution'],
                "test_cases": p['test_cases'],
                "difficulty": "medium"
            })
            
            # Add variation with example
            if p['test_cases']:
                example_prompt = f"{base_prompt}\n\nExample: {p['test_cases'][0]['input']} should return {p['test_cases'][0]['output']}"
                formatted.append({
                    "prompt": example_prompt,
                    "solution": p['solution'],
                    "test_cases": p['test_cases'],
                    "difficulty": "easy"
                })
        
        return formatted
    
    def scrape_and_save(self, output_file="leetcode_problems.json", limit=20):
        """Main function to scrape and save problems"""
        print(f"🕷️ Scraping {limit} Leetcode problems...")
        
        # Get problems
        problems = self.get_problem_list(limit=limit)
        
        # Format for RLAIF
        formatted = self.format_for_rlaif(problems)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(formatted, f, indent=2)
        
        print(f"✅ Saved {len(formatted)} training examples to {output_file}")
        return formatted

def main():
    """Example usage"""
    parser = argparse.ArgumentParser(description="Scrape Leetcode problems")
    parser.add_argument("--count", type=int, default=10, 
                       help="Number of problems to scrape")
    parser.add_argument("--output", default="leetcode_problems.json",
                       help="Output file name")
    args = parser.parse_args()
    
    scraper = LeetcodeScraper()
    
    # Scrape problems with specified count
    problems = scraper.scrape_and_save(
        output_file=args.output, 
        limit=args.count
    )
    
    # Show sample
    print("\n📝 Sample problem:")
    print(f"Prompt: {problems[0]['prompt'][:100]}...")
    print(f"Solution preview: {problems[0]['solution'][:50]}...")

if __name__ == "__main__":
    main()
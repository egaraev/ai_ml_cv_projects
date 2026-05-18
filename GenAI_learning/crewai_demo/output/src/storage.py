import json
import os
from typing import List, Dict

class TodoStorage:
    """
    Handles storage and retrieval of todo tasks in JSON format.
    """
    
    def __init__(self, filename: str):
        """
        Initialize the storage with a filename.
        
        Args:
            filename: Path to the JSON file
        """
        self.filename = filename
        
    def load(self) -> List[Dict]:
        """
        Load tasks from the storage file.
        
        Returns:
            List of task dictionaries
        """
        if not os.path.exists(self.filename):
            return []
            
        try:
            with open(self.filename, 'r') as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except (json.JSONDecodeError, IOError):
            # Return empty list when file is corrupted or unreadable
            return []
            
    def save(self, tasks: List[Dict]) -> None:
        """
        Save tasks to the storage file.
        
        Args:
            tasks: List of task dictionaries to save
        """
        try:
            with open(self.filename, 'w') as f:
                json.dump(tasks, f, indent=2)
        except IOError:
            raise IOError("Failed to write to storage file")
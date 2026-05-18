#!/usr/bin/env python3
import json
import os

class Storage:
    def __init__(self, filename="tasks.json"):
        self.filename = filename
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        """Create the tasks file if it doesn't exist"""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as f:
                json.dump([], f)
    
    def load_tasks(self):
        """Load tasks from the storage file"""
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_tasks(self, tasks):
        """Save tasks to the storage file"""
        with open(self.filename, 'w') as f:
            json.dump(tasks, f, indent=2)
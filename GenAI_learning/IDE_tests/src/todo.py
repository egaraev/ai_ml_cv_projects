#!/usr/bin/env python3
from storage import Storage

class TodoApp:
    def __init__(self, storage=None):
        self.storage = storage or Storage()
        self.tasks = self.storage.load_tasks()
    
    def add_task(self, description):
        """Add a new task to the todo list"""
        task = {
            "id": len(self.tasks) + 1,
            "description": description,
            "done": False
        }
        self.tasks.append(task)
        self.storage.save_tasks(self.tasks)
    
    def list_tasks(self):
        """List all tasks with their status"""
        if not self.tasks:
            print("No tasks found")
            return
        
        print("Tasks:")
        for task in self.tasks:
            status = "✓" if task["done"] else "○"
            print(f"{task['id']}. {status} {task['description']}")
    
    def mark_done(self, task_id):
        """Mark a task as done"""
        for task in self.tasks:
            if task["id"] == task_id:
                task["done"] = True
                self.storage.save_tasks(self.tasks)
                return
        print(f"Task {task_id} not found")
    
    def delete_task(self, task_id):
        """Delete a task"""
        for i, task in enumerate(self.tasks):
            if task["id"] == task_id:
                self.tasks.pop(i)
                # Re-number tasks
                for j, task in enumerate(self.tasks):
                    task["id"] = j + 1
                self.storage.save_tasks(self.tasks)
                return
        print(f"Task {task_id} not found")
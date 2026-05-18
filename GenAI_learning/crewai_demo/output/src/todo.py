from typing import List, Tuple
from storage import TodoStorage

class TodoApp:
    """
    A CLI todo application that manages tasks.
    
    Provides functionality to add, list, mark as done, and delete tasks.
    """
    
    def __init__(self, storage_file: str = "todo.json"):
        """
        Initialize the TodoApp with a storage file.
        
        Args:
            storage_file: Path to the JSON file for storing tasks
        """
        self.storage = TodoStorage(storage_file)
        
    def add_task(self, task: str) -> None:
        """
        Add a new task to the todo list.
        
        Args:
            task: The task description to add
            
        Raises:
            ValueError: If task is empty or only whitespace
        """
        if not task or not task.strip():
            raise ValueError("Task cannot be empty")
        
        tasks = self.storage.load()
        tasks.append({"task": task.strip(), "done": False})
        self.storage.save(tasks)
        
    def list_tasks(self) -> List[Tuple[str, bool]]:
        """
        Get all tasks in the todo list.
        
        Returns:
            List of tuples containing (task_description, is_done)
        """
        tasks = self.storage.load()
        return [(task["task"], task["done"]) for task in tasks]
        
    def mark_done(self, index: int) -> None:
        """
        Mark a task as done.
        
        Args:
            index: Index of the task to mark done (0-based)
            
        Raises:
            IndexError: If index is out of range
        """
        tasks = self.storage.load()
        
        if index < 0 or index >= len(tasks):
            raise IndexError("Task index out of range")
            
        tasks[index]["done"] = True
        self.storage.save(tasks)
        
    def delete_task(self, index: int) -> None:
        """
        Delete a task from the todo list.
        
        Args:
            index: Index of the task to delete (0-based)
            
        Raises:
            IndexError: If index is out of range
        """
        tasks = self.storage.load()
        
        if index < 0 or index >= len(tasks):
            raise IndexError("Task index out of range")
            
        tasks.pop(index)
        self.storage.save(tasks)
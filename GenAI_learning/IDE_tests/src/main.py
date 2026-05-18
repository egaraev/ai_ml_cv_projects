#!/usr/bin/env python3
import sys
from todo import TodoApp

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [add|list|done|delete] [task]")
        return
    
    app = TodoApp()
    
    command = sys.argv[1]
    
    if command == "add":
        if len(sys.argv) < 3:
            print("Usage: python main.py add \"task description\"")
            return
        task = sys.argv[2]
        app.add_task(task)
        print(f"Added task: {task}")
        
    elif command == "list":
        app.list_tasks()
        
    elif command == "done":
        if len(sys.argv) < 3:
            print("Usage: python main.py done <task_id>")
            return
        try:
            task_id = int(sys.argv[2])
            app.mark_done(task_id)
            print(f"Marked task {task_id} as done")
        except ValueError:
            print("Task ID must be a number")
            
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python main.py delete <task_id>")
            return
        try:
            task_id = int(sys.argv[2])
            app.delete_task(task_id)
            print(f"Deleted task {task_id}")
        except ValueError:
            print("Task ID must be a number")
            
    else:
        print("Unknown command. Use: add, list, done, or delete")

if __name__ == "__main__":
    main()
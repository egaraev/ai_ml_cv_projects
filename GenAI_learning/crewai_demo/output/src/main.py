import sys
from todo import TodoApp

def main():
    """Main entry point for the CLI todo application."""
    if len(sys.argv) < 2:
        print("Usage: todo add <task> | todo list | todo done <index> | todo delete <index>")
        return
    
    app = TodoApp()
    
    try:
        command = sys.argv[1]
        
        if command == "add":
            if len(sys.argv) < 3:
                print("Error: Task description required for add command")
                return
            task = " ".join(sys.argv[2:])
            app.add_task(task)
            print(f"Added task: {task}")
            
        elif command == "list":
            tasks = app.list_tasks()
            if not tasks:
                print("No tasks found.")
            else:
                for i, (task, done) in enumerate(tasks):
                    status = "✓" if done else "○"
                    print(f"{i+1}. {status} {task}")
                    
        elif command == "done":
            if len(sys.argv) < 3:
                print("Error: Task index required for done command")
                return
            try:
                index = int(sys.argv[2]) - 1
                app.mark_done(index)
                print(f"Marked task {index+1} as done")
            except ValueError:
                print("Error: Index must be a number")
            except IndexError:
                print("Error: Task index out of range")
                
        elif command == "delete":
            if len(sys.argv) < 3:
                print("Error: Task index required for delete command")
                return
            try:
                index = int(sys.argv[2]) - 1
                app.delete_task(index)
                print(f"Deleted task {index+1}")
            except ValueError:
                print("Error: Index must be a number")
            except IndexError:
                print("Error: Task index out of range")
                
        else:
            print(f"Unknown command: {command}")
            print("Usage: todo add <task> | todo list | todo done <index> | todo delete <index>")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
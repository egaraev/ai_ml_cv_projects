#!/usr/bin/env python3
import pytest
from todo import TodoApp
from storage import Storage

def test_storage_init(tmp_path):
    """Test Storage initialization"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    storage = Storage()
    assert (original_cwd / "tasks.json").exists()
    assert storage.filename == "tasks.json"

def test_storage_load_empty(tmp_path):
    """Test loading tasks from empty file"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    storage = Storage()
    tasks = storage.load_tasks()
    assert tasks == []

def test_storage_save_load(tmp_path):
    """Test saving and loading tasks"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    storage = Storage()
    
    # Save some tasks
    tasks = [
        {"id": 1, "description": "Test task", "done": False},
        {"id": 2, "description": "Another task", "done": True}
    ]
    storage.save_tasks(tasks)
    
    # Load them back
    loaded_tasks = storage.load_tasks()
    assert loaded_tasks == tasks

def test_todo_app_init(tmp_path):
    """Test TodoApp initialization"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    app = TodoApp()
    assert hasattr(app, 'storage')
    assert hasattr(app, 'tasks')

def test_add_task(tmp_path):
    """Test adding a task"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    app = TodoApp()
    
    initial_count = len(app.tasks)
    app.add_task("Test task")
    
    assert len(app.tasks) == initial_count + 1
    assert app.tasks[-1]["description"] == "Test task"
    assert app.tasks[-1]["done"] == False
    assert app.tasks[-1]["id"] == 1

def test_list_tasks_empty(tmp_path):
    """Test listing tasks when empty"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    app = TodoApp()
    # This test just verifies it doesn't crash
    app.list_tasks()

def test_list_tasks_with_content(tmp_path):
    """Test listing tasks with content"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    app = TodoApp()
    app.add_task("First task")
    app.add_task("Second task")
    
    # Capture output
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    app.list_tasks()
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    assert "1. ○ First task" in output
    assert "2. ○ Second task" in output

def test_mark_done(tmp_path):
    """Test marking a task as done"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    app = TodoApp()
    app.add_task("Test task")
    
    app.mark_done(1)
    
    assert app.tasks[0]["done"] == True
    assert app.tasks[0]["description"] == "Test task"

def test_mark_done_nonexistent(tmp_path):
    """Test marking a nonexistent task as done"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    app = TodoApp()
    app.add_task("Test task")
    
    # This should print an error message but not crash
    app.mark_done(999)

def test_delete_task(tmp_path):
    """Test deleting a task"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    app = TodoApp()
    app.add_task("First task")
    app.add_task("Second task")
    app.add_task("Third task")
    
    app.delete_task(2)
    
    # Task IDs should be renumbered
    assert len(app.tasks) == 2
    assert app.tasks[0]["id"] == 1
    assert app.tasks[0]["description"] == "First task"
    assert app.tasks[1]["id"] == 2
    assert app.tasks[1]["description"] == "Third task"

def test_delete_task_nonexistent(tmp_path):
    """Test deleting a nonexistent task"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    app = TodoApp()
    app.add_task("Test task")
    
    # This should print an error message but not crash
    app.delete_task(999)

def test_main_integration(tmp_path):
    """Test the main function integration"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    
    # Test adding a task
    app = TodoApp()
    app.add_task("Integration test task")
    
    # Test listing tasks
    tasks = app.tasks
    assert len(tasks) == 1
    assert tasks[0]["description"] == "Integration test task"
    
    # Test marking done
    app.mark_done(1)
    assert tasks[0]["done"] == True
    
    # Test deleting
    app.delete_task(1)
    assert len(app.tasks) == 0

def test_todo_app_init_with_custom_storage(tmp_path):
    """Test TodoApp initialization with custom storage"""
    original_cwd = tmp_path
    import os
    os.chdir(original_cwd)
    custom_storage = Storage()
    app = TodoApp(storage=custom_storage)
    assert app.storage is custom_storage
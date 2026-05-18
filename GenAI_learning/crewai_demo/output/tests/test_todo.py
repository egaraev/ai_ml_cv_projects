import os
import tempfile
import pytest
from todo import TodoApp
from storage import TodoStorage

def test_todo_storage_load_empty():
    """Test loading from empty storage file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write("")
        filename = f.name
    
    try:
        storage = TodoStorage(filename)
        assert storage.load() == []
    finally:
        os.unlink(filename)

def test_todo_storage_load_nonexistent():
    """Test loading from non-existent file."""
    storage = TodoStorage("nonexistent.json")
    assert storage.load() == []

def test_todo_storage_save_load():
    """Test saving and loading tasks."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        storage = TodoStorage(filename)
        tasks = [{"task": "test task", "done": False}]
        storage.save(tasks)
        loaded_tasks = storage.load()
        assert loaded_tasks == tasks
    finally:
        os.unlink(filename)

def test_todo_app_add_task():
    """Test adding a task."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        app.add_task("Buy milk")
        tasks = app.list_tasks()
        assert len(tasks) == 1
        assert tasks[0][0] == "Buy milk"
        assert tasks[0][1] == False
    finally:
        os.unlink(filename)

def test_todo_app_add_empty_task():
    """Test adding an empty task."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        with pytest.raises(ValueError):
            app.add_task("")
        with pytest.raises(ValueError):
            app.add_task("   ")
    finally:
        os.unlink(filename)

def test_todo_app_list_tasks():
    """Test listing tasks."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        assert app.list_tasks() == []
        
        app.add_task("Task 1")
        app.add_task("Task 2")
        
        tasks = app.list_tasks()
        assert len(tasks) == 2
        assert tasks[0][0] == "Task 1"
        assert tasks[0][1] == False
        assert tasks[1][0] == "Task 2"
        assert tasks[1][1] == False
    finally:
        os.unlink(filename)

def test_todo_app_mark_done():
    """Test marking a task as done."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        app.add_task("Task 1")
        app.add_task("Task 2")
        
        # Mark first task as done
        app.mark_done(0)
        
        tasks = app.list_tasks()
        assert tasks[0][1] == True
        assert tasks[1][1] == False
    finally:
        os.unlink(filename)

def test_todo_app_mark_done_invalid_index():
    """Test marking a task with invalid index."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        app.add_task("Task 1")
        with pytest.raises(IndexError):
            app.mark_done(-1)
        with pytest.raises(IndexError):
            app.mark_done(5)
    finally:
        os.unlink(filename)

def test_todo_app_delete_task():
    """Test deleting a task."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        app.add_task("Task 1")
        app.add_task("Task 2")
        app.add_task("Task 3")
        
        # Delete second task
        app.delete_task(1)
        
        tasks = app.list_tasks()
        assert len(tasks) == 2
        assert tasks[0][0] == "Task 1"
        assert tasks[1][0] == "Task 3"
    finally:
        os.unlink(filename)

def test_todo_app_delete_invalid_index():
    """Test deleting a task with invalid index."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        app.add_task("Task 1")
        with pytest.raises(IndexError):
            app.delete_task(-1)
        with pytest.raises(IndexError):
            app.delete_task(5)
    finally:
        os.unlink(filename)

def test_todo_app_mark_done_boundary_cases():
    """Test marking tasks as done with boundary index values."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        app.add_task("Task 1")
        app.add_task("Task 2")
        
        # Test boundary indices
        app.mark_done(0)  # First task
        app.mark_done(1)  # Last task
        
        tasks = app.list_tasks()
        assert tasks[0][1] == True
        assert tasks[1][1] == True
    finally:
        os.unlink(filename)

def test_todo_app_delete_boundary_cases():
    """Test deleting tasks with boundary index values."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        app.add_task("Task 1")
        app.add_task("Task 2")
        app.add_task("Task 3")
        
        # Delete first and last task
        app.delete_task(0)  # First task
        app.delete_task(1)  # Last remaining task (originally index 2)
        
        tasks = app.list_tasks()
        assert len(tasks) == 1
        assert tasks[0][0] == "Task 2"
    finally:
        os.unlink(filename)

def test_todo_app_empty_storage_file():
    """Test working with an empty storage file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write("")
        filename = f.name
    
    try:
        app = TodoApp(filename)
        assert app.list_tasks() == []
        
        app.add_task("Test Task")
        tasks = app.list_tasks()
        assert len(tasks) == 1
        assert tasks[0][0] == "Test Task"
    finally:
        os.unlink(filename)

def test_todo_app_corrupted_storage_file():
    """Test handling of corrupted storage file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write("this is not valid json")
        filename = f.name
    
    try:
        app = TodoApp(filename)
        # Should not raise an exception, should return empty list
        tasks = app.list_tasks()
        assert tasks == []
        
        # Should still be able to add tasks
        app.add_task("Test Task")
        tasks = app.list_tasks()
        assert len(tasks) == 1
        assert tasks[0][0] == "Test Task"
    finally:
        os.unlink(filename)

def test_todo_app_no_tasks():
    """Test operations when no tasks exist."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        assert app.list_tasks() == []
        
        # Should raise IndexError when trying to mark nonexistent task done
        with pytest.raises(IndexError):
            app.mark_done(0)
        
        # Should raise IndexError when trying to delete nonexistent task
        with pytest.raises(IndexError):
            app.delete_task(0)
    finally:
        os.unlink(filename)

def test_todo_app_whitespace_task():
    """Test handling of tasks with only whitespace."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        with pytest.raises(ValueError):
            app.add_task("   ")
        with pytest.raises(ValueError):
            app.add_task("\t\n")
        with pytest.raises(ValueError):
            app.add_task(" \t \n ")
    finally:
        os.unlink(filename)

def test_todo_app_task_with_leading_trailing_whitespace():
    """Test that leading/trailing whitespace is stripped from tasks."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filename = f.name
    
    try:
        app = TodoApp(filename)
        app.add_task("  Task with spaces  ")
        tasks = app.list_tasks()
        assert tasks[0][0] == "Task with spaces"
        assert tasks[0][1] == False
    finally:
        os.unlink(filename)
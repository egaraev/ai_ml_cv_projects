import json
import os
import tempfile
from unittest.mock import patch
from storage import TodoStorage

def test_todo_storage_load_empty_file():
    """Test loading from an empty file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write("")
        temp_filename = f.name
    
    try:
        storage = TodoStorage(temp_filename)
        tasks = storage.load()
        assert tasks == []
    finally:
        os.unlink(temp_filename)

def test_todo_storage_load_nonexistent_file():
    """Test loading from a non-existent file."""
    storage = TodoStorage("nonexistent_file.json")
    tasks = storage.load()
    assert tasks == []

def test_todo_storage_load_valid_json():
    """Test loading from a file with valid JSON."""
    tasks_data = [{"id": 1, "task": "Test task", "completed": False}]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(tasks_data, f)
        temp_filename = f.name
    
    try:
        storage = TodoStorage(temp_filename)
        tasks = storage.load()
        assert tasks == tasks_data
    finally:
        os.unlink(temp_filename)

def test_todo_storage_load_corrupted_json():
    """Test loading from a file with corrupted JSON - should return empty list."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write("this is not valid json")
        temp_filename = f.name
    
    try:
        storage = TodoStorage(temp_filename)
        tasks = storage.load()
        assert tasks == []
    finally:
        os.unlink(temp_filename)

def test_todo_storage_save():
    """Test saving tasks to file."""
    tasks_data = [{"id": 1, "task": "Test task", "completed": False}]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_filename = f.name
    
    try:
        storage = TodoStorage(temp_filename)
        storage.save(tasks_data)
        
        # Verify file content
        with open(temp_filename, 'r') as f:
            content = f.read().strip()
            assert content != ""
            loaded_tasks = json.loads(content)
            assert loaded_tasks == tasks_data
    finally:
        os.unlink(temp_filename)
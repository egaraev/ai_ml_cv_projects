import pytest
import csv
import os
from unittest.mock import patch, mock_open
from src.library import Book
from src.storage import load_books, save_books

def test_load_books_creates_file_with_headers_if_not_exists():
    """Test that load_books creates file with headers if it doesn't exist."""
    with patch('src.storage.os.path.exists', return_value=False):
        with patch('src.storage.open', mock_open()) as mock_file:
            books = load_books()
            
            mock_file.assert_called_once()
            mock_file().write.assert_any_call('isbn,title,author,available\n')

def test_load_books_reads_existing_file():
    """Test that load_books reads existing file correctly."""
    csv_content = """isbn,title,author,available
978-0-123456-78-9,The Great Gatsby,F. Scott Fitzgerald,true
978-0-987654-32-1,To Kill a Mockingbird,Harper Lee,false"""

    with patch('src.storage.os.path.exists', return_value=True):
        with patch('src.storage.open', mock_open(read_data=csv_content)) as mock_file:
            books = load_books()
            
            assert len(books) == 2
            assert books[0].isbn == "978-0-123456-78-9"
            assert books[0].title == "The Great Gatsby"
            assert books[0].author == "F. Scott Fitzgerald"
            assert books[0].available == True
            assert books[1].isbn == "978-0-987654-32-1"
            assert books[1].title == "To Kill a Mockingbird"
            assert books[1].author == "Harper Lee"
            assert books[1].available == False

def test_save_books_writes_correct_data():
    """Test that save_books writes data in correct format."""
    books = [
        Book("978-0-123456-78-9", "The Great Gatsby", "F. Scott Fitzgerald", True),
        Book("978-0-987654-32-1", "To Kill a Mockingbird", "Harper Lee", False)
    ]
    
    with patch('src.storage.open', mock_open()) as mock_file:
        save_books(books)
        
        mock_file().write.assert_any_call('isbn,title,author,available\n')
        mock_file().write.assert_any_call('978-0-123456-78-9,The Great Gatsby,F. Scott Fitzgerald,true\n')
        mock_file().write.assert_any_call('978-0-987654-32-1,To Kill a Mockingbird,Harper Lee,false\n')
        
        # Check that writeheader is called
        assert mock_file().write.call_count >= 3

def test_save_books_handles_empty_list():
    """Test that save_books handles empty book list correctly."""
    with patch('src.storage.open', mock_open()) as mock_file:
        save_books([])
        
        # Should write header but no data rows
        assert mock_file().write.call_count >= 1  # At least header
        mock_file().write.assert_any_call('isbn,title,author,available\n')

def test_load_books_handles_malformed_rows():
    """Test that load_books gracefully handles malformed rows."""
    csv_content = """isbn,title,author,available
978-0-123456-78-9,The Great Gatsby,F. Scott Fitzgerald,true
,The Great Gatsby,F. Scott Fitzgerald,true
978-0-987654-32-1,To Kill a Mockingbird,Harper Lee,false"""

    with patch('src.storage.os.path.exists', return_value=True):
        with patch('src.storage.open', mock_open(read_data=csv_content)) as mock_file:
            books = load_books()
            
            # Should skip rows with empty ISBN
            assert len(books) == 2  # Only two valid books
            assert books[0].isbn == "978-0-123456-78-9"
            assert books[1].isbn == "978-0-987654-32-1"

def test_load_books_handles_missing_columns():
    """Test that load_books gracefully handles missing columns."""
    csv_content = """isbn,title,author
978-0-123456-78-9,The Great Gatsby,F. Scott Fitzgerald
978-0-987654-32-1,To Kill a Mockingbird,Harper Lee"""

    with patch('src.storage.os.path.exists', return_value=True):
        with patch('src.storage.open', mock_open(read_data=csv_content)) as mock_file:
            books = load_books()
            
            # Should skip rows with missing columns
            assert len(books) == 2  # Two valid books
            assert books[0].isbn == "978-0-123456-78-9"
            assert books[1].isbn == "978-0-987654-32-1"
            # available should default to False for missing column
            assert books[0].available == False
            assert books[1].available == False

def test_load_books_handles_undefined_available_field():
    """Test that load_books gracefully handles undefined available field."""
    csv_content = """isbn,title,author,available
978-0-123456-78-9,The Great Gatsby,F. Scott Fitzgerald,undefined
978-0-987654-32-1,To Kill a Mockingbird,Harper Lee,false"""

    with patch('src.storage.os.path.exists', return_value=True):
        with patch('src.storage.open', mock_open(read_data=csv_content)) as mock_file:
            books = load_books()
            
            assert len(books) == 2
            # First book should have available = False due to invalid 'undefined' value
            assert books[0].available == False  
            assert books[1].available == False
--- data/books.csv ---
isbn,title,author,available
978-0-123456-78-9,The Great Gatsby,F. Scott Fitzgerald,true
978-0-987654-32-1,To Kill a Mockingbird,Harper Lee,false
978-0-55321-355-7,1984,George Orwell,true
978-0-34539-146-0,The Catcher in the Rye,J.D. Salinger,false
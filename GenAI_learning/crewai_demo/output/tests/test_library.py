import pytest
from unittest.mock import patch, mock_open
from src.library import LibraryManager, Book
from src.storage import load_books, save_books

@pytest.fixture
def sample_books():
    return [
        Book("978-0-123456-78-9", "The Great Gatsby", "F. Scott Fitzgerald", True),
        Book("978-0-987654-32-1", "To Kill a Mockingbird", "Harper Lee", False),
        Book("978-0-55321-355-7", "1984", "George Orwell", True),
        Book("978-0-34539-146-0", "The Catcher in the Rye", "J.D. Salinger", False)
    ]

@pytest.fixture
def library_manager(sample_books):
    with patch('src.library.load_books', return_value=sample_books):
        return LibraryManager()

def test_add_book_happy_path(library_manager):
    """Test adding a new book successfully."""
    initial_count = len(library_manager.books)
    
    library_manager.add_book("978-0-111111-11-1", "New Book", "New Author")
    
    assert len(library_manager.books) == initial_count + 1
    new_book = library_manager.books[-1]
    assert new_book.isbn == "978-0-111111-11-1"
    assert new_book.title == "New Book"
    assert new_book.author == "New Author"
    assert new_book.available == True

def test_add_book_duplicate_isbn_raises_value_error(library_manager):
    """Test that adding a book with existing ISBN raises ValueError."""
    with pytest.raises(ValueError, match="Book with ISBN 978-0-123456-78-9 already exists"):
        library_manager.add_book("978-0-123456-78-9", "Duplicate Book", "Author")

def test_add_book_empty_inputs_raises_value_error(library_manager):
    """Test that adding a book with empty inputs raises ValueError."""
    with pytest.raises(ValueError, match="ISBN cannot be empty"):
        library_manager.add_book("", "Book Title", "Author")
    
    with pytest.raises(ValueError, match="Title cannot be empty"):
        library_manager.add_book("978-0-111111-11-1", "", "Author")
    
    with pytest.raises(ValueError, match="Author cannot be empty"):
        library_manager.add_book("978-0-111111-11-1", "Book Title", "")

def test_remove_book_happy_path(library_manager):
    """Test removing a book successfully."""
    initial_count = len(library_manager.books)
    
    library_manager.remove_book("978-0-123456-78-9")
    
    assert len(library_manager.books) == initial_count - 1
    assert all(book.isbn != "978-0-123456-78-9" for book in library_manager.books)

def test_remove_book_nonexistent_raises_value_error(library_manager):
    """Test that removing a non-existent book raises ValueError."""
    with pytest.raises(ValueError, match="Book with ISBN 978-0-111111-11-1 not found"):
        library_manager.remove_book("978-0-111111-11-1")

def test_checkout_happy_path(library_manager):
    """Test checking out a book successfully."""
    book = library_manager.books[0]
    assert book.available == True
    
    library_manager.checkout("978-0-123456-78-9")
    
    assert book.available == False

def test_checkout_already_checked_out_raises_value_error(library_manager):
    """Test that checking out an already checked out book raises ValueError."""
    library_manager.checkout("978-0-987654-32-1")  # Check out first
    
    with pytest.raises(ValueError, match="Book with ISBN 978-0-987654-32-1 is already checked out"):
        library_manager.checkout("978-0-987654-32-1")

def test_checkout_nonexistent_book_raises_value_error(library_manager):
    """Test that checking out a non-existent book raises ValueError."""
    with pytest.raises(ValueError, match="Book with ISBN 978-0-111111-11-1 not found"):
        library_manager.checkout("978-0-111111-11-1")

def test_return_book_happy_path(library_manager):
    """Test returning a book successfully."""
    library_manager.checkout("978-0-987654-32-1")  # Check out first
    book = library_manager.books[1]
    assert book.available == False
    
    library_manager.return_book("978-0-987654-32-1")
    
    assert book.available == True

def test_return_book_nonexistent_raises_value_error(library_manager):
    """Test that returning a non-existent book raises ValueError."""
    with pytest.raises(ValueError, match="Book with ISBN 978-0-111111-11-1 not found"):
        library_manager.return_book("978-0-111111-11-1")

def test_search_by_author_case_insensitive(library_manager):
    """Test that search is case insensitive."""
    results = library_manager.search_by_author("george orwell")
    assert len(results) == 1
    assert results[0].author == "George Orwell"
    
    results = library_manager.search_by_author("GEORGE ORWELL")
    assert len(results) == 1
    assert results[0].author == "George Orwell"

def test_list_available_returns_only_available_books(library_manager):
    """Test that list_available only returns available books."""
    available_books = library_manager.list_available()
    
    assert len(available_books) == 2
    assert all(book.available for book in available_books)
    assert "978-0-123456-78-9" in [book.isbn for book in available_books]
    assert "978-0-55321-355-7" in [book.isbn for book in available_books]
    assert "978-0-987654-32-1" not in [book.isbn for book in available_books]
    assert "978-0-34539-146-0" not in [book.isbn for book in available_books]

def test_search_by_author_invalid_input(library_manager):
    """Test that search handles invalid author input."""
    with pytest.raises(ValueError, match="Author must be a string"):
        library_manager.search_by_author(None)
    
    with pytest.raises(ValueError, match="Author must be a string"):
        library_manager.search_by_author(123)
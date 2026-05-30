from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Book:
    """Represents a book in the library with ISBN, title, author, and availability status."""
    isbn: str
    title: str
    author: str
    available: bool

class LibraryManager:
    """Manages a collection of books with operations to add, remove, checkout, return, search, and list books."""
    
    def __init__(self):
        """Initialize the library manager with loaded books from storage."""
        from .storage import load_books
        self.books: List[Book] = load_books()
        # Create a dictionary for O(1) lookups by ISBN
        self._books_by_isbn = {book.isbn: book for book in self.books}
    
    def add_book(self, isbn: str, title: str, author: str) -> None:
        """
        Add a new book to the library.
        
        Args:
            isbn: Unique ISBN of the book
            title: Title of the book
            author: Author of the book
            
        Raises:
            ValueError: If a book with the given ISBN already exists or if inputs are empty
        """
        # Validate inputs
        if not isbn or not isbn.strip():
            raise ValueError("ISBN cannot be empty")
        if not title or not title.strip():
            raise ValueError("Title cannot be empty")
        if not author or not author.strip():
            raise ValueError("Author cannot be empty")
        
        # Check if book with this ISBN already exists
        if isbn in self._books_by_isbn:
            raise ValueError(f"Book with ISBN {isbn} already exists")
        
        new_book = Book(isbn=isbn.strip(), title=title.strip(), author=author.strip(), available=True)
        self.books.append(new_book)
        self._books_by_isbn[new_book.isbn] = new_book
        from .storage import save_books
        try:
            save_books(self.books)
        except Exception as e:
            # Revert the change if save fails
            self.books.remove(new_book)
            del self._books_by_isbn[new_book.isbn]
            raise e
    
    def remove_book(self, isbn: str) -> None:
        """
        Remove a book from the library.
        
        Args:
            isbn: ISBN of the book to remove
            
        Raises:
            ValueError: If no book with the given ISBN exists
        """
        if isbn not in self._books_by_isbn:
            raise ValueError(f"Book with ISBN {isbn} not found")
        
        book_to_remove = self._books_by_isbn[isbn]
        self.books.remove(book_to_remove)
        del self._books_by_isbn[isbn]
        from .storage import save_books
        try:
            save_books(self.books)
        except Exception as e:
            # Revert the change if save fails
            self.books.append(book_to_remove)
            self._books_by_isbn[isbn] = book_to_remove
            raise e
    
    def _find_book(self, isbn: str) -> Optional[Book]:
        """
        Find a book by ISBN.
        
        Args:
            isbn: ISBN of the book to find
            
        Returns:
            The book if found, None otherwise
        """
        return self._books_by_isbn.get(isbn)
    
    def checkout(self, isbn: str) -> None:
        """
        Checkout a book from the library.
        
        Args:
            isbn: ISBN of the book to checkout
            
        Raises:
            ValueError: If the book doesn't exist or is already checked out
        """
        book = self._find_book(isbn)
        if book is None:
            raise ValueError(f"Book with ISBN {isbn} not found")
        
        if not book.available:
            raise ValueError(f"Book with ISBN {isbn} is already checked out")
        
        book.available = False
        from .storage import save_books
        try:
            save_books(self.books)
        except Exception as e:
            # Revert the change if save fails
            book.available = True
            raise e
    
    def return_book(self, isbn: str) -> None:
        """
        Return a book to the library.
        
        Args:
            isbn: ISBN of the book to return
            
        Raises:
            ValueError: If the book doesn't exist in the system
        """
        book = self._find_book(isbn)
        if book is None:
            raise ValueError(f"Book with ISBN {isbn} not found")
        
        book.available = True
        from .storage import save_books
        try:
            save_books(self.books)
        except Exception as e:
            # Revert the change if save fails
            book.available = False
            raise e
    
    def search_by_author(self, author: str) -> List[Book]:
        """
        Search for books by a specific author.
        
        Args:
            author: Author name to search for
            
        Returns:
            List of books by the given author
        """
        if not isinstance(author, str):
            raise ValueError("Author must be a string")
        return [book for book in self.books if book.author.lower() == author.lower()]
    
    def list_available(self) -> List[Book]:
        """
        Get a list of all available books.
        
        Returns:
            List of available books
        """
        return [book for book in self.books if book.available]
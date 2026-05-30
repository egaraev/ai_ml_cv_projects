import csv
import os
from typing import List
from .library import Book

def load_books() -> List[Book]:
    """
    Load books from CSV file.
    
    Returns:
        List of Book objects loaded from file
    """
    books = []
    file_path = "data/books.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['isbn', 'title', 'author', 'available'])
    
    # Read books from file
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Validate that all required columns exist
            if not all(col in row for col in ['isbn', 'title', 'author', 'available']):
                continue  # Skip malformed rows
            
            # Strip whitespace from all fields
            isbn = row['isbn'].strip()
            title = row['title'].strip()
            author = row['author'].strip()
            
            # Skip empty entries
            if not isbn or not title or not author:
                continue
            
            try:
                available = row['available'].strip().lower() == 'true'
            except (KeyError, AttributeError):
                available = False  # Default value for missing or invalid available field
            
            books.append(Book(isbn=isbn, title=title, author=author, available=available))
    
    return books

def save_books(books: List[Book]) -> None:
    """
    Save books to CSV file.
    
    Args:
        books: List of Book objects to save
    """
    file_path = "data/books.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['isbn', 'title', 'author', 'available']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for book in books:
            writer.writerow({
                'isbn': book.isbn,
                'title': book.title,
                'author': book.author,
                'available': str(book.available).lower()
            })
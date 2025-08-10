#!/usr/bin/env python3
"""
Test script to verify database persistence functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.vector_database import VectorDatabase
from database.embedding_generator import EmbeddingGenerator
from config.settings import settings
import tempfile
import shutil

def test_database_persistence():
    """Test if database can persist data across sessions."""
    
    # Create temporary directory for testing
    test_dir = Path(tempfile.mkdtemp())
    print(f"Testing database persistence in: {test_dir}")
    
    try:
        # Test 1: Create new database
        print("\n=== Test 1: Creating new database ===")
        config = {'db_path': str(test_dir)}
        db = VectorDatabase(config)
        
        # Add some test data
        test_texts = [
            "This is a test document about machine learning",
            "Another document about artificial intelligence",
            "A third document about data science"
        ]
        
        test_metadata = [
            {'source': 'test1.txt', 'type': 'text'},
            {'source': 'test2.txt', 'type': 'text'},
            {'source': 'test3.txt', 'type': 'text'}
        ]
        
        print(f"Adding {len(test_texts)} test documents...")
        ids = db.add_to_vectordb(test_texts, test_metadata)
        print(f"Added documents with IDs: {ids}")
        
        # Check stats
        stats = db.get_database_stats()
        print(f"Database stats: {stats}")
        
        # Test 2: Save database
        print("\n=== Test 2: Saving database ===")
        db.save_database()
        print("Database saved successfully")
        
        # Test 3: Create new instance and load
        print("\n=== Test 3: Loading database in new instance ===")
        del db  # Delete old instance
        
        # Create new instance
        db2 = VectorDatabase(config)
        print("New database instance created")
        
        # Check if data was loaded
        stats2 = db2.get_database_stats()
        print(f"Loaded database stats: {stats2}")
        
        # Test 4: Search in loaded database
        print("\n=== Test 4: Testing search in loaded database ===")
        results = db2.search("machine learning", k=5)
        print(f"Search results: {len(results)} found")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: Score={result['score']:.3f}, Text='{result['metadata']['text'][:50]}...'")
        
        # Test 5: Verify persistence
        print("\n=== Test 5: Verifying persistence ===")
        if stats2['total_vectors'] == len(test_texts):
            print("✅ SUCCESS: Database persistence working correctly!")
        else:
            print(f"❌ FAILED: Expected {len(test_texts)} vectors, got {stats2['total_vectors']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")

if __name__ == "__main__":
    print("Testing ElevateAI Database Persistence")
    print("=" * 50)
    
    success = test_database_persistence()
    
    if success:
        print("\n🎉 All tests passed! Database persistence is working correctly.")
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        sys.exit(1)

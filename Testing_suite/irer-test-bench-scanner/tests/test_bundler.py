import unittest
from core.file_walker import FileWalker
from core.ast_parser import ASTParser
from core.ai_orchestrator import AIOrchestrator

class TestBundler(unittest.TestCase):

    def setUp(self):
        self.file_walker = FileWalker()
        self.ast_parser = ASTParser()
        self.ai_orchestrator = AIOrchestrator()

    def test_scan_directory(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file and a directory to ignore
            os.mkdir(os.path.join(tmpdir, 'ignoreme'))
            with open(os.path.join(tmpdir, 'a.py'), 'w') as f:
                f.write('print("hi")')
            with open(os.path.join(tmpdir, 'ignoreme', 'b.py'), 'w') as f:
                f.write('print("ignore")')
            fw = FileWalker(ignore_dirs=['ignoreme'])
            files = fw.scan_directory(tmpdir)
            self.assertIsInstance(files, list)
            self.assertTrue(any('a.py' in f for f in files))
            self.assertFalse(any('b.py' in f for f in files))

    def test_get_filtered_files(self):
        test_path = './test_directory'  # Replace with a valid test directory
        self.file_walker.scan_directory(test_path)
        filtered_files = self.file_walker.get_filtered_files()
        self.assertTrue(all(isinstance(file, str) for file in filtered_files))

    def test_parse_file(self):
        test_file = './test_directory/test_file.py'  # Replace with a valid test file
        result = self.ast_parser.parse_file(test_file)
        self.assertIsNotNone(result)

    def test_run_analysis(self):
        test_data = {'file_name': 'test_file.py', 'content': 'print("Hello World")'}  # Example data
        insights = self.ai_orchestrator.run_analysis(test_data)
        self.assertIsInstance(insights, dict)

if __name__ == '__main__':
    unittest.main()
class FileWalker:
    def __init__(self, ignore_dirs=None):
        if ignore_dirs is None:
            ignore_dirs = []
        self.ignore_dirs = ignore_dirs
        self.filtered_files = []

    def scan_directory(self, path):
        import os
        filtered = []
        for root, dirs, files in os.walk(path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            for file in files:
                filtered.append(os.path.join(root, file))
        self.filtered_files = filtered
        return filtered

    def get_filtered_files(self):
        return self.filtered_files
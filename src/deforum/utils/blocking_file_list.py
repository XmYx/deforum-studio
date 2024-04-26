import os
import time

class BlockingFileList:
    def __init__(self, base_directory, expected_file_count=0, extensions=["jpg", "png"], timeout_seconds=30):
        self.base_directory = base_directory
        self.expected_file_count = expected_file_count
        self.extensions = extensions
        self.timeout_seconds = timeout_seconds

    def __getitem__(self, index):
        start_time = time.time()
        waited = 0

        while waited < self.timeout_seconds:
            for ext in self.extensions:
                file_path = os.path.join(self.base_directory, f"{index}.{ext}")
                if os.path.exists(file_path):
                    return file_path
            waited = time.time() - start_time
            print(f"Could not find matching {self.extensions} file for index {index} in {self.base_directory}. Waited {waited:.2f}/{self.timeout_seconds}s...")
            time.sleep(1)  # Wait for 1 second before checking again
        
        raise FileNotFoundError(f"No file with matching {self.extensions} file for index {index} in {self.base_directory} after waiting for {self.timeout_seconds} seconds")

    def __len__(self):
        return self.expected_file_count
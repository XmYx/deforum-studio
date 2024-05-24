import time
from threading import Event, Thread
from unittest.mock import patch

import pytest

from deforum.utils.blocking_file_list import BlockingFileList


class TestBlockingFileList:
    @patch('deforum.utils.blocking_file_list.os.path.exists')
    def test_getitem_file_exists(self, mock_exists):
        # Setup
        mock_exists.side_effect = lambda filepath: filepath == "/base/dir/optional_prefix0.jpg"
        base_directory = "/base/dir"
        expected_file_count = 10
        extensions = ["jpg", "png"]
        timeout_seconds = 1
        optional_prefix = "optional_prefix"

        file_list = BlockingFileList(base_directory, expected_file_count, optional_prefix, extensions, timeout_seconds)

        # Act
        result = file_list[0]

        # Assert
        assert result == "/base/dir/optional_prefix0.jpg"


    @patch('deforum.utils.blocking_file_list.os.path.exists')
    def test_getitem_file_not_exists(self, mock_exists):
        # Setup
        mock_exists.return_value = False
        base_directory = "/base/dir"
        expected_file_count = 10
        extensions = ["jpg", "png"]
        timeout_seconds = 1
        optional_prefix = "optional_prefix"

        file_list = BlockingFileList(base_directory, expected_file_count, optional_prefix, extensions, timeout_seconds)

        # Act & Assert
        start_time = time.time()
        with pytest.raises(FileNotFoundError):
            file_list[0]
        elapsed_time = time.time() - start_time

        assert elapsed_time >= timeout_seconds


    @patch('deforum.utils.blocking_file_list.os.path.exists')
    def test_len(self, mock_exists):
        # Setup
        base_directory = "/base/dir"
        expected_file_count = 10
        extensions = ["jpg", "png"]
        timeout_seconds = 1
        optional_prefix = "optional_prefix"

        file_list = BlockingFileList(base_directory, expected_file_count, optional_prefix, extensions, timeout_seconds)

        # Act & Assert
        assert len(file_list) == expected_file_count


    @patch('deforum.utils.blocking_file_list.os.path.exists')
    def test_filename_order(self, mock_exists):
        # Setup
        mock_exists.return_value = False
        base_directory = "/base/dir"
        expected_file_count = 10
        extensions = ["jpg", "png"]
        timeout_seconds = 1
        optional_prefix = "foo"

        file_list = BlockingFileList(base_directory, expected_file_count, optional_prefix, extensions, timeout_seconds)

        # Expected order of filename checks
        expected_call_arguments = [
            ("/base/dir/foo000000034.jpg",),
            ("/base/dir/foo000000034.png",),
            ("/base/dir/foo34.jpg",),
            ("/base/dir/foo34.png",),
            ("/base/dir/000000034.jpg",),
            ("/base/dir/000000034.png",),
            ("/base/dir/34.jpg",),
            ("/base/dir/34.png",),
        ]

        # Act
        with pytest.raises(FileNotFoundError):
            file_list[34]

        # remove duplicates but preserve order, and filter out files unrelated to this test that may
        # end up in call_args_list due to other interactions.
        call_args_list = [args for args, _ in mock_exists.call_args_list if args[0].startswith("/base/dir")]
        actual_calls = list(dict.fromkeys(call_args_list))

        # Assert
        assert actual_calls == expected_call_arguments


    @patch('deforum.utils.blocking_file_list.os.path.exists')
    def test_file_appears_after_wait(self, mock_exists):
        # Setup
        base_directory = "/base/dir"
        expected_file_count = 10
        extensions = ["jpg", "png"]
        timeout_seconds = 5  # Longer timeout to allow for file appearance
        optional_prefix = "foo"

        # Initial mock setup: File does not exist initially, but appears after a delay
        mock_exists.side_effect = lambda filepath: filepath == "/base/dir/foo000000034.jpg" and file_created.is_set()

        file_list = BlockingFileList(base_directory, expected_file_count, optional_prefix, extensions, timeout_seconds)

        # Thread event to simulate file creation
        file_created = Event()
        def create_file():
            time.sleep(2)
            file_created.set()
        creation_thread = Thread(target=create_file)
        creation_thread.start()

        # Act
        result = file_list[34]
        creation_thread.join()

        # Assert
        assert result == "/base/dir/foo000000034.jpg"


# Run the tests
if __name__ == "__main__":
    pytest.main()

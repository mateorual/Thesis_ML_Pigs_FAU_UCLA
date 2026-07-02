"""
Box Client Wrapper
Handles Box API operations for file/folder navigation and downloads
"""

from boxsdk import OAuth2, Client
from io import BytesIO
import re
import requests


class BoxClientWrapper:
    """Wrapper class for Box API operations"""

    def __init__(self, credentials):
        """
        Initialize Box client with credentials

        Args:
            credentials: Dictionary with CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN
        """
        self.credentials = credentials
        self.client = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Box using refresh token for automatic renewal"""
        oauth = OAuth2(
            client_id=self.credentials['CLIENT_ID'],
            client_secret=self.credentials['CLIENT_SECRET'],
            access_token=self.credentials['ACCESS_TOKEN'],
            refresh_token=self.credentials['REFRESH_TOKEN']
        )

        # This will automatically refresh the token if expired
        self.client = Client(oauth)

        # Get the potentially new tokens (IMPORTANT: Box gives you a new refresh token too!)
        self.credentials['ACCESS_TOKEN'] = oauth.access_token
        self.credentials['REFRESH_TOKEN'] = oauth._refresh_token

    def get_user_info(self):
        """
        Get authenticated user information

        Returns:
            User object with name and login
        """
        return self.client.user().get()

    def get_folder_items(self, folder_id, limit=1000):
        """
        Get all items in a folder

        Args:
            folder_id: Box folder ID
            limit: Maximum number of items to retrieve

        Returns:
            List of items (files and folders)
        """
        try:
            # Request specific fields including 'size' for files
            fields = ['type', 'id', 'name', 'size']
            items = list(self.client.folder(folder_id).get_items(limit=limit, fields=fields))
            return items
        except Exception as e:
            print(f"Error accessing folder {folder_id}: {e}")
            return []

    def find_folders_by_name(self, parent_folder_id, folder_names):
        """
        Find folders by name within a parent folder

        Args:
            parent_folder_id: Parent folder ID to search in
            folder_names: List of folder names to find (exact match)

        Returns:
            Dictionary mapping folder_name -> folder_id
        """
        items = self.get_folder_items(parent_folder_id)
        found_folders = {}

        for item in items:
            if item.type == 'folder' and item.name in folder_names:
                found_folders[item.name] = item.id

        return found_folders

    def find_processed_folders(self, parent_folder_id, processed_patterns, max_depth=3):
        """
        Find folders matching "Processed" patterns within a parent folder (recursive search)

        Args:
            parent_folder_id: Parent folder ID to search in
            processed_patterns: List of folder name patterns to match
            max_depth: Maximum depth to search (default: 3 levels)

        Returns:
            List of tuples (folder_name, folder_id) for matching folders
        """
        processed_folders = []
        self._find_processed_folders_recursive(parent_folder_id, processed_patterns, processed_folders, current_depth=0, max_depth=max_depth)
        return processed_folders

    def _find_processed_folders_recursive(self, folder_id, processed_patterns, result_list, current_depth=0, max_depth=3):
        """
        Helper method for recursive search of "Processed" folders

        Args:
            folder_id: Current folder ID to search in
            processed_patterns: List of folder name patterns to match
            result_list: List to accumulate results
            current_depth: Current recursion depth
            max_depth: Maximum depth to search
        """
        if current_depth > max_depth:
            return

        try:
            items = self.get_folder_items(folder_id)

            for item in items:
                if item.type == 'folder':
                    # Check if folder name matches any pattern
                    is_processed = False
                    for pattern in processed_patterns:
                        if pattern.lower() in item.name.lower():
                            result_list.append((item.name, item.id))
                            is_processed = True
                            break

                    # If not a "Processed" folder, search inside it
                    if not is_processed:
                        self._find_processed_folders_recursive(
                            item.id,
                            processed_patterns,
                            result_list,
                            current_depth + 1,
                            max_depth
                        )
        except Exception as e:
            print(f"Error searching for processed folders in {folder_id}: {e}")

    def get_all_wav_files_recursive(self, folder_id, audio_extension='.wav'):
        """
        Recursively get all WAV files from a folder and its subfolders

        Args:
            folder_id: Starting folder ID
            audio_extension: Audio file extension to filter

        Returns:
            List of dictionaries with file metadata:
            {
                'file_id': str,
                'file_name': str,
                'size_bytes': int,
                'parent_folder_name': str,
                'path_info': str
            }
        """
        wav_files = []
        self._recursive_file_search(folder_id, audio_extension, wav_files, path="")
        return wav_files

    def _recursive_file_search(self, folder_id, audio_extension, result_list, path=""):
        """
        Helper method for recursive file search

        Args:
            folder_id: Current folder ID
            audio_extension: File extension to filter
            result_list: List to accumulate results
            path: Current path for tracking location
        """
        try:
            items = self.get_folder_items(folder_id)

            for item in items:
                if item.type == 'file':
                    # Check if it's a WAV file
                    if item.name.lower().endswith(audio_extension.lower()) and not item.name.startswith('._'):
                        file_info = {
                            'file_id': item.id,
                            'file_name': item.name,
                            'size_bytes': item.size,
                            'parent_folder_name': path.split('/')[-1] if path else '',
                            'path_info': path
                        }
                        result_list.append(file_info)

                elif item.type == 'folder':
                    # Recursively search in subfolders
                    new_path = f"{path}/{item.name}" if path else item.name
                    self._recursive_file_search(item.id, audio_extension, result_list, new_path)

        except Exception as e:
            print(f"Error searching in folder {folder_id}: {e}")

    def download_file_to_bytes(self, file_id):
        """
        Download file content to BytesIO object

        Args:
            file_id: Box file ID

        Returns:
            BytesIO object with file content, or None if error
        """
        try:
            file_content = self.client.file(file_id).content()
            return BytesIO(file_content)
        except Exception as e:
            print(f"Error downloading file {file_id}: {e}")
            return None

    def download_file_header(self, file_id, byte_range='0-511'):
        """
        Download only the file header (first bytes) for metadata extraction
        This is MUCH faster than downloading entire files

        Args:
            file_id: Box file ID
            byte_range: Range of bytes to download (default: first 512 bytes)

        Returns:
            BytesIO object with header content, or None if error
        """
        try:
            # Use Box API's byte range support (HTTP Range header)
            # Using requests directly to make Range request
            url = f"https://api.box.com/2.0/files/{file_id}/content"
            headers = {
                'Authorization': f'Bearer {self.client._oauth.access_token}',
                'Range': f'bytes={byte_range}'
            }

            response = requests.get(url, headers=headers)

            if response.status_code in [200, 206]:  # 206 = Partial Content
                return BytesIO(response.content)
            else:
                print(f"Warning: Could not get header for file {file_id}, status {response.status_code}")
                return None

        except Exception as e:
            print(f"Error downloading file header {file_id}: {e}")
            # Fallback: download full file if header download fails
            return None

    def get_folder_info(self, folder_id):
        """
        Get folder information

        Args:
            folder_id: Box folder ID

        Returns:
            Folder object or None if error
        """
        try:
            return self.client.folder(folder_id).get()
        except Exception as e:
            print(f"Error getting folder info for {folder_id}: {e}")
            return None

    def extract_week_from_path(self, path_info):
        """
        Extract week number from path

        Args:
            path_info: Path string

        Returns:
            Week number (int) or None
        """
        week_patterns = [
            r'[Ww]eek[\s_-]*(\d+)',
            r'[Ww](\d+)',
            r'week[\s_-]*(\d+)',
        ]

        for pattern in week_patterns:
            match = re.search(pattern, path_info)
            if match:
                return int(match.group(1))

        return None

    def extract_subject_name(self, folder_name):
        """
        Extract clean subject name from folder name

        Args:
            folder_name: Raw folder name (e.g., "P24-1-Elvis-DONE")

        Returns:
            Clean subject name (e.g., "Elvis")
        """
        # Remove "-DONE" suffix
        name = re.sub(r'-DONE$', '', folder_name, flags=re.IGNORECASE)

        # Remove various prefix patterns: "P24-X", "P24-X & Y", "P24-X and Y"
        name = re.sub(r'^P24-\d+[\s&]*', '', name)
        name = re.sub(r'^and\s+\d+-\d+\s+', '', name)  # Remove "and  24-3 "
        name = re.sub(r'^\d+\s+', '', name)  # Remove leading numbers
        name = re.sub(r'^-', '', name)  # Remove leading hyphen

        return name.strip()

    def parse_snip_folder_name(self, folder_name):
        """
        Parse folder name from "All Pig Snips" structure
        Format: week_X_SubjectName_TreatmentType_Snip
        Examples: "week_26_Madonna and Beyonce_B_Snip", "week_1_Michael and Prince_U_Snip"

        Args:
            folder_name: Folder name to parse

        Returns:
            Dictionary with 'week', 'subject', 'treatment' or None if parsing fails
        """
        # Pattern: week_NUMBER_SUBJECTNAME_LETTER_Snip
        pattern = r'week[_\s]*(\d+)[_\s]+(.+?)[_\s]+([BUS])[_\s]*[Ss]nip'

        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            week = int(match.group(1))
            subject = match.group(2).strip()
            treatment = match.group(3).upper()

            return {
                'week': week,
                'subject': subject,
                'treatment': treatment
            }

        return None

    def get_snip_folders_by_treatment(self, all_pig_snips_folder_id, treatment_folders_dict, target_subjects):
        """
        Get all snip folders organized by treatment type from "All Pig Snips" structure

        Args:
            all_pig_snips_folder_id: ID of "All Pig Snips" folder
            treatment_folders_dict: Dictionary mapping treatment code -> folder name
                                   e.g., {'B': 'Bilateral - DONE', 'U': 'Unilateral', 'S': 'Scar'}
            target_subjects: List of subject names we're interested in

        Returns:
            Dictionary: subject_name -> list of (folder_name, folder_id, week, treatment)
        """
        print(f"\nSearching in 'All Pig Snips' folder (ID: {all_pig_snips_folder_id})...")

        # Get treatment folders (Bilateral, Unilateral, Scar)
        treatment_folder_names = list(treatment_folders_dict.values())
        treatment_folders = self.find_folders_by_name(all_pig_snips_folder_id, treatment_folder_names)

        print(f"Found {len(treatment_folders)} treatment folders:")
        for folder_name in treatment_folders:
            print(f"  - {folder_name}")

        # Organize results by subject
        files_by_subject = {subject: [] for subject in target_subjects}

        # Process each treatment folder
        for treatment_code, treatment_folder_name in treatment_folders_dict.items():
            if treatment_folder_name not in treatment_folders:
                print(f"Warning: Treatment folder '{treatment_folder_name}' not found")
                continue

            treatment_folder_id = treatment_folders[treatment_folder_name]
            print(f"\nScanning '{treatment_folder_name}'...")

            # Get all snip folders in this treatment folder
            items = self.get_folder_items(treatment_folder_id)

            for item in items:
                if item.type == 'folder':
                    # Parse folder name
                    parsed = self.parse_snip_folder_name(item.name)

                    if parsed:
                        subject = parsed['subject']
                        week = parsed['week']
                        treatment = parsed['treatment']

                        # Check if this subject is in our target list
                        if subject in target_subjects:
                            print(f"  Found: {item.name} (Week {week}, Subject: {subject})")
                            files_by_subject[subject].append({
                                'folder_name': item.name,
                                'folder_id': item.id,
                                'week': week,
                                'treatment': treatment
                            })

        return files_by_subject

    def get_all_wav_files_with_metadata(self, folder_id, subject_name, week, audio_extension='.wav'):
        """
        Get all WAV files from a folder recursively with metadata

        Args:
            folder_id: Folder ID to search in
            subject_name: Subject name for metadata
            week: Week number for metadata
            audio_extension: Audio file extension

        Returns:
            List of file info dictionaries with subject and week metadata
        """
        wav_files = []
        self._recursive_file_search_with_metadata(
            folder_id,
            audio_extension,
            wav_files,
            subject_name,
            week,
            path=""
        )
        return wav_files

    def _recursive_file_search_with_metadata(self, folder_id, audio_extension, result_list, subject_name, week, path=""):
        """
        Helper method for recursive file search with metadata

        Args:
            folder_id: Current folder ID
            audio_extension: File extension to filter
            result_list: List to accumulate results
            subject_name: Subject name to add to metadata
            week: Week number to add to metadata
            path: Current path for tracking location
        """
        try:
            items = self.get_folder_items(folder_id)

            for item in items:
                if item.type == 'file':
                    # Check if it's a WAV file
                    if item.name.lower().endswith(audio_extension.lower()) and not item.name.startswith('._'):
                        file_info = {
                            'file_id': item.id,
                            'file_name': item.name,
                            'size_bytes': item.size,
                            'parent_folder_name': path.split('/')[-1] if path else '',
                            'path_info': path,
                            'subject': subject_name,
                            'week': week
                        }
                        result_list.append(file_info)

                elif item.type == 'folder':
                    # Recursively search in subfolders
                    new_path = f"{path}/{item.name}" if path else item.name
                    self._recursive_file_search_with_metadata(
                        item.id,
                        audio_extension,
                        result_list,
                        subject_name,
                        week,
                        new_path
                    )

        except Exception as e:
            print(f"Error searching in folder {folder_id}: {e}")

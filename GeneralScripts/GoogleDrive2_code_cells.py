# --- Cell 0 ---
import os
import io
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.http import MediaIoBaseDownload
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

def download_file(service, file_obj, parent_path, max_retries=3):
    import time
    import socket
    import traceback
    from googleapiclient.http import MediaIoBaseDownload

    file_id = file_obj['id']
    file_name = file_obj['name']
    file_path = os.path.join(parent_path, file_name)

    if os.path.exists(file_path):
        print(f"✅ Skipped (already exists): {file_name}")
        return file_name, 0, os.path.getsize(file_path)

    attempt = 0
    while attempt < max_retries:
        try:
            request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
            with io.FileIO(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                start = time.time()

                while not done:
                    try:
                        status, done = downloader.next_chunk()
                    except socket.timeout:
                        raise TimeoutError("Chunk read timed out")

                duration = time.time() - start
                size = os.path.getsize(file_path)
                return file_name, duration, size

        except (socket.timeout, TimeoutError):
            attempt += 1
            print(f"⏱️ Timeout on {file_name}, retrying ({attempt}/{max_retries})...")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error on {file_name}: {e}")
            traceback.print_exc()
            break

        # Cleanup if retrying
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

    print(f"❌ Failed after {max_retries} attempts: {file_name}")
    return file_name, 0, 0


# --- Cell 1 ---
def list_folder_contents(service, folder_id, parent_path='PhdFiles', drive_id=None, file_counter=None):
    import sys
    from googleapiclient.http import MediaIoBaseDownload

    os.makedirs(parent_path, exist_ok=True)

    if file_counter is None:
        file_counter = {'total': 0, 'current': 0}

    query = f"'{folder_id}' in parents and trashed = false"
    list_kwargs = {
        'q': query,
        'fields': "files(id, name, mimeType)",
        'supportsAllDrives': True,
        'includeItemsFromAllDrives': True
    }

    if drive_id:
        list_kwargs['corpora'] = 'drive'
        list_kwargs['driveId'] = drive_id
    else:
        list_kwargs['corpora'] = 'user'

    # Only calculate total file count once
    if file_counter['total'] == 0:
        def count_files(folder_id):
            count = 0
            sub_kwargs = list_kwargs.copy()
            sub_kwargs['q'] = f"'{folder_id}' in parents and trashed = false"
            sub_results = service.files().list(**sub_kwargs).execute()
            for item in sub_results.get('files', []):
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    count += count_files(item['id'])
                else:
                    count += 1
            return count

        file_counter['total'] = count_files(folder_id)
        print(f"\n🔢 Total files to download: {file_counter['total']}\n")

    results = service.files().list(**list_kwargs).execute()
    items = results.get('files', [])

    if not items:
        print("No files found in this folder.")
    else:
        for item in items:
            name = item['name']
            file_id = item['id']
            mime_type = item['mimeType']

            if mime_type == 'application/vnd.google-apps.folder':
                new_path = os.path.join(parent_path, name)
                list_folder_contents(service, file_id, new_path, drive_id, file_counter)
            else:
                file_counter['current'] += 1
                print(f"\n📥 [{file_counter['current']} / {file_counter['total']}] Downloading: {name}")
                file_path = os.path.join(parent_path, name)

                request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
                with io.FileIO(file_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        progress = int(status.progress() * 100)
                        sys.stdout.write(f"\r    {progress}% complete")
                        sys.stdout.flush()
                print()  # Newline after progress

# --- Cell 2 ---
def collect_files(service, folder_id, parent_path='downloads', drive_id=None):
    os.makedirs(parent_path, exist_ok=True)
    files_to_download = []

    query = f"'{folder_id}' in parents and trashed = false"
    list_kwargs = {
        'q': query,
        'fields': "files(id, name, mimeType)",
        'supportsAllDrives': True,
        'includeItemsFromAllDrives': True,
        'corpora': 'drive' if drive_id else 'user'
    }
    if drive_id:
        list_kwargs['driveId'] = drive_id

    response = service.files().list(**list_kwargs).execute()
    items = response.get('files', [])

    for item in items:
        name = item['name']
        mime_type = item['mimeType']
        file_id = item['id']

        if mime_type == 'application/vnd.google-apps.folder':
            new_path = os.path.join(parent_path, name)
            sub_files = collect_files(service, file_id, new_path, drive_id)
            files_to_download.extend(sub_files)
        else:
            files_to_download.append({'id': file_id, 'name': name, 'path': parent_path})

    return files_to_download

# --- Cell 3 ---
def download_all_files(service, files_to_download, max_workers=4):
    print(f"\n🔄 Starting downloads ({len(files_to_download)} files)...\n")
    total_size = 0
    total_time = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_file, service, f, f['path'])
            for f in files_to_download
        ]

        for future in as_completed(futures):
            try:
                name, duration, size = future.result()
                if duration > 0:
                    speed_mbps = (size * 8) / (duration * 1_000_000)
                    print(f"📥 {name} - {size/1_000:.1f} KB in {duration:.2f}s ({speed_mbps:.2f} Mbps)")
                total_size += size
                total_time += duration
            except Exception as e:
                print(f"❌ Error: {e}")

    if total_time > 0:
        avg_speed = (total_size * 8) / (total_time * 1_000_000)
        print(f"\n✅ All done. {len(files_to_download)} files, {total_size/1_000_000:.2f} MB total, avg {avg_speed:.2f} Mbps")

# --- Cell 4 ---
def authenticate():
    import pickle
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_console()  # ✅ Jupyter-friendly auth
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds

# --- Cell 5 ---
import ssl
import urllib3

# Override SSL context for httplib2 (used under the hood by Google API)
from googleapiclient.http import set_user_agent

try:
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
except Exception as e:
    print("⚠️ Could not set up custom SSL context:", e)

# --- Cell 6 ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = '18BMH-PhTZNiN8rbXyqZvz5J9qRGrRqbA'  # just the ID
DRIVE_ID = '0AHMeY0Fo1d79Uk9PVA'
CUSTOM_DOWNLOAD_PATH = r'C:\Users\Ry\Documents\PhD Thesis Writing\Data'

creds = authenticate()
service = build('drive', 'v3', credentials=creds)

files = collect_files(service, FOLDER_ID, CUSTOM_DOWNLOAD_PATH, DRIVE_ID)
download_all_files(service, files, max_workers=4)


from label_studio_sdk import Client
# from label_studio_sdk.client import LabelStudio
import time

# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '5cc455f7320bb14ba4539dab6ea60e815dc30e09'


# Import the SDK and the client module
def check_ls_sdk_connection():
    # Connect to the Label Studio API and check the connection
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()


def export_all_frames_ls():
    project_id = 15  # integer, get from url of the project, e.g., http://localhost:8080/projects/4/data?tab=8
    save_dir = '/Users/jiangao/Documents/Work/QUBRF/Piglets/Data/labelstudio/annos'
    # Connect to Label Studio
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    # Get your project by ID
    project = ls.get_project(project_id)
    # Create an export snapshot with interpolation enabled
    export_result = project.export_snapshot_create(
        title='Export with Interpolated Keyframes',
        interpolate_key_frames=True
    )
    # Get the export ID
    export_id = export_result['id']
    # # Wait for the export to complete
    # while True:
    #     export_status = project.get_export_status(export_id)
    #     if export_status['status'] == 'completed':
    #         break
    #     elif export_status['status'] == 'failed':
    #         raise Exception('Export failed')
    #     else:
    #         time.sleep(5)  # Wait for 5 seconds before checking again
    # Download the export
    export_file_path = project.export_snapshot_download(
        export_id, export_type='JSON', path=save_dir
    )
    print(f'Exported data saved to {export_file_path}')


if __name__ == '__main__':
    # check_ls_sdk_connection()
    export_all_frames_ls()
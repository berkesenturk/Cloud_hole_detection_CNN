import os
import sys
from azure.storage.filedatalake import DataLakeServiceClient


def upload_to_datalake(source_dir, container_name, connection_string):
    try:
        service_client = DataLakeServiceClient.from_connection_string(
            connection_string
        )
        file_system_client = service_client.get_file_system_client(
            file_system=container_name
        )

        print(
            f"""
        Starting upload from '{source_dir}' to container '{container_name}'...
        """
        )

        # Walk through local directory
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # Calculate relative path for
                # ADLS (removes local source_dir prefix)
                relative_path = os.path.relpath(
                    local_path, start=os.path.dirname(source_dir)
                )

                # Create file client and upload
                file_client = file_system_client.get_file_client(relative_path)
                with open(local_path, "rb") as data:
                    file_client.upload_data(data, overwrite=True)
                print(f"Uploaded: {relative_path}")

        print("Upload complete.")

    except Exception as e:
        print(f"Error: {e} {connection_string}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload_zarr.py <local_path> <container_name>")
        sys.exit(1)

    source_path = sys.argv[1]
    target_container = sys.argv[2]
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if not conn_str:
        print("""
            Error: AZURE_STORAGE_CONNECTION_STRING is missing.
        """)
        sys.exit(1)

    upload_to_datalake(source_path, target_container, conn_str)

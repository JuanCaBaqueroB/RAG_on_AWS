import boto3

def read_md_file_from_s3(bucket_name, file_key):
    # Initialize the S3 client
    s3 = boto3.client('s3')
    
    # Specify the bucket name and file key
    bucket_name = bucket_name
    file_key = file_key
    
    try:
        # Get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        
        # Read the contents of the file
        md_content = response['Body'].read().decode('utf-8')
        
        # Return the Markdown content
        return md_content
    except Exception as e:
        print(f"Error reading file from S3: {e}")
        return None

# Example usage
bucket_name = 'sagemakerdocuments'
file_key = 'amazon-sagemaker-toolkits.md'

md_content = read_md_file_from_s3(bucket_name, file_key)
if md_content:
    print("Markdown file content:")
    print(md_content)

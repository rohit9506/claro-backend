import os
import sys
from huggingface_hub import HfApi, create_repo, upload_folder

def deploy(token: str, space_name: str = "claro-backend-api"):
    if not token or not token.strip():
        print("Error: Hugging Face User Access Token is required.")
        sys.exit(1)
        
    clean_token = token.strip()
    api = HfApi(token=clean_token)
    
    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("username")
        print(f"Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"Authentication failed with provided token: {e}")
        sys.exit(1)
        
    repo_id = f"{username}/{space_name}"
    print(f"Creating / verifying Space repository: {repo_id} (SDK: docker)...")
    
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            token=clean_token,
            exist_ok=True,
            private=False
        )
        print(f"Space repository confirmed: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Space creation error: {e}")
        sys.exit(1)
        
    print("Uploading backend files to Hugging Face Cloud...")
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    upload_folder(
        folder_path=backend_dir,
        repo_id=repo_id,
        repo_type="space",
        token=clean_token,
        ignore_patterns=[
            "__pycache__/*",
            "*/__pycache__/*",
            ".pytest_cache/*",
            "*/.pytest_cache/*",
            "*.log",
            "cloudflared.exe"
        ]
    )
    
    subdomain = f"{username}-{space_name}".lower().replace("_", "-")
    api_url = f"https://{subdomain}.hf.space"
    
    print("\n========================================================")
    print("CLARO BACKEND SUCCESSFULLY DEPLOYED TO HUGGING FACE CLOUD!")
    print(f"Permanent Cloud HTTPS API: {api_url}")
    print(f"Space Dashboard: https://huggingface.co/spaces/{repo_id}")
    print("========================================================\n")
    return api_url

if __name__ == "__main__":
    hf_token = sys.argv[1] if len(sys.argv) > 1 else os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = input("Enter your Hugging Face User Access Token (from huggingface.co/settings/tokens): ").strip()
    deploy(hf_token)

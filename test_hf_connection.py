from uni2ts.model.moirai import MoiraiModule
import requests
if __name__ == "__main__":
    try:
        url = "https://huggingface.co/Salesforce/moirai-1.1-R-base/resolve/main/config.json"

        response = requests.head(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Headers:\n{response.headers}")
        if response.status_code == 200:
            print("✅ URL is accessible from SLURM job.")
        else:
            print("❌ URL is not accessible.")
        print("Attempting to load Moirai model...")
        model = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base",cache_dir="./hf_cache")
        print("✅ Moirai model loaded successfully!")
    except Exception as e:
        print("❌ Failed to load Moirai model:")
        print(e)
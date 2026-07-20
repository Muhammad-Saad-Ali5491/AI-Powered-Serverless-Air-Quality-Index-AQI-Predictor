import hopsworks
import os

# Fixed filename - this always gets overwritten, never creates a new timestamped file
OUTPUT_PATH = "aqi_data_export.csv"

def main():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)

    df = fg.read()
    df = df.sort_values("timestamp")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Exported {len(df)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

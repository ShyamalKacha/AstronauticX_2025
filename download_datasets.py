import pandas as pd
import requests
import io
import os

def download_kepler_data():
    """Download Kepler Objects of Interest (KOI) dataset"""
    print("Downloading Kepler Objects of Interest dataset...")
    
    # Using the NASA Exoplanet Archive API to get KOI data
    # This is a query to get confirmed exoplanets and candidates from the Kepler mission
    url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    
    # Query for Kepler Objects of Interest with disposition information
    params = {
        'table': 'q1_q17_dr25_koi',
        'format': 'csv',
        'select': 'kepid,koi_disposition,koi_period,koi_time0bk,koi_duration,koi_depth,koi_prad,koi_teq'
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # Parse the CSV data
            koi_data = pd.read_csv(io.StringIO(response.text))
            print(f"Downloaded Kepler data with {len(koi_data)} rows")
            print("Columns:", list(koi_data.columns))
            
            # Save to data directory
            koi_data.to_csv('data/koi_data.csv', index=False)
            print("Kepler data saved to data/koi_data.csv")
            
            return koi_data
        else:
            print(f"Error downloading Kepler data: {response.status_code}")
            print("Trying alternative endpoint...")
            
            # Alternative: Try to download using the TAP service
            tap_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/tap"
            tap_query = '''
                SELECT kepid,koi_disposition,koi_period,koi_time0bk,koi_duration,koi_depth,koi_prad,koi_teq
                FROM q1_q17_dr25_koi
                WHERE koi_disposition IS NOT NULL LIMIT 1000
            '''
            params = {
                'query': tap_query,
                'format': 'csv'
            }
            
            response = requests.get(tap_url, params=params)
            if response.status_code == 200:
                koi_data = pd.read_csv(io.StringIO(response.text))
                print(f"Downloaded Kepler data with {len(koi_data)} rows from TAP service")
                koi_data.to_csv('data/koi_data.csv', index=False)
                print("Kepler data saved to data/koi_data.csv")
                return koi_data
            else:
                print(f"Error downloading from TAP service: {response.status_code}")
                return None
                
    except Exception as e:
        print(f"Exception downloading Kepler data: {e}")
        return None

def download_tess_data():
    """Download TESS Objects of Interest (TOI) dataset"""
    print("Downloading TESS Objects of Interest dataset...")
    
    # Using the NASA Exoplanet Archive API to get TOI data
    url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    
    # Query for TESS Objects of Interest
    params = {
        'table': 'toi',
        'format': 'csv',
        'select': 'toi,tfopwg_disposition,toi_period,toi_duration,toi_depth,toi_rads,toi_steff,toi_slogg'
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # Parse the CSV data
            toi_data = pd.read_csv(io.StringIO(response.text))
            print(f"Downloaded TESS data with {len(toi_data)} rows")
            print("Columns:", list(toi_data.columns))
            
            # Save to data directory
            toi_data.to_csv('data/toi_data.csv', index=False)
            print("TESS data saved to data/toi_data.csv")
            
            return toi_data
        else:
            print(f"Error downloading TESS data: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Exception downloading TESS data: {e}")
        return None

def download_k2_data():
    """Download K2 dataset"""
    print("Downloading K2 dataset...")
    
    # Using the NASA Exoplanet Archive API to get K2 data
    url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    
    # Query for K2 Objects of Interest
    params = {
        'table': 'k2pandc',
        'format': 'csv',
        'select': 'k2_id,epic_name,k2_disposition,period,epoch_k2,brightness'
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # Parse the CSV data
            k2_data = pd.read_csv(io.StringIO(response.text))
            print(f"Downloaded K2 data with {len(k2_data)} rows")
            print("Columns:", list(k2_data.columns))
            
            # Save to data directory
            k2_data.to_csv('data/k2_data.csv', index=False)
            print("K2 data saved to data/k2_data.csv")
            
            return k2_data
        else:
            print(f"Error downloading K2 data: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Exception downloading K2 data: {e}")
        return None

def explore_dataset(df, name):
    """Basic exploration of the dataset"""
    if df is not None:
        print(f"\n--- {name} Dataset Exploration ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nMissing values:")
        print(df.isnull().sum())
        if 'koi_disposition' in df.columns:
            print(f"\nDisposition counts:")
            print(df['koi_disposition'].value_counts())
        elif 'tfopwg_disposition' in df.columns:
            print(f"\nDisposition counts:")
            print(df['tfopwg_disposition'].value_counts())
        elif 'k2_disposition' in df.columns:
            print(f"\nDisposition counts:")
            print(df['k2_disposition'].value_counts())
        print("-" * 40)

def main():
    """Main function to download and explore datasets"""
    print("Starting to download NASA exoplanet datasets...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download datasets
    koi_data = download_kepler_data()
    toi_data = download_tess_data() 
    k2_data = download_k2_data()
    
    # Explore datasets
    explore_dataset(koi_data, "Kepler KOI")
    explore_dataset(toi_data, "TESS TOI")
    explore_dataset(k2_data, "K2")
    
    # Summary
    print("\n" + "="*50)
    print("DATASET DOWNLOAD SUMMARY:")
    print(f"Kepler KOI: {'SUCCESS' if koi_data is not None else 'FAILED'} ({len(koi_data) if koi_data is not None else 0} rows)")
    print(f"TESS TOI: {'SUCCESS' if toi_data is not None else 'FAILED'} ({len(toi_data) if toi_data is not None else 0} rows)")
    print(f"K2: {'SUCCESS' if k2_data is not None else 'FAILED'} ({len(k2_data) if k2_data is not None else 0} rows)")
    print("="*50)

if __name__ == "__main__":
    main()
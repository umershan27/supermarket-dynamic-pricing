import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample header data
def create_header_data():
    n_records = 1000
    start_date = datetime(2024, 1, 1)
    
    data = {
        'voucher_id': [f'V{i:05d}' for i in range(n_records)],
        'godown_code': np.random.choice(['GOD001', 'GOD002', 'GOD003'], n_records),
        'voucher_date': [(start_date + timedelta(days=np.random.randint(0, 60))) for _ in range(n_records)],
        'PRICE': np.random.uniform(100, 1000, n_records),
    }
    
    # Add dateandtime
    data['dateandtime'] = [d + timedelta(hours=np.random.randint(8, 20)) for d in data['voucher_date']]
    
    df = pd.DataFrame(data)
    df.to_csv('data/sample_dataheader.csv', index=False)
    return df

# Create sample details data
def create_details_data():
    n_records = 5000
    
    data = {
        'voucher_id': np.random.choice([f'V{i:05d}' for i in range(1000)], n_records),
        'item_no': [f'ITEM{i:05d}' for i in range(n_records)],
        'quantity': np.random.randint(1, 100, n_records),
        'retail_rate': np.random.uniform(10, 1000, n_records),
    }
    
    # Add cost_rate and mrp
    data['cost_rate'] = data['retail_rate'] * np.random.uniform(0.6, 0.8, n_records)
    data['mrp'] = data['retail_rate'] * np.random.uniform(1.1, 1.3, n_records)
    data['godown_code'] = np.random.choice(['GOD001', 'GOD002', 'GOD003'], n_records)
    
    df = pd.DataFrame(data)
    df.to_csv('data/sample_datadetails.csv', index=False)
    return df

if __name__ == "__main__":
    header_df = create_header_data()
    details_df = create_details_data()
    print("Sample data created successfully!") 
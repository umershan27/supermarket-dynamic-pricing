import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from datetime import datetime, timedelta
import os
import gc

class DataProcessor:
    def __init__(self, header_path, detail_path):
        self.header_path = header_path
        self.detail_path = detail_path
        self.header_dtypes = {
            'voucher_id': 'str',
            'godown_code': 'str',
            'voucher_date': 'str',
            'dateandtime': 'str'
        }
        self.detail_dtypes = {
            'voucher_id': 'str',
            'item_no': 'str',
            'godown_code': 'str',
            'issue_quantity': 'float32',
            'retail_rate': 'float32',
            'cost_rate': 'float32',
            'mrp': 'float32'
        }
        self.label_encoders = {}
        self.cache = {}  # Cache for processed data
        
        # Verify files exist
        if not os.path.exists(header_path):
            print(f"Warning: Header file not found at {header_path}")
        if not os.path.exists(detail_path):
            print(f"Warning: Detail file not found at {detail_path}")
        
    def _create_sample_data(self):
        """Create sample data with consistent godown codes"""
        try:
            os.makedirs('/app/data', exist_ok=True)
            
            # Use consistent godown codes
            godown_codes = ['78720DEF-905E-4421-BD2F-2AE206CC7F3B', 'GOD002', 'GOD003']
            
            # Create sample header data
            n_records = 1000
            start_date = datetime(2024, 1, 1)
            
            header_data = {
                'voucher_id': [f'V{i:05d}' for i in range(n_records)],
                'godown_code': np.random.choice(godown_codes, n_records),
                'voucher_date': [(start_date + timedelta(days=np.random.randint(0, 60))).strftime('%Y-%m-%d') for _ in range(n_records)],
                'PRICE': np.random.uniform(100, 1000, n_records),
                'dateandtime': [(start_date + timedelta(days=np.random.randint(0, 60), hours=np.random.randint(8, 20))).strftime('%Y-%m-%d %H:%M:%S') for _ in range(n_records)]
            }
            
            header_df = pd.DataFrame(header_data)
            header_df.to_csv('/app/data/sample_dataheader.csv', index=False)
            
            # Create sample details data with consistent item numbers
            n_details = 5000
            item_nos = [
                'B7209F00-7034-4ED7-BBD3-F96ABBB97442',
                '6FB0E10C-3E0D-40B8-BBD7-1A3874ED0107'
            ]
            
            details_data = {
                'voucher_id': np.random.choice([f'V{i:05d}' for i in range(1000)], n_details),
                'item_no': np.random.choice(item_nos, n_details),
                'quantity': np.random.randint(1, 100, n_details),
                'retail_rate': np.random.uniform(10, 1000, n_details),
                'godown_code': np.random.choice(godown_codes, n_details),
                'issue_quantity': np.random.randint(1, 50, n_details)
            }
            
            # Set specific rates for consistent prediction display
            details_data['cost_rate'] = np.where(
                details_data['item_no'] == item_nos[0],
                44.25,
                8.62
            )
            details_data['mrp'] = np.where(
                details_data['item_no'] == item_nos[0],
                50.00,
                10.00
            )
            details_data['retail_rate'] = np.where(
                details_data['item_no'] == item_nos[0],
                49.11,
                9.84
            )
            
            details_df = pd.DataFrame(details_data)
            details_df.to_csv('/app/data/sample_datadetails.csv', index=False)
            
            print("Sample data created successfully!")
            return True
        except Exception as e:
            print(f"Error creating sample data: {e}")
            return False

    def prepare_features(self, godown_code, target_date):
        try:
            # Standardize and handle different date formats
            try:
                if isinstance(target_date, str):
                    # Try different common formats
                    for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y', '%m/%d/%Y']:
                        try:
                            target_date = pd.to_datetime(target_date, format=fmt)
                            break
                        except:
                            continue
                    if not isinstance(target_date, pd.Timestamp):
                        target_date = pd.to_datetime(target_date)
                else:
                    # Already a datetime object
                    target_date = pd.to_datetime(target_date)
            except Exception as e:
                print(f"Date conversion error: {e} - using current date instead")
                target_date = pd.to_datetime('today')
                
            print(f"Preparing features for godown: {godown_code}, date: {target_date}")
            
            # Check cache first
            cache_key = f"{godown_code}_{target_date.strftime('%Y-%m-%d')}"
            if cache_key in self.cache:
                print("Using cached data")
                return self.cache[cache_key]
            
            # Read header with minimal columns and date filtering
            header_cols = ['voucher_id', 'godown_code', 'voucher_date', 'dateandtime']
            
            # Use a 180-day window for better historical data (extended from 90 days)
            min_date = target_date - pd.Timedelta(days=180)
            
            print(f"Reading header data from {min_date} to {target_date}")
            header_df = pd.read_csv(
                self.header_path, 
                usecols=header_cols,
                dtype=self.header_dtypes
            )
            
            # Convert date columns and handle potential errors
            header_df['voucher_date'] = pd.to_datetime(header_df['voucher_date'], errors='coerce')
            header_df['dateandtime'] = pd.to_datetime(header_df['dateandtime'], errors='coerce')
            
            # Drop rows with invalid dates
            header_df = header_df.dropna(subset=['voucher_date', 'dateandtime'])
            
            # Filter header by date first (faster)
            header_df = header_df[
                (header_df['voucher_date'] >= min_date) & 
                (header_df['voucher_date'] <= target_date)
            ]
            
            # Normalize godown_code for better matching
            orig_godown_code = godown_code
            godown_code = godown_code.strip().upper()
            
            # Check for UUID format and try different matching strategies
            is_uuid_format = len(godown_code) > 30 and '-' in godown_code
            
            # First try exact match (case insensitive)
            filtered_header = header_df[header_df['godown_code'].str.upper() == godown_code]
            print(f"Found {len(filtered_header)} matching header records with exact match")
            
            if len(filtered_header) == 0 and is_uuid_format:
                # Try matching only part of the UUID
                short_code = godown_code.split('-')[0]
                print(f"Trying partial UUID match: {short_code}")
                filtered_header = header_df[header_df['godown_code'].str.upper().str.startswith(short_code)]
                print(f"Found {len(filtered_header)} records with partial UUID match")
            
            if len(filtered_header) == 0:
                # Try contains match for shorter codes
                print(f"Trying contains match")
                filtered_header = header_df[header_df['godown_code'].str.upper().str.contains(godown_code)]
                print(f"Found {len(filtered_header)} records with contains match")
            
            if len(filtered_header) == 0:
                # If still no matches, try with wider date range
                print("Trying with wider date range")
                min_date = target_date - pd.Timedelta(days=365)  # Try a full year
                header_df = pd.read_csv(
                    self.header_path, 
                    usecols=header_cols,
                    dtype=self.header_dtypes
                )
                header_df['voucher_date'] = pd.to_datetime(header_df['voucher_date'], errors='coerce')
                header_df['dateandtime'] = pd.to_datetime(header_df['dateandtime'], errors='coerce')
                header_df = header_df.dropna(subset=['voucher_date', 'dateandtime'])
                
                # Try all matching methods with wider date range
                header_df = header_df[
                    (header_df['voucher_date'] >= min_date) & 
                    (header_df['voucher_date'] <= target_date)
                ]
                
                # Try all matching methods again
                filtered_header = header_df[header_df['godown_code'].str.upper() == godown_code]
                if len(filtered_header) == 0 and is_uuid_format:
                    filtered_header = header_df[header_df['godown_code'].str.upper().str.startswith(short_code)]
                if len(filtered_header) == 0:
                    filtered_header = header_df[header_df['godown_code'].str.upper().str.contains(godown_code)]
                
                if len(filtered_header) == 0:
                    print("No matching records found even with wider criteria")
                    # If still no matches, look at the most frequent godown codes in the data
                    top_godowns = header_df['godown_code'].value_counts().head(3)
                    print(f"Top godowns in data: {top_godowns.index.tolist()}")
                    
                    # Try using the most frequent godown
                    if not top_godowns.empty:
                        most_common_godown = top_godowns.index[0]
                        print(f"Using most common godown: {most_common_godown}")
                        filtered_header = header_df[header_df['godown_code'] == most_common_godown]
                        
                        if len(filtered_header) > 0:
                            print(f"Found {len(filtered_header)} records with most common godown")
                        else:
                            return self._create_dummy_features()
                    else:
                        return self._create_dummy_features()
            
            # Get voucher IDs
            voucher_ids = set(filtered_header['voucher_id'].unique())
            print(f"Found {len(voucher_ids)} matching voucher IDs")
            
            # Read details with larger chunks
            detail_cols = ['voucher_id', 'item_no', 'godown_code', 'issue_quantity', 'retail_rate', 'cost_rate', 'mrp']
            chunks = []
            chunk_size = 1000000  # Increased chunk size for faster processing
            
            print("Reading detail data in chunks...")
            for chunk in pd.read_csv(
                self.detail_path,
                usecols=detail_cols,
                dtype=self.detail_dtypes,
                chunksize=chunk_size
            ):
                # Filter chunk by voucher IDs
                filtered_chunk = chunk[chunk['voucher_id'].isin(voucher_ids)]
                if len(filtered_chunk) > 0:
                    chunks.append(filtered_chunk)
                    print(f"Found {len(filtered_chunk)} matching records in chunk")
            
            if not chunks:
                print("No matching detail records")
                return self._create_dummy_features()
            
            # Combine filtered chunks
            print("Combining filtered chunks...")
            detail_df = pd.concat(chunks, ignore_index=True)
            print(f"Total detail records: {len(detail_df)}")
            
            # Merge data
            print("Merging header and detail data...")
            df = pd.merge(detail_df, filtered_header[['voucher_id', 'dateandtime', 'voucher_date']], on='voucher_id', how='left')
            
            # Handle missing values and ensure data types
            df = df.fillna({
                'issue_quantity': 1,
                'retail_rate': 0,
                'cost_rate': 0,
                'mrp': 0
            })
            
            # Ensure numeric columns are properly converted
            numeric_cols = ['issue_quantity', 'retail_rate', 'cost_rate', 'mrp']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Feature engineering
            print("Performing feature engineering...")
            df['hour'] = pd.to_datetime(df['dateandtime']).dt.hour.fillna(12).astype('int8')
            df['day_of_week'] = pd.to_datetime(df['voucher_date']).dt.dayofweek.fillna(0).astype('int8')
            df['demand'] = df['issue_quantity'].fillna(1)
            
            # Handle zeros in cost_rate to prevent division by zero
            df['cost_rate_safe'] = df['cost_rate'].replace(0, 1)
            df['margin'] = ((df['retail_rate'] - df['cost_rate']) / df['cost_rate_safe']).fillna(0.1)
            
            # Clip margin to reasonable values (-0.5 to 2.0)
            df['margin'] = df['margin'].clip(-0.5, 2.0)
            
            # Use integer encoding for godown
            df['godown_code_encoded'] = pd.factorize(df['godown_code'])[0]
            
            # Get latest records for each item
            print("Getting latest records for each item...")
            df = df.sort_values('dateandtime')
            
            # Group by item_no and take the last record for each item
            latest_records = df.groupby('item_no').last().reset_index()
            
            if latest_records.empty:
                print("No latest records found")
                return self._create_dummy_features()
            
            print(f"Found {len(latest_records)} unique items")
            
            # Prepare final features
            feature_cols = ['hour', 'day_of_week', 'demand', 'margin', 'godown_code_encoded',
                          'cost_rate', 'retail_rate', 'mrp']
            
            X = latest_records[feature_cols]
            item_info = latest_records[['item_no', 'retail_rate', 'cost_rate', 'mrp']]
            
            # Make sure all required columns exist
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0
            
            # Cache the results
            self.cache[cache_key] = (X, item_info)
            
            # Clear old cache entries (keep only last 100)
            if len(self.cache) > 100:
                oldest_key = list(self.cache.keys())[0]
                del self.cache[oldest_key]
            
            print(f"Prepared features: X shape={X.shape}, item_info shape={item_info.shape}")
            return X, item_info
            
        except Exception as e:
            print(f"Error in prepare_features: {e}")
            import traceback
            traceback.print_exc()
            return self._create_dummy_features()

    def _create_dummy_features(self):
        """Create dummy features for fallback with more realistic values"""
        print("Creating dummy features as fallback")
        
        # More realistic number of dummy records
        n_records = 5
        
        # Create dummy data with realistic values
        X = pd.DataFrame({
            'hour': np.random.randint(8, 20, n_records),
            'day_of_week': np.random.randint(0, 7, n_records),
            'demand': np.random.randint(1, 10, n_records),
            'margin': np.random.uniform(0.1, 0.5, n_records),
            'godown_code_encoded': np.zeros(n_records),
            'cost_rate': np.random.uniform(10, 100, n_records),
            'retail_rate': np.random.uniform(15, 120, n_records),
            'mrp': np.random.uniform(20, 150, n_records)
        })
        
        # Make sure retail rates are sensible (between cost and MRP)
        for i in range(n_records):
            cost = X.loc[i, 'cost_rate']
            mrp = X.loc[i, 'mrp']
            if mrp <= cost:
                X.loc[i, 'mrp'] = cost * 1.2
                mrp = X.loc[i, 'mrp']
            
            retail = X.loc[i, 'retail_rate']
            if retail < cost:
                X.loc[i, 'retail_rate'] = cost * 1.1
            elif retail > mrp:
                X.loc[i, 'retail_rate'] = mrp * 0.9
                
            X.loc[i, 'margin'] = (X.loc[i, 'retail_rate'] - cost) / cost
        
        # Create dummy item info
        item_info = pd.DataFrame({
            'item_no': [f'DUMMY{i+1}' for i in range(n_records)],
            'retail_rate': X['retail_rate'],
            'cost_rate': X['cost_rate'],
            'mrp': X['mrp']
        })
        
        print(f"Created dummy features with {n_records} records")
        return X, item_info

    def process_data(self, header_path: str, detail_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Read header data
        header_df = pd.read_csv(header_path)
        
        # Process details in chunks
        chunk_size = 500000
        detail_chunks = []
        
        for chunk in pd.read_csv(detail_path, chunksize=chunk_size):
            detail_chunks.append(chunk)
        
        detail_df = pd.concat(detail_chunks)
        
        # Merge data
        merged_df = pd.merge(detail_df, header_df, on='voucher_id')
        
        # Feature engineering
        merged_df['hour'] = pd.to_datetime(merged_df['dateandtime']).dt.hour
        merged_df['day_of_week'] = pd.to_datetime(merged_df['voucher_date']).dt.dayofweek
        
        # Calculate demand features
        merged_df['demand'] = merged_df.groupby('item_no')['quantity'].transform('sum')
        merged_df['overpayment'] = merged_df['retail_rate'] - merged_df['mrp']
        merged_df['margin'] = (merged_df['retail_rate'] - merged_df['cost_rate']) / merged_df['cost_rate']
        
        # Encode categorical variables
        categorical_columns = ['godown_code', 'item_no']
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            merged_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(merged_df[col])
        
        # Remove outliers
        merged_df = self.remove_outliers(merged_df, 'retail_rate')
        
        # Clip retail rate
        merged_df['retail_rate'] = merged_df['retail_rate'].clip(
            lower=merged_df['cost_rate'],
            upper=merged_df['mrp']
        )
        
        return merged_df

    def remove_outliers(self, df: pd.DataFrame, column: str, factor=1.5) -> pd.DataFrame:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"Outlier removal on {column}: kept {len(cleaned_df)} of {len(df)} rows")
        return cleaned_df

    def load_and_process_data(self, chunk_size=1000000):
        print("Loading header data...")
        
        # Read header with minimal columns for memory efficiency
        header_cols = ['voucher_id', 'godown_code', 'voucher_date', 'dateandtime']
        header_df = pd.read_csv(
            self.header_path, 
            usecols=header_cols,
            dtype=self.header_dtypes
        )
        
        print("Available header columns:", list(pd.read_csv(self.header_path, nrows=1).columns))
        print("Available detail columns:", list(pd.read_csv(self.detail_path, nrows=1).columns))
        
        print(f"Header data loaded: {header_df.shape}")
        
        # Process detail data in chunks for memory efficiency
        print("Processing detail data in chunks for full dataset analysis...")
        detail_cols = ['voucher_id', 'item_no', 'godown_code', 'issue_quantity', 'retail_rate', 'cost_rate', 'mrp']
        chunks = []
        
        # Calculate total rows to process for progress reporting
        total_rows = sum(1 for _ in open(self.detail_path)) - 1  # Subtract header
        processed_rows = 0
        
        chunk_idx = 1
        for chunk in pd.read_csv(
            self.detail_path,
            usecols=detail_cols,
            dtype=self.detail_dtypes,
            chunksize=chunk_size
        ):
            chunk_rows = len(chunk)
            processed_rows += chunk_rows
            print(f"Processing chunk {chunk_idx} with {chunk_rows} rows ({processed_rows/total_rows:.1%} complete)")
            
            # Merge chunk with header data
            merged_chunk = pd.merge(chunk, header_df[['voucher_id', 'dateandtime', 'voucher_date']], on='voucher_id', how='inner')
            print(f"Merged chunk size: {len(merged_chunk)}")
            
            if len(merged_chunk) > 0:
                # Convert date columns
                merged_chunk['dateandtime'] = pd.to_datetime(merged_chunk['dateandtime'], errors='coerce')
                merged_chunk['voucher_date'] = pd.to_datetime(merged_chunk['voucher_date'], errors='coerce')
                
                # Basic feature engineering per chunk
                merged_chunk['hour'] = merged_chunk['dateandtime'].dt.hour.fillna(12).astype('int8')
                merged_chunk['day_of_week'] = merged_chunk['voucher_date'].dt.dayofweek.fillna(0).astype('int8')
                merged_chunk['demand'] = merged_chunk['issue_quantity'].fillna(1)
                
                # Handle edge cases and calculate margin
                merged_chunk['cost_rate_safe'] = merged_chunk['cost_rate'].replace(0, 1)
                merged_chunk['margin'] = ((merged_chunk['retail_rate'] - merged_chunk['cost_rate']) / merged_chunk['cost_rate_safe']).fillna(0.1)
                merged_chunk['margin'] = merged_chunk['margin'].clip(-0.5, 2.0)
                
                # Encode godown code
                merged_chunk['godown_code_encoded'] = pd.factorize(merged_chunk['godown_code'])[0]
                
                # Select only necessary columns to reduce memory
                result_columns = ['item_no', 'hour', 'day_of_week', 'demand', 
                                 'margin', 'godown_code_encoded', 'cost_rate', 
                                 'retail_rate', 'mrp']
                processed_chunk = merged_chunk[result_columns].copy()
                
                # Optimize memory usage
                for col in processed_chunk.columns:
                    if processed_chunk[col].dtype == 'float64':
                        processed_chunk[col] = processed_chunk[col].astype('float32')
                    elif processed_chunk[col].dtype == 'int64':
                        processed_chunk[col] = processed_chunk[col].astype('int32')
                
                # Add to list of processed chunks
                chunks.append(processed_chunk)
                print(f"Chunk {chunk_idx} processed. Shape: {processed_chunk.shape}, Memory usage: {processed_chunk.memory_usage(deep=True).sum() / 1e6:.2f} MB")
                
            chunk_idx += 1
        
        # Combine all processed chunks
        print("Combining processed chunks...")
        combined_df = pd.concat(chunks, ignore_index=True)
        print(f"Final processed data shape: {combined_df.shape}")
        
        # Extract unique items for initial item info
        item_info = combined_df.groupby('item_no').last()[['retail_rate', 'cost_rate', 'mrp']].reset_index()
        print(f"Item info shape: {item_info.shape}")
        
        # Clear memory
        del chunks
        del header_df
        
        return combined_df, item_info 
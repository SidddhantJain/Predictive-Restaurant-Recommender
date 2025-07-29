import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RestaurantRecommendationEngine:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        
        # Load training data
        self.train_customers = pd.read_csv('Train/train_customers.csv')
        self.train_locations = pd.read_csv('Train/train_locations.csv')
        self.train_orders = pd.read_csv('Train/orders.csv', low_memory=False)
        self.vendors = pd.read_csv('Train/vendors.csv')
        
        # Load test data
        self.test_customers = pd.read_csv('Test/test_customers.csv')
        self.test_locations = pd.read_csv('Test/test_locations.csv')
        
        # Load sample submission
        self.sample_submission = pd.read_csv('SampleSubmission.csv')
        
        print(f"Train customers: {self.train_customers.shape}")
        print(f"Train locations: {self.train_locations.shape}")
        print(f"Train orders: {self.train_orders.shape}")
        print(f"Vendors: {self.vendors.shape}")
        print(f"Test customers: {self.test_customers.shape}")
        print(f"Test locations: {self.test_locations.shape}")
        
    def explore_data(self):
        """Explore the datasets to understand patterns"""
        print("\n=== DATA EXPLORATION ===")
        
        # Order statistics
        print(f"\nOrder Statistics:")
        print(f"Total orders: {len(self.train_orders)}")
        print(f"Unique customers in orders: {self.train_orders['customer_id'].nunique()}")
        print(f"Unique vendors: {self.train_orders['vendor_id'].nunique()}")
        print(f"Average grand total: ${self.train_orders['grand_total'].mean():.2f}")
        print(f"Average vendor rating: {self.train_orders['vendor_rating'].mean():.2f}")
        
        # Customer behavior
        orders_per_customer = self.train_orders.groupby('customer_id').size()
        print(f"\nCustomer Behavior:")
        print(f"Average orders per customer: {orders_per_customer.mean():.2f}")
        print(f"Max orders by a customer: {orders_per_customer.max()}")
        
        # Vendor popularity
        orders_per_vendor = self.train_orders.groupby('vendor_id').size()
        print(f"\nVendor Popularity:")
        print(f"Average orders per vendor: {orders_per_vendor.mean():.2f}")
        print(f"Most popular vendor orders: {orders_per_vendor.max()}")
        
    def feature_engineering(self):
        """Create features for the recommendation model"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Merge datasets to create comprehensive training data
        # Start with orders as the base
        train_data = self.train_orders.copy()
        
        # Add customer information
        train_data = train_data.merge(self.train_customers, on='customer_id', how='left')
        
        # Add location information
        train_data = train_data.merge(
            self.train_locations, 
            left_on=['customer_id', 'LOCATION_NUMBER'], 
            right_on=['customer_id', 'location_number'], 
            how='left'
        )
        
        # Add vendor information
        train_data = train_data.merge(self.vendors[['id', 'latitude', 'longitude', 'vendor_tag_name']], 
                                    left_on='vendor_id', right_on='id', how='left')
        
        # Feature engineering
        # 1. Customer features
        train_data['customer_age'] = 2024 - train_data['dob']
        train_data['customer_age'].fillna(train_data['customer_age'].median(), inplace=True)
        
        # 2. Distance between customer and vendor
        train_data['distance_to_vendor'] = np.sqrt(
            (train_data['latitude_x'] - train_data['latitude_y'])**2 + 
            (train_data['longitude_x'] - train_data['longitude_y'])**2
        )
        
        # 3. Time-based features
        train_data['created_at_orders'] = pd.to_datetime(train_data['created_at_x'])
        train_data['order_hour'] = train_data['created_at_orders'].dt.hour
        train_data['order_day_of_week'] = train_data['created_at_orders'].dt.dayofweek
        
        # 4. Customer history features
        customer_stats = self.train_orders.groupby('customer_id').agg({
            'grand_total': ['mean', 'std', 'count'],
            'vendor_rating': 'mean',
            'is_favorite': 'sum',
            'item_count': 'mean'
        }).reset_index()
        customer_stats.columns = ['customer_id', 'avg_spend', 'spend_std', 'order_count', 
                                'avg_rating_given', 'favorite_count', 'avg_items']
        
        train_data = train_data.merge(customer_stats, on='customer_id', how='left')
        
        # 5. Vendor features
        vendor_stats = self.train_orders.groupby('vendor_id').agg({
            'grand_total': 'mean',
            'vendor_rating': 'mean',
            'delivery_time': 'mean',
            'preparationtime': 'mean'
        }).reset_index()
        vendor_stats.columns = ['vendor_id', 'vendor_avg_order_value', 'vendor_avg_rating',
                               'vendor_avg_delivery_time', 'vendor_avg_prep_time']
        
        train_data = train_data.merge(vendor_stats, on='vendor_id', how='left')
        
        # Handle missing values
        numeric_columns = train_data.select_dtypes(include=[np.number]).columns
        train_data[numeric_columns] = train_data[numeric_columns].fillna(train_data[numeric_columns].median())
        
        # Encode categorical variables
        categorical_columns = ['gender', 'language', 'payment_mode', 'location_type', 'vendor_tag_name']
        for col in categorical_columns:
            if col in train_data.columns:
                le = LabelEncoder()
                train_data[col] = le.fit_transform(train_data[col].astype(str))
                self.encoders[col] = le
        
        # Select features for modeling
        self.feature_columns = [
            'customer_age', 'gender', 'language', 'status', 'verified',
            'distance_to_vendor', 'order_hour', 'order_day_of_week',
            'avg_spend', 'spend_std', 'order_count', 'avg_rating_given', 'favorite_count', 'avg_items',
            'vendor_avg_order_value', 'vendor_avg_rating', 'vendor_avg_delivery_time', 'vendor_avg_prep_time',
            'location_type', 'item_count', 'payment_mode', 'is_favorite'
        ]
        
        # Filter available columns
        self.feature_columns = [col for col in self.feature_columns if col in train_data.columns]
        
        print(f"Created {len(self.feature_columns)} features: {self.feature_columns}")
        
        return train_data
    
    def train_models(self, train_data):
        """Train multiple models for different prediction targets"""
        print("\n=== TRAINING MODELS ===")
        
        # Prepare features
        X = train_data[self.feature_columns]
        
        # Target 1: Predict grand total (order value)
        y_total = train_data['grand_total']
        X_train, X_test, y_train_total, y_test_total = train_test_split(
            X, y_total, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest for grand total prediction
        rf_total = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_total.fit(X_train_scaled, y_train_total)
        
        # Predictions and evaluation
        y_pred_total = rf_total.predict(X_test_scaled)
        
        print(f"\nGrand Total Prediction Model:")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test_total, y_pred_total)):.2f}")
        print(f"MAE: {mean_absolute_error(y_test_total, y_pred_total):.2f}")
        print(f"R²: {r2_score(y_test_total, y_pred_total):.3f}")
        
        self.models['grand_total'] = rf_total
        
        # Target 2: Predict vendor rating
        rating_data = train_data[train_data['vendor_rating'].notna()]
        if len(rating_data) > 0:
            X_rating = rating_data[self.feature_columns]
            y_rating = rating_data['vendor_rating']
            
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_rating, y_rating, test_size=0.2, random_state=42
            )
            
            X_train_r_scaled = self.scaler.transform(X_train_r)
            X_test_r_scaled = self.scaler.transform(X_test_r)
            
            rf_rating = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_rating.fit(X_train_r_scaled, y_train_r)
            
            y_pred_rating = rf_rating.predict(X_test_r_scaled)
            
            print(f"\nVendor Rating Prediction Model:")
            print(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_rating)):.3f}")
            print(f"MAE: {mean_absolute_error(y_test_r, y_pred_rating):.3f}")
            print(f"R²: {r2_score(y_test_r, y_pred_rating):.3f}")
            
            self.models['vendor_rating'] = rf_rating
        
        # Plot feature importance
        self.plot_feature_importance(rf_total, 'Grand Total Model')
        
    def plot_feature_importance(self, model, title):
        """Plot feature importance"""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]  # Top 15 features
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {title}')
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [self.feature_columns[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def create_test_features(self):
        """Create features for test data"""
        print("\n=== CREATING TEST FEATURES ===")
        
        # Create all customer-vendor combinations for test set
        test_combinations = []
        
        for _, customer in self.test_customers.iterrows():
            customer_id = customer['customer_id']
            customer_locations = self.test_locations[self.test_locations['customer_id'] == customer_id]
            
            for _, location in customer_locations.iterrows():
                for _, vendor in self.vendors.iterrows():
                    combo = {
                        'customer_id': customer_id,
                        'vendor_id': vendor['id'],
                        'location_number': location['location_number'],
                        'CID X LOC_NUM X VENDOR': f"{customer_id} X {location['location_number']} X {vendor['id']}"
                    }
                    
                    # Add customer features
                    combo.update({
                        'customer_age': 2024 - customer['dob'] if pd.notna(customer['dob']) else 30,
                        'gender': customer['gender'],
                        'language': customer['language'],
                        'status': customer['status'],
                        'verified': customer['verified']
                    })
                    
                    # Add location features
                    combo.update({
                        'location_type': location['location_type'],
                        'distance_to_vendor': np.sqrt(
                            (location['latitude'] - vendor['latitude'])**2 + 
                            (location['longitude'] - vendor['longitude'])**2
                        )
                    })
                    
                    # Add time features (using current time as default)
                    combo.update({
                        'order_hour': 12,  # Default lunch time
                        'order_day_of_week': 1  # Default weekday
                    })
                    
                    # Add default values for customer history (new customers)
                    combo.update({
                        'avg_spend': 50.0,  # Default values for new customers
                        'spend_std': 20.0,
                        'order_count': 1,
                        'avg_rating_given': 4.0,
                        'favorite_count': 0,
                        'avg_items': 2,
                        'item_count': 2,
                        'payment_mode': 'card',
                        'is_favorite': 0
                    })
                    
                    # Add vendor features from precomputed stats
                    if hasattr(self, 'vendor_stats'):
                        vendor_stats = self.train_orders[self.train_orders['vendor_id'] == vendor['id']]
                        if len(vendor_stats) > 0:
                            combo.update({
                                'vendor_avg_order_value': vendor_stats['grand_total'].mean(),
                                'vendor_avg_rating': vendor_stats['vendor_rating'].mean(),
                                'vendor_avg_delivery_time': vendor_stats['delivery_time'].mean(),
                                'vendor_avg_prep_time': vendor_stats['preparationtime'].mean()
                            })
                        else:
                            combo.update({
                                'vendor_avg_order_value': 50.0,
                                'vendor_avg_rating': 4.0,
                                'vendor_avg_delivery_time': 30.0,
                                'vendor_avg_prep_time': 20.0
                            })
                    
                    test_combinations.append(combo)
        
        test_df = pd.DataFrame(test_combinations)
        
        # Encode categorical variables using fitted encoders
        for col, encoder in self.encoders.items():
            if col in test_df.columns:
                # Handle unseen categories
                test_df[col] = test_df[col].astype(str)
                test_df[col] = test_df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0
                )
        
        print(f"Created {len(test_df)} test combinations")
        return test_df
    
    def generate_recommendations(self, test_df):
        """Generate recommendations for test customers"""
        print("\n=== GENERATING RECOMMENDATIONS ===")
        
        # Prepare features
        X_test = test_df[self.feature_columns]
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict scores using available models
        test_df['predicted_grand_total'] = self.models['grand_total'].predict(X_test_scaled)
        
        if 'vendor_rating' in self.models:
            test_df['predicted_rating'] = self.models['vendor_rating'].predict(X_test_scaled)
            # Combine predictions (weighted average)
            test_df['recommendation_score'] = (
                0.7 * (test_df['predicted_grand_total'] / test_df['predicted_grand_total'].max()) +
                0.3 * (test_df['predicted_rating'] / 5.0)
            )
        else:
            test_df['recommendation_score'] = test_df['predicted_grand_total']
        
        # Adjust score based on distance (closer restaurants get higher scores)
        max_distance = test_df['distance_to_vendor'].max()
        distance_penalty = test_df['distance_to_vendor'] / max_distance
        test_df['recommendation_score'] = test_df['recommendation_score'] * (1 - 0.2 * distance_penalty)
        
        # Create submission file
        submission = test_df[['CID X LOC_NUM X VENDOR', 'recommendation_score']].copy()
        submission.columns = ['CID X LOC_NUM X VENDOR', 'target']
        
        # Normalize scores to match expected range
        submission['target'] = (submission['target'] - submission['target'].min()) / (
            submission['target'].max() - submission['target'].min()
        )
        
        return submission
    
    def run_pipeline(self):
        """Run the complete recommendation pipeline"""
        print("=== RESTAURANT RECOMMENDATION ENGINE ===")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Feature engineering
        train_data = self.feature_engineering()
        
        # Step 4: Train models
        self.train_models(train_data)
        
        # Step 5: Create test features
        test_df = self.create_test_features()
        
        # Step 6: Generate recommendations
        submission = self.generate_recommendations(test_df)
        
        # Step 7: Save submission
        submission.to_csv('restaurant_recommendations.csv', index=False)
        print(f"\nRecommendations saved to 'restaurant_recommendations.csv'")
        print(f"Submission shape: {submission.shape}")
        print(f"Sample predictions:")
        print(submission.head(10))
        
        return submission

# Run the recommendation engine
if __name__ == "__main__":
    engine = RestaurantRecommendationEngine()
    recommendations = engine.run_pipeline()
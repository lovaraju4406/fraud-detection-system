"""
Fraud Detection Prediction Script with Business Logic
Usage: python src/predict.py
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import json


class FraudDetectorOptimized:
    """
    Fraud Detection System with Business Logic
    """
    def __init__(self, model, optimal_threshold=0.7):
        self.model = model
        self.optimal_threshold = optimal_threshold
        self.low_risk_threshold = 0.4
        self.high_risk_threshold = 0.7
    
    def predict_proba(self, X):
        """Get fraud probability"""
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """Predict with optimal threshold"""
        probabilities = self.predict_proba(X)
        return (probabilities >= self.optimal_threshold).astype(int)
    
    def get_risk_level(self, X):
        """Classify into risk levels"""
        probabilities = self.predict_proba(X)
        risk_levels = []
        
        for prob in probabilities:
            if prob < self.low_risk_threshold:
                risk_levels.append('Low')
            elif prob < self.high_risk_threshold:
                risk_levels.append('Medium')
            else:
                risk_levels.append('High')
        
        return np.array(risk_levels)
    
    def get_action(self, X):
        """Recommend action for each transaction"""
        risk_levels = self.get_risk_level(X)
        probabilities = self.predict_proba(X)
        
        actions = []
        for risk, prob in zip(risk_levels, probabilities):
            if risk == 'Low':
                action = 'Approve'
            elif risk == 'Medium':
                action = 'Request OTP'
            else:
                action = 'Block & Review'
            
            actions.append({
                'risk_level': risk,
                'probability': float(prob),
                'action': action
            })
        
        return actions


class FraudPredictor:
    """Production-ready fraud detection with business rules"""
    
    def __init__(self, model_path='models/fraud_detector_xgboost.pkl'):
        print("📂 Loading fraud detection system...")
        
        try:
            # Try loading optimized detector
            with open(model_path, 'rb') as f:
                loaded_obj = pickle.load(f)
            
            # Check if it's a FraudDetectorOptimized object
            if isinstance(loaded_obj, FraudDetectorOptimized):
                self.detector = loaded_obj
                print("✅ Optimized detector loaded!")
            else:
                # If it's a raw model, wrap it
                self.detector = FraudDetectorOptimized(loaded_obj, optimal_threshold=0.7)
                print("✅ Base model loaded and wrapped!")
                
        except Exception as e:
            print(f"⚠️  Error loading pickle: {e}")
            print("   Trying joblib...")
            # Fallback: Load with joblib
            base_model = joblib.load(model_path)
            self.detector = FraudDetectorOptimized(base_model, optimal_threshold=0.7)
            print("✅ Model loaded with joblib!")
    
    def predict_single(self, transaction_features):
        """
        Predict fraud for a single transaction
        
        Parameters:
        -----------
        transaction_features : dict or pd.DataFrame
            Transaction features (must match training features)
        
        Returns:
        --------
        dict : Prediction result with risk level and action
        """
        # Convert to DataFrame if dict
        if isinstance(transaction_features, dict):
            transaction_features = pd.DataFrame([transaction_features])
        
        # Get prediction
        predictions = self.detector.get_action(transaction_features)
        
        return predictions[0]
    
    def predict_batch(self, transactions_df):
        """
        Predict fraud for multiple transactions
        
        Parameters:
        -----------
        transactions_df : pd.DataFrame
            Multiple transactions
        
        Returns:
        --------
        list : List of prediction results
        """
        return self.detector.get_action(transactions_df)
    
    def explain_prediction(self, prediction):
        """Generate human-readable explanation"""
        
        risk = prediction['risk_level']
        prob = prediction['probability']
        action = prediction['action']
        
        explanation = {
            'fraud_probability': f"{prob:.1%}",
            'risk_assessment': risk,
            'recommended_action': action,
            'reasoning': self._get_reasoning(risk, prob),
            'customer_message': self._get_customer_message(action)
        }
        
        return explanation
    
    def _get_reasoning(self, risk, prob):
        """Internal: Generate reasoning"""
        if risk == 'Low':
            return f"Low fraud probability ({prob:.1%}). Transaction pattern is normal."
        elif risk == 'Medium':
            return f"Moderate fraud probability ({prob:.1%}). Some suspicious patterns detected."
        else:
            return f"High fraud probability ({prob:.1%}). Multiple fraud indicators present."
    
    def _get_customer_message(self, action):
        """Internal: Generate customer-facing message"""
        messages = {
            'Approve': "Your transaction has been approved.",
            'Request OTP': "For your security, please verify this transaction with the OTP sent to your phone.",
            'Block & Review': "This transaction has been blocked for security reasons. Please contact customer support."
        }
        return messages.get(action, "Processing...")


def main():
    """Demo: Test fraud prediction"""
    print("\n" + "="*70)
    print("FRAUD DETECTION SYSTEM - PREDICTION DEMO")
    print("="*70 + "\n")
    
    # Load predictor
    predictor = FraudPredictor()
    
    # Sample transaction (31 features: V1-V28 + Amount + Hour + Amount_Log)
    sample_transaction = {
        'V1': -1.35, 'V2': -0.07, 'V3': 2.54, 'V4': 1.38,
        'V5': -0.33, 'V6': 0.46, 'V7': 0.24, 'V8': 0.09,
        'V9': 0.36, 'V10': 0.09, 'V11': -0.55, 'V12': -0.62,
        'V13': -0.99, 'V14': -0.31, 'V15': 1.47, 'V16': -0.47,
        'V17': 0.21, 'V18': 0.03, 'V19': 0.40, 'V20': 0.25,
        'V21': -0.02, 'V22': 0.28, 'V23': -0.11, 'V24': 0.07,
        'V25': 0.13, 'V26': -0.19, 'V27': 0.13, 'V28': -0.02,
        'Amount': 125.50,
        'Hour': 14.5,
        'Amount_Log': 4.83
    }
    
    print("🔍 Analyzing transaction...")
    print(f"   Amount: ₹{sample_transaction['Amount']}")
    print(f"   Hour: {sample_transaction['Hour']}")
    
    # Get prediction
    prediction = predictor.predict_single(sample_transaction)
    
    # Explain
    explanation = predictor.explain_prediction(prediction)
    
    print("\n📊 PREDICTION RESULT:")
    print("-"*70)
    print(f"   Fraud Probability:    {explanation['fraud_probability']}")
    print(f"   Risk Assessment:      {explanation['risk_assessment']}")
    print(f"   Recommended Action:   {explanation['recommended_action']}")
    print(f"\n💡 Reasoning: {explanation['reasoning']}")
    print(f"\n📱 Customer Message: {explanation['customer_message']}")
    print("-"*70)
    
    print("\n✅ Prediction complete!\n")


if __name__ == "__main__":
    main()
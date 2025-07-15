#!/usr/bin/env python3
"""
Test script to demonstrate backend validation approach
"""

# Simulate the backend validation
class BackendValidator:
    @staticmethod
    def validate_config(config):
        """Validate configuration and return errors"""
        errors = {}
        
        # Test validation for metadata
        if 'Metadata' in config:
            metadata = config['Metadata']
            if 'num_processors' in metadata:
                value = metadata['num_processors']
                if value < 1:
                    errors['num_processors'] = "Number of processors must be at least 1"
                elif value > 128:
                    errors['num_processors'] = "Number of processors seems too large (>128)"
        
        # Test validation for Potts Core
        if 'PottsCore' in config:
            potts = config['PottsCore']
            if 'dim_x' in potts:
                value = potts['dim_x']
                if value < 1:
                    errors['dim_x'] = "X dimension must be at least 1"
                elif value > 1000:
                    errors['dim_x'] = "X dimension seems too large (>1000)"
        
        return errors

# Test the validation
def test_backend_validation():
    print("🧪 Testing Backend Validation Approach")
    
    # Test case 1: Valid configuration
    print("\n1️⃣ Testing valid configuration:")
    valid_config = {
        'Metadata': {'num_processors': 4},
        'PottsCore': {'dim_x': 100}
    }
    
    errors = BackendValidator.validate_config(valid_config)
    if errors:
        print(f"❌ Unexpected errors: {errors}")
    else:
        print("✅ Valid configuration passed")
    
    # Test case 2: Invalid configuration
    print("\n2️⃣ Testing invalid configuration:")
    invalid_config = {
        'Metadata': {'num_processors': -1},  # Invalid: negative
        'PottsCore': {'dim_x': 2000}         # Invalid: too large
    }
    
    errors = BackendValidator.validate_config(invalid_config)
    if errors:
        print("✅ Validation correctly caught errors:")
        for field, error in errors.items():
            print(f"  {field}: {error}")
    else:
        print("❌ Validation failed to catch errors")
    
    # Test case 3: Mixed valid/invalid
    print("\n3️⃣ Testing mixed configuration:")
    mixed_config = {
        'Metadata': {'num_processors': 8},   # Valid
        'PottsCore': {'dim_x': -5}           # Invalid: negative
    }
    
    errors = BackendValidator.validate_config(mixed_config)
    print(f"Errors found: {len(errors)}")
    for field, error in errors.items():
        print(f"  {field}: {error}")
    
    print("\n✅ Backend validation test complete!")

if __name__ == "__main__":
    test_backend_validation()

"""
Validators
Data validation and integrity checking utilities
"""
import re
from typing import Any, List, Dict, Optional

class DataValidator:
    """Validates data integrity and constraints"""
    
    @staticmethod
    def validate_id(id_str: str) -> bool:
        """Validate UUID format"""
        uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
        return bool(re.match(uuid_pattern, id_str))
    
    @staticmethod
    def validate_name(name: str) -> bool:
        """Validate name format"""
        return len(name.strip()) > 0 and len(name) <= 100
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float) -> bool:
        """Validate value is within range"""
        return min_val <= value <= max_val
    
    @staticmethod
    def validate_enum(value: Any, enum_values: List) -> bool:
        """Validate value is in allowed enum"""
        return value in enum_values
"""
Week 1 Exercise Solutions
This file contains solutions to the Week 1 exercises.
"""

# Exercise 2A: Data Types and Variables

def data_types_demo():
    """Solution to Exercise 2A: Data Types and Variables"""
    # 1. Different data types
    integer_var = 42
    float_var = 3.14159
    string_var = "Hello, AI!"
    boolean_var = True
    list_var = [1, 2, 3, 4, 5]
    dict_var = {"name": "AI Model", "version": 1.0, "is_trained": False}
    
    # 2. Function that returns sum and product
    def sum_and_product(a, b):
        return a + b, a * b
    
    # 3. List of AI applications
    ai_applications = ["Chatbots", "Image Recognition", "Autonomous Vehicles", 
                      "Recommendation Systems", "Fraud Detection"]
    
    print("AI Applications:")
    for app in ai_applications:
        print(f"- {app}")
    
    return {
        "integer_var": integer_var,
        "float_var": float_var,
        "string_var": string_var,
        "boolean_var": boolean_var,
        "list_var": list_var,
        "dict_var": dict_var,
        "sum_product_3_4": sum_and_product(3, 4)
    }

# Exercise 2B: Functions and Control Flow

def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def filter_evens(numbers):
    """Return a new list with only even numbers."""
    return [num for num in numbers if num % 2 == 0]

class SimpleCalculator:
    """A simple calculator class."""
    
    @staticmethod
    def add(a, b):
        return a + b
    
    @staticmethod
    def subtract(a, b):
        return a - b
    
    @staticmethod
    def multiply(a, b):
        return a * b
    
    @staticmethod
    def divide(a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# Example usage
if __name__ == "__main__":
    # Test Exercise 2A
    print("=== Exercise 2A Solutions ===")
    data_types_demo()
    
    # Test Exercise 2B
    print("\n=== Exercise 2B Solutions ===")
    print(f"Is 17 prime? {is_prime(17)}")
    print(f"Even numbers from [1, 2, 3, 4, 5, 6]: {filter_evens([1, 2, 3, 4, 5, 6])}")
    
    calc = SimpleCalculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 * 7 = {calc.multiply(6, 7)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")

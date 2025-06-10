# 🏠 Real Estate Estimator

A NumPy-based linear algebra project to predict flat prices using linear regression mathematics.

## 🎯 Project Overview

This project explores whether there exists a **LINEAR RELATIONSHIP** between:
- The price of a flat (our target variable)
- Common features such as surface area, bedrooms, and floor number

**Key Constraint**: ❗️ Pandas is forbidden in this challenge ❗️

## 📊 Dataset

We collected data for 4 flats with the following features:

| Flat  | Surface (sq ft) | Bedrooms | Floors | Price (k USD) |
|-------|----------------|----------|---------|---------------|
| flat1 | 620            | 1        | 1       | 244           |
| flat2 | 3280           | 4        | 2       | 671           |
| flat3 | 1900           | 2        | 2       | 504           |
| flat4 | 1320           | 3        | 3       | 510           |

## 🧮 Mathematical Approach

We solve the linear system of equations:

```
X · θ = y
```

Where:
- **y** is the target vector (prices)
- **X** is the matrix of features
- **θ** (theta) is the vector of coefficients to be found

The coefficients represent:
- θ₀: Base price for a flat with no features
- θ₁: Price increase per square foot
- θ₂: Price increase per additional bedroom
- θ₃: Price increase per additional floor

## 🛠️ Implementation Steps

### 1. Feature Matrix Creation
```python
import numpy as np

# Create the 4x3 matrix of features
X_features = np.array([
    [620, 1, 1],
    [3280, 4, 2],
    [1900, 2, 2],
    [1320, 3, 3]
])

# Add constant vector for y-intercept
x0 = np.ones((4,1), dtype="int")
X = np.hstack((x0, X_features))
```

### 2. Target Vector Definition
```python
# Define target prices as column vector
y = np.array([[244], [671], [504], [510]])
```

### 3. System Solution
```python
# Solve using matrix inversion: θ = X⁻¹ · y
X_inv = np.linalg.inv(X)
theta = np.dot(X_inv, y)
```

### 4. Price Prediction
```python
# Predict price for new flat (3000 sq ft, 5 bedrooms, 1 floor)
X5 = np.array([1, 3000, 5, 1])
y5 = np.dot(X5, theta)
# Result: ~$526,000
```

## 📈 Results

The model successfully predicts a flat price of approximately **$526,000** for a flat with:
- 3000 square feet
- 5 bedrooms
- 1 floor

## ⚠️ Limitations

When attempting to add a 5th observation to improve our model, we encounter a fundamental limitation:

- The resulting matrix X becomes non-square (5×4)
- Non-square matrices are not invertible
- The system X·θ = y has no exact solution

This leads us to the concept of **Linear Regression**, where instead of solving X·θ = y exactly, we find θ that minimizes the error ||X·θ - y||².

## 🔧 Technologies Used

- **NumPy**: For all matrix operations and linear algebra computations
- **Python**: Core programming language

## 📚 Key NumPy Methods Used

- `np.array()`: Array creation
- `np.ones()`: Creating vectors of ones
- `np.hstack()`: Horizontal array concatenation
- `np.linalg.inv()`: Matrix inversion
- `np.dot()`: Matrix multiplication
- `np.eye()`: Identity matrix creation
- `np.allclose()`: Numerical comparison with tolerance

## 🎓 Learning Outcomes

This project demonstrates:
- Linear algebra applications in machine learning
- Matrix operations using NumPy
- The mathematical foundation of linear regression
- Limitations of exact solutions in overdetermined systems
- The transition from exact solutions to optimization problems

## 🚀 Next Steps

Future improvements could include:
- Implementing Linear Regression to handle more observations
- Adding regularization techniques
- Exploring non-linear relationships
- Cross-validation for model evaluation

## 📝 Usage

1. Clone the repository
2. Ensure NumPy is installed: `pip install numpy`
3. Run the notebook or Python script
4. Follow the step-by-step implementation

## 🧪 Testing

The project includes automated tests to verify:
- Feature matrix shape and content
- Target vector format
- Solution accuracy
- Mathematical correctness

Run tests with:
```bash
pytest tests/
```

---

*This project serves as an introduction to the mathematical foundations of machine learning, specifically linear regression, using pure NumPy implementations.*

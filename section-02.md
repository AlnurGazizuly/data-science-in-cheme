# Section 2: Introduction to Python

Python is a versatile and beginner-friendly programming language widely used for various applications, including web development, data analysis, artificial intelligence, and scientific computing. In this section, you'll learn the fundamentals of Python, including its syntax, core concepts, and how to write basic programs.

### Exercise 2.1: Variables and Data Types

#### Explanation:
Variables are like storage containers for data in your program. In Python, you can store different types of data such as numbers (integers, floats), text (strings), and truth values (booleans). These data types are essential for performing calculations, managing conditions, and organizing information in programs. For example, in chemical engineering, you might use variables to store temperatures, pressures, or flow rates.

Python automatically determines the type of data you assign to a variable, so you don’t need to declare the type explicitly.

#### 2.1. Understanding Variables
A variable is a name that refers to a value stored in your program. In Python, you can assign values to variables using the `=` operator. Python automatically determines the type of the variable based on the value you assign.

**Example:**
```python
# Defining variables
temperature = 300  # Temperature in Kelvin
```

- **Variable Naming:** Use descriptive names (e.g., `temperature`, `pressure`, `flowrate`) that make your code easy to read and understand.
- **Comments:** Use the `#` symbol to add comments that explain what your code does.

#### 2.2. Data Types
Python supports various data types. In the example above:
- `20` is an integer (`int`). An integer is a whole number that contains no decimal points.
- `1.0` is a floating-point number (`float`). A float is a number that contains a decimal point.
- `True` is a boolean (`bool`), which can be either `True` or `False`.
- `"Flowrate"` is a string (`str`), used for text. Strings can also contain numbers, but they are represented as a text when there are surrounded by either '' or "". So while 1 is an integer, '1' is a string.

**Key Points:**
- Strings must be enclosed in single (`'`) or double (`"`) quotes.
- Boolean values are case-sensitive and must be written as `True` or `False`.

#### 2.3. Best Practices for Formatting Code
- **Indentation:** Python relies on indentation (typically 4 spaces) to define blocks of code.
- **PEP 8 Guidelines:** Follow Python's official style guide, which promotes readability and consistency.
- **Whitespace:** Use spaces around operators (`=`) for clarity.

#### 2.4. Printing and Debugging
To check the values of your variables or output information, use the `print()` function.

**Example:**
```python
print(f"Reactor temperature:")
print(f"Temperature: {temperature} K")
```
- The `f` before the string enables **f-strings**, which allow you to include variable values directly within the string using curly braces (`{}`).

#### 2.5. Installing Python
Before writing any code, ensure Python is installed on your computer. You can download it from [python.org](https://www.python.org/). It's recommended to use Python 3.x, as Python 2.x is no longer supported.

#### 2.6. Writing and Running Code
You can write Python code in:
- **Interactive Mode:** Open a terminal and type `python` (or `python3` depending on your system).
- **Scripts:** Use a text editor (e.g., VS Code, PyCharm, or Jupyter Notebook) to write your code and save it with a `.py` extension. Run it in a terminal using:
  ```bash
  python your_script.py
  ```

#### 2.7. Error Handling
If you make a mistake, Python will provide an error message. Read the error carefully to understand what went wrong and how to fix it.

**Common Errors:**
- **SyntaxError:** Missing or misplaced punctuation, such as forgetting quotes or parentheses.
- **NameError:** Using a variable that hasn’t been defined.

By practicing these basics, you'll build a strong foundation for using Python to solve chemical engineering problems effectively.

#### Task:
1. Define a variable `temperature` and assign it the value 300 (Kelvin).
2. Define a variable `pressure` and assign it the value 1.0 (atm).
3. Define a variable `is_reactor_on` and assign it the value `True`.
4. Define a string variable `reactor_status` with the value "Active".

```python
# Your code goes here
```

### Exercise 2.2: Basic Operations

#### Explanation:
Python allows basic mathematical operations such as addition (`+`), subtraction (`-`), multiplication (`*`), and division (`/`). These operations are used to calculate values and model relationships between variables. For example, the Ideal Gas Law (PV = nRT) relates pressure, volume, and temperature to the amount of gas in a system.

Python follows the standard mathematical order of operations (PEMDAS): Parentheses, Exponents, Multiplication/Division, and Addition/Subtraction.

#### 2.8. Basic Arithmetic Operations

Python provides built-in operators and functions for performing a variety of mathematical calculations. These are essential for solving engineering problems.

Addition: +

Subtraction: -

Multiplication: *

Division: /

Exponentiation: **

Modulus (remainder): %

**Example:**

```python
# Basic operations
sum_value = 5 + 3      # 8
difference = 10 - 4    # 6
product = 7 * 2        # 14
quotient = 9 / 3       # 3.0
power = 2 ** 3         # 8
remainder = 10 % 3     # 1

print(f"Sum: {sum_value}")
print(f"Difference: {difference}")
print(f"Product: {product}")
print(f"Quotient: {quotient}")
print(f"Power: {power}")
print(f"Remainder: {remainder}")
```

Python follows the standard mathematical order of operations: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction.

Example:
```python
result = 5 + 3 * 2  # 5 + (3 * 2) = 11
print(result)

result = (5 + 3) * 2  # (5 + 3) * 2 = 16
print(result)
```

#### 2.9. Using the math Module

The math module provides additional mathematical functions, including trigonometric, logarithmic, and advanced calculations.

Example:
```python
import math

# Trigonometric functions
angle_rad = math.radians(45)  # Convert degrees to radians
sin_value = math.sin(angle_rad)
cos_value = math.cos(angle_rad)

# Logarithmic functions
log_value = math.log(10)      # Natural log
log10_value = math.log10(100) # Base-10 log

print(f"Sin(45°): {sin_value:.2f}")
print(f"Cos(45°): {cos_value:.2f}")
print(f"ln(10): {log_value:.2f}")
print(f"log10(100): {log10_value:.2f}")
```

These operations and functions provide the foundation for performing numerical calculations in Python, enabling you to solve complex engineering problems efficiently.

#### Task:
1. Calculate the moles of a gas using the Ideal Gas Law: PV = nRT. Assume:
   - `P = 1` atm
   - `V = 22.4` L
   - `R = 0.0821` L.atm/(mol.K)
   - `T = 300` K
   Assign the result to a variable `moles`.
2. Compute the work done by a system using W = -PΔV, where:
   -  P = 2 atm
   - ΔV = 5 L.
3. Calculate the energy required to heat a substance using Q = mcΔT, where:
   -  m = 10 kg
   - c = 4.18 J/g.K
   -  ΔT = 30 K.

```python
# Your code goes here
```

### Exercise 2.3: Lists and Loops

#### Explanation:
Lists are used to store multiple values in a single variable. This is useful when working with repeated measurements or datasets. Loops, like `for` loops, allow you to process each value in a list, making it easy to apply calculations or transformations to multiple values.

In chemical engineering, you might use lists to store data such as temperature readings at different times or flow rates for different streams.

#### 2.10. Working with Lists

Lists are a versatile data structure in Python that can store multiple values in a single variable. Lists can hold items of any data type and are commonly used for organizing and manipulating data.

You can create a list using square brackets [] and separate items with commas.

Example:
```
# Creating a list of reactor temperatures
temperatures = [300, 350, 400, 450]

# Accessing elements
print(temperatures[0])  # First element: 300
print(temperatures[-1])  # Last element: 450
```

Lists are mutable, meaning you can change their contents.

Example:
```
# Updating a value
temperatures[1] = 360

# Adding values
temperatures.append(500)  # Add 500 to the end

# Removing values
temperatures.remove(400)  # Remove the value 400

print(temperatures)
```

Use a for loop to iterate through a list and process each element.

Example:
```
# Printing all temperatures
for temp in temperatures:
    print(f"Temperature: {temp} K")
```

#### 2.11. Using Loops

Loops allow you to repeat a block of code multiple times. Python supports two main types of loops: for loops and while loops.

for loops are used to iterate over a sequence (e.g., a list or a range of numbers).

Example:

```
# Calculating the square of each number in a list
numbers = [1, 2, 3, 4, 5]

for num in numbers:
    square = num ** 2
    print(f"The square of {num} is {square}")
```

while loops execute as long as a specified condition is True.

Example:
```
# Countdown from 5
count = 5
while count > 0:
    print(f"Countdown: {count}")
    count -= 1  # Decrease count by 1

print("Blast off!")
```

There are control statements that you can use to edit the function of a loop. If you want a loop to either skip and iteration or break entirely once it has reached a certain condition, you can use the following control statements:

break: Exit the loop entirely.

continue: Skip to the next iteration.

Example:
```
# Printing odd numbers only
for num in range(1, 10):
    if num % 2 == 0:
        continue  # Skip even numbers
    print(num)
```

Lists and loops are powerful tools for managing and processing data in Python. By combining them, you can efficiently handle large datasets and automate repetitive tasks.

#### Task:
1. Create a list `temperatures` with values [300, 310, 320, 330]. Then use a `for` loop to convert each temperature to Celsius (C = K - 273.15). Finally, print each converted value.
2. Create a list of pressures [1.0, 1.5, 2.0, 2.5] and calculate the corresponding volumes for a fixed number of moles (n = 1) and temperature (T = 298 K) using PV = nRT.
3. Create a list of flow rates [5, 10, 15] and double each value using a loop.

```python
# Your code goes here
```

### Exercise 2.4: Functions

#### Explanation:
Functions are reusable blocks of code that perform a specific task. They make your code more organized and efficient, especially for repeated calculations. Functions can take inputs (parameters), perform computations, and return outputs.

For instance, in engineering, you can use functions to calculate properties like enthalpy, entropy, or reaction rates given certain parameters.

#### 2.12. Functions

Functions are reusable blocks of code designed to perform a specific task. They allow you to organize your code, avoid repetition, and improve readability.

Use the def keyword to define a function, followed by the function name and parentheses. Parameters can be included within the parentheses.

Example:
```
# Define a function to calculate pressure using the Ideal Gas Law
def calculate_pressure(n, V, T, R=0.0821):
    """Calculate pressure using PV = nRT."""
    P = (n * R * T) / V
    return P

# Call the function
pressure = calculate_pressure(n=1, V=22.4, T=300)
print(f"Pressure: {pressure:.2f} atm")
```

In a function, the default arguments allow you to specify default values for parameters. Keyword arguments let you call a function with parameters in any order.

Example:
```
# Using default and keyword arguments
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Maya")  # Output: Hello, Maya!
greet("Maya", greeting="Hi")  # Output: Hi, Maya!
```

#### Task:
1. Write a function `ideal_gas_law` that calculates moles given P, V, R, and T. Call the function with P = 2 atm, V = 10 L, R = 0.0821, and T = 298 K.
2. Write a function to calculate the density of a substance given mass and volume.
3. Write a function to calculate Reynolds number given velocity, characteristic length, density, and viscosity.

When using functions, make sure to use `return` instead of just`print`. If you don't return the variable out of the funciton, it will just be floating around in space! You cannot use a variable outside of a function unless you return it.

Just be careful, using return is always the last thing you do! If you write code in a funciton after the return, then that code will **not run.**

```python
# Your code goes here
```

### Exercise 2.5: Numpy Basics

#### Explanation:
Numpy is a Python library that allows efficient operations on arrays and matrices. It’s particularly useful in engineering and data science for handling numerical data, performing mathematical operations, and working with multidimensional datasets.

Using numpy arrays is faster and more memory-efficient than using lists, especially for large datasets.

#### 2.13. NumPy

NumPy is a powerful library for numerical computing. It provides support for arrays, matrices, and mathematical functions.

You must first import the library, typically using the alias np, which should look like: `import numpy as np`

NumPy can be used for a variety of operations, but is most commonly used to make arrays. Arrays are the core of NumPy and are more efficient than Python lists.

Example:
```
# Create a NumPy array
pressures = np.array([1.0, 2.0, 3.0])

# Perform operations on the array
pressures_in_pascals = pressures * 101325
print(pressures_in_pascals)
```

NumPy arrays support element-wise operations.

Example:
```
# Temperature in Kelvin
temperatures = np.array([300, 350, 400])

# Convert to Celsius
temperatures_celsius = temperatures - 273.15
print(temperatures_celsius)
```

#### Task:
1. Import numpy as `np`. Create a numpy array `pressures` with values [1, 2, 3, 4] atm. Finally, multiply the array by 101.325 to convert pressures to kPa. Return the new array.
2. Create a numpy array for temperatures in Celsius [25, 50, 75, 100] and convert them to Kelvin. Return the new array.
3. Calculate the mean and standard deviation of an array representing flow rates [10, 20, 30, 40]. Retyrn the new array.

```python
# Your code goes here
```

### Exercise 2.6: Importing and Manipulating an Excel Spreadsheet

#### Explanation:
Chemical engineers often work with data in Excel spreadsheets. Python’s `pandas` library makes it easy to import, manipulate, and analyze data. Pandas can read data directly from Excel files into a DataFrame, which is a structured format similar to a table.

A DataFrame allows you to filter rows, compute statistics, and transform data efficiently.

#### 2.14. Pandas and Importing Spreadsheets

Pandas is a library for data manipulation and analysis, especially useful for working with tabular data. You need to import it as `import pandas as pd`.

Using pandas, you can read data from Excel or CSV files into a DataFrame. This is one of the most useful tools in Python, as it allows you to take pre-existing datasets and do analysis with them in Python.

Example:
```
# Read an Excel file
data = pd.read_excel("data.xlsx")

# Display the first few rows
print(data.head())
```

Use built-in methods to explore and analyze your data.

Example:
```
# Summarize data
print(data.describe())

# Select a column
temperatures = data["Temperature"]

# Filter rows
high_pressure = data[data["Pressure"] > 2]
print(high_pressure)
```

#### Task:
1. Import the pandas library. Load an Excel spreadsheet named `Section2.xlsx` into a DataFrame. Assume the file has a sheet named "Sheet1".Display the first five rows of the DataFrame.
2. Calculate the mean of a column named `FlowRate`.
3. Filter rows where `Temperature > 300`.
4. Create a new column `Pressure_kPa` by converting a column `Pressure` from atm to kPa.

```python
# Your code goes here
```

### Exercise 2.7: Data Visualization with Matplotlib and Seaborn

#### Explanation:
Visualization is crucial for understanding trends and patterns in data. `matplotlib` is a Python library for creating static, interactive, and animated visualizations. `seaborn` builds on matplotlib and provides a high-level interface for creating informative and attractive statistical graphics.

#### 2.15. Matplotlib and Seaborn

Matplotlib and Seaborn are libraries for data visualization. Follow the same procedure to import libraries as before, where:
```
import matplotlib.pyplot as plt
import seaborn as sns
```

You can use Matplotlib for simple plots. This allows you to visualize your dataset.

Example:
```
# Plot temperature vs. pressure
temperatures = [300, 350, 400]
pressures = [1, 1.5, 2]

plt.plot(temperatures, pressures, marker="o")
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (atm)")
plt.title("Temperature vs. Pressure")
plt.show()
```
Seaborn builds on Matplotlib and provides beautiful default styles. Both tools can be used together in order to create nice plots that clearly show your data.

Example:
```
# Create a scatterplot with Seaborn
data = pd.DataFrame({
    "Temperature": [300, 350, 400],
    "Pressure": [1, 1.5, 2]
})

sns.scatterplot(x="Temperature", y="Pressure", data=data)
plt.title("Temperature vs. Pressure")
plt.show()
```

#### Task:
1. Import `matplotlib.pyplot` and `seaborn`.
2. Create a line plot for a dataset where `x = [0, 1, 2, 3]` and `y = [0, 1, 4, 9]`.
3. Use seaborn to create a scatterplot of `Temperature` vs `Pressure` from the DataFrame `data`.
4. Plot a histogram of the `FlowRate` column.

```python
# Your code goes here
```

### Answer Key

#### Exercise 2.1:
```python
temperature = 300
pressure = 1.0
is_reactor_on = True
reactor_status = "Active"
```

#### Exercise 2.2:

```python
# Moles of an ideal gas
P = 1
V = 22.4
R = 0.0821
T = 300
moles = (P * V) / (R * T)
print("Moles of gas:", moles)

# Work done by a system
P = 2
delta_V = 5
work = -P * delta_V
print("Work done:", work)

# Energy required for heating
m = 10 * 1000  # Convert kg to grams
c = 4.18
delta_T = 30
Q = m * c * delta_T
print("Energy required (Q):", Q, "Joules")
```

#### Exercise 2.3:

```python
# Converting temperatures
temperatures = [300, 310, 320, 330]
for temp in temperatures:
    celsius = temp - 273.15
    print(f"{temp} K = {celsius:.2f} °C"

# List of pressures
pressures = [1.0, 1.5, 2.0, 2.5]
n = 1
R = 0.0821
T = 298
volumes = []
for P in pressures:
    V = (n * R * T) / P
    volumes.append(V)
print("Volumes:", volumes)

# Double flow rates
flow_rates = [5, 10, 15]
for rate in flow_rates:
    doubled_rate = rate * 2
    print(f"Original: {rate} L/s, Doubled: {doubled_rate} L/s")
```

#### Exercise 2.4:

```python
# Ideal gas law function
def ideal_gas_law(P, V, R, T):
    return (P * V) / (R * T)

moles = ideal_gas_law(2, 10, 0.0821, 298)
print("Calculated moles:", moles)

# Density function
def calculate_density(mass, volume):
    return mass / volume

density = calculate_density(25, 10)
print("Density:", density)

# Reynolds number function
def reynolds_number(velocity, length, density, viscosity):
    return (velocity * length * density) / viscosity

Re = reynolds_number(2, 0.5, 1000, 1.2)
print("Reynolds Number:", Re)
```

#### Exercise 2.5:

```python
# Pressures to kPA
import numpy as np

pressures = np.array([1, 2, 3, 4])
pressures_kpa = pressures * 101.325
print("Pressures in kPa:", pressures_kpa)
return pressures_kpa

# Temperatures in Kelvin
temps_celsius = np.array([25, 50, 75, 100])
temps_kelvin = temps_celsius + 273.15
print("Temperatures in Kelvin:", temps_kelvin)
return temps_kelvin

# Flow rates
flow_rates = np.array([10, 20, 30, 40])
mean_flow = np.mean(flow_rates)
std_flow = np.std(flow_rates)
print("Mean flow rate:", mean_flow)
print("Standard deviation of flow rates:", std_flow)
return std_flow
```


```

#### Exercise 2.6:

```python
import pandas as pd

# Load Excel file
data = pd.read_excel("data.xlsx", sheet_name="Sheet1")
print(data.head())

# Calculate the mean of FlowRate
mean_flowrate = data["FlowRate"].mean()
print("Mean FlowRate:", mean_flowrate)

# Filter rows
temp_filtered = data[data["Temperature"] > 300]
print(temp_filtered)

# Convert Pressure to kPa
data["Pressure_kPa"] = data["Pressure"] * 101.325
print(data.head())
```

#### Exercise 2.7:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot
x = [0, 1, 2, 3]
y = [0, 1, 4, 9]
plt.plot(x, y, marker="o")
plt.title("Line Plot")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

# Scatterplot
sns.scatterplot(data=data, x="Temperature", y="Pressure")
plt.title("Temperature vs Pressure")
plt.show()

# Histogram
sns.histplot(data=data, x="FlowRate", bins=10, kde=True)
plt.title("FlowRate Distribution")
plt.show()
```
# Section 6: Deploying a Graphical User Inteface (GUI)

Learn to build and deploy interactive web applications using Streamlit and Hugging Face Spaces, making it easy to share and deploy models, dashboards, and data tools. There are some xifferences between Streamlit and Hugging Face as shown in the table below:

#### 6.1. Streamlit versus Hugging Face

Before choosing how to deploy an AI or data science app, it helps to understand the difference between popular options like **Streamlit** and **Hugging Face Spaces**. Both tools make it easy to turn models into interactive demos, but they serve slightly different purposes. Streamlit focuses on flexible app building and customization in Python, while Hugging Face Spaces emphasizes sharing, hosting, and discoverability within the machine learning community.


**Streamlit**
- Designed for building interactive data and machine learning web apps directly in Python.  
- Great for local development and rapid prototyping — you can run apps with `streamlit run app.py`.  
- Flexible: can connect to any model (local, API, or cloud-hosted).  
- Allows custom UI design and logic using widgets (`st.slider`, `st.text_input`, etc.).  
- Deployment options include Streamlit Community Cloud, Docker, or internal servers.  
- Ideal for data scientists or engineers who want control over UI and logic in one Python script.

**Hugging Face Spaces**
- Platform for hosting and sharing ML demos publicly (or privately with paid plans).  
- Integrates directly with Hugging Face models and datasets — easy to showcase a model from the Hub.  
- Supports both **Gradio** and **Streamlit** apps as runtimes.  
- Automatically handles hosting, version control, and discoverability through the Hugging Face Hub.  
- Best suited for sharing finished demos with the community, teaching, or showcasing models for research and outreach.

#### 6.2. Installing and Importing Streamlit

Streamlit is a Python framework that allows users to create interactive web-based applications with minimal effort. Unlike traditional web development, Streamlit simplifies the process by allowing users to focus on writing Python scripts without worrying about front-end or back-end development. Streamlit is particularly useful for building interactive dashboards, machine learning apps, and engineering tools.

To get started with Streamlit, you need to install it and import the library.
```
# Uncomment and run this line if you haven't installed Streamlit
# !pip install streamlit
```
Now, import Streamlit into your Python script:
```
import streamlit as st
```

#### 6.3. Creating a Basic Streamlit App

Every Streamlit application starts with defining the interface elements. These include titles, text, buttons, and more. Below is a simple example of a basic Streamlit app:


```
import streamlit as st

st.title("Welcome to My First Streamlit App")
st.write("This is a simple interactive web application built with Streamlit.")
```

Run the script using:
```
streamlit run my_script.py
```

This will launch your Streamlit app in a web browser.

#### 6.4. Adding User Input

To make applications interactive, we need to accept user input. Streamlit provides various input elements such as text boxes, sliders, and dropdown menus. The example below demonstrates how to accept text input from a user and dynamically update the display based on their input.

```
import streamlit as st

st.title("Interactive Streamlit App")
name = st.text_input("Enter your name:")
st.write(f"Hello, {name}!")
```

Here, when a user types their name into the input field, it is displayed back dynamically using `st.write()`.

#### 6.5. Adding Buttons and Conditional Actions

Buttons allow users to trigger actions within a Streamlit app. You can use them to process inputs, execute functions, or display messages when clicked. Below is an example of a button that, when clicked, displays a success message.

```
import streamlit as st

st.title("Button Interaction Example")

if st.button("Click me!"):
    st.success("You clicked the button!")
```

#### 6.6. Displaying Data with Tables and Charts

Streamlit allows seamless integration with pandas to display data in a tabular format. This is particularly useful for chemical engineering applications where data analysis is a key component. Below is an example of how to display a DataFrame using Streamlit.

```{code-cell} ipython3
import streamlit as st
import pandas as pd

st.title("Displaying a DataFrame")

# Sample Data
data = {"Name": ["Alice", "Bob", "Charlie"], "Score": [85, 90, 78]}
df = pd.DataFrame(data)

st.write(df)
```

#### 6.7. **Deploying to Hugging Face Spaces**

Instead of running Streamlit locally, you can deploy your app to **Hugging Face Spaces**, a free hosting platform that allows you to share and showcase interactive applications. This is useful for engineers and researchers who want to share their work without setting up cloud servers.

**Step 1: Push Your Code to GitHub**

Ensure your Streamlit script (`app.py`) is stored in a GitHub repository.

**Step 2: Create a Hugging Face Space**

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Choose **Streamlit** as the application type.
4. Link your GitHub repository or manually upload your `app.py` file.

**Step 3: Add a `requirements.txt` File**

Your app needs a **requirements.txt** file to install dependencies. Example:

```
streamlit
pandas
numpy
```

**Step 4: Deploy and Share**

Once Hugging Face builds your Space, you’ll get a **public URL** to share your app with others!

#### 6.8. Deploying with Streamlit Community Cloud (Alternative)

Another option for deployment is **Streamlit Community Cloud**, which allows you to host apps directly from GitHub.

**Steps to Deploy:**

1. Push your script to **GitHub**.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Deploy and share your live app with a simple link!

#### Task:

In chemical engineering, some caluclations are multi-step and complicated. Creating a a GUI interface can be helpful to prevent mistakes when doing repetitive and common calculations.

Create the below calculators and deploy to github. Then, create a GUI using streamlit. Your github should contain the following files once you are complete:

Calc_Wastewater - an app to calcualte the dissolved oxygen in wastewater

Calc_MCT - an app to do McCabe-Thiele analysis

Calc_LLE - an app to calculate ratios and plot tie-lines in the LLE

Calc_CSTR - compares CSTR and PFR reactors for a first-order irreversible reaction and calculates reactor volumes

requirements.txt - requirements file

```{code-cell} ipython3
# Your code goes here
```

#### Answer Key:

```{code-cell} ipython3
# Wastewater Calculation

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the dissolved oxygen level iteratively
def calculate_do(do, t, otr, consumption_rate):
    # Iteratively calculate DO at each time step
    dt = t[1] - t[0]  # Time step
    for i in range(1, len(t)):
        # Oxygen transfer rate (OTR) - Oxygen consumption rate
        ddo = otr * (1 - do[i-1]/9.0) - consumption_rate
        do[i] = do[i-1] + ddo * dt  # Update DO based on OTR and consumption
        # DO can't fall below zero
        if do[i] < 0:
            do[i] = 0
    return do

# Streamlit app interface
st.title("Wastewater Treatment: Dissolved Oxygen Profile in Aeration Tank")

# Display a fancy wastewater treatment image
st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Wastewater_Treatment_2.jpg", caption="Aeration Tank in Wastewater Treatment", use_column_width=True)

# Input parameters for simulation
st.sidebar.header("Oxygen Transfer and Consumption Parameters")
ote = st.sidebar.slider("Oxygen Transfer Efficiency (OTE, %)", 0, 100, 20)  # Oxygen transfer efficiency
otr = ote / 100  # Convert to a fraction for OTR calculation
consumption_rate = st.sidebar.slider("Oxygen Consumption Rate (mg/L/min)", 0.0, 1.0, 0.1)
initial_do = st.sidebar.slider("Initial Dissolved Oxygen (DO, mg/L)", 0.0, 9.0, 2.0)  # Initial DO level

# Time array for simulation (minutes)
t = np.linspace(0, 120, 120)

# Initialize DO array with initial value
do = np.zeros_like(t)
do[0] = initial_do

# Calculate DO profile
do_profile = calculate_do(do, t, otr, consumption_rate)

# Plot the results
st.header("Dissolved Oxygen Profile in the Aeration Tank")
fig, ax = plt.subplots()
ax.plot(t, do_profile, label="DO Level", color='blue')
ax.axhline(y=2.0, color='red', linestyle='--', label="Minimum DO Threshold (2 mg/L)")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Dissolved Oxygen (mg/L)")
ax.set_title("DO Concentration Over Time in Aeration Tank")
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Display DO levels over time
st.write(f"Final Dissolved Oxygen Level: {do_profile[-1]:.2f} mg/L")

# Explanation of the simulation
st.write("""
### Oxygen Transfer in Wastewater Treatment
This simulation models the dissolved oxygen (DO) concentration over time in an aeration tank during wastewater treatment. Oxygen is transferred into the water to support microbial processes that consume oxygen to break down organic material.

- **Oxygen Transfer Efficiency (OTE)**: The percentage of oxygen that is effectively transferred into the water.
- **Oxygen Consumption Rate**: The rate at which oxygen is consumed by the microbes.
- **Initial Dissolved Oxygen (DO)**: The starting concentration of DO in the tank.

You can adjust these parameters to see how they affect the DO profile in the aeration tank.
""")
```

```{code-cell} ipython3
# MCT Calculation

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate equilibrium curve (y = alpha * x / (1 + (alpha - 1) * x))
def equilibrium_curve(x, alpha):
    return alpha * x / (1 + (alpha - 1) * x)

# Function to calculate the rectifying section operating line
def rectifying_line(x, R, xd):
    return (R / (R + 1)) * x + xd / (R + 1)

# Function to calculate the stripping section operating line
def stripping_line(x, xb, xf, q):
    return (q / (q - 1)) * (x - xb) + xb

# Streamlit app interface
st.title("Distillation Column Calculator using McCabe-Thiele Method")

# Input section for column design
st.header("Input Data for Distillation Column")

alpha = st.number_input("Relative Volatility (α)", value=2.0)
xf = st.number_input("Feed Mole Fraction (xF)", value=0.4, min_value=0.0, max_value=1.0)
xd = st.number_input("Distillate Mole Fraction (xD)", value=0.95, min_value=0.0, max_value=1.0)
xb = st.number_input("Bottom Mole Fraction (xB)", value=0.05, min_value=0.0, max_value=1.0)
R = st.number_input("Reflux Ratio (R)", value=1.5)
q = st.number_input("Feed Condition (q)", value=1.0, help="q = 1 for saturated liquid, q = 0 for saturated vapor")

# Generate x values for plotting (mole fractions)
x = np.linspace(0, 1, 500)
y_eq = equilibrium_curve(x, alpha)

# Calculate points for operating lines
y_rectifying = rectifying_line(x, R, xd)
y_stripping = stripping_line(x, xb, xf, q)

# McCabe-Thiele plot
st.header("McCabe-Thiele Diagram")

# Plotting equilibrium curve and operating lines
fig, ax = plt.subplots()

# Plot equilibrium curve
ax.plot(x, y_eq, label="Equilibrium Curve", color='blue')

# Plot rectifying section operating line
ax.plot(x, y_rectifying, label="Rectifying Line (R)", linestyle='--', color='green')

# Plot stripping section operating line
ax.plot(x, y_stripping, label="Stripping Line (q)", linestyle='--', color='red')

# Plot x = y line (45-degree line)
ax.plot(x, x, label="y = x", linestyle=':', color='black')

# Add labels and title
ax.set_xlabel("Mole Fraction of Light Component (x)")
ax.set_ylabel("Mole Fraction of Light Component (y)")
ax.set_title("McCabe-Thiele Diagram")
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Conclusion and analysis
st.write("""
This McCabe-Thiele diagram shows the equilibrium curve and the operating lines for the rectifying and stripping sections of the distillation column. The reflux ratio, feed condition, and relative volatility all influence the number of theoretical stages required for the separation process.
""")
```

```{code-cell} ipython3
# LLE Calculation

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate solute distribution between two phases (based on distribution ratio)
def calculate_distribution(c_s, D):
    c_o = D * c_s  # Solute concentration in organic phase
    return c_o

# Streamlit app interface
st.title("Liquid-Liquid Extraction: Tie-Line Diagram")

# Input section for solute concentration in aqueous phase and distribution ratio
st.header("Input Data for Liquid-Liquid Extraction")
c_s = st.number_input("Initial solute concentration in aqueous phase (mol/L)", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
D = st.number_input("Distribution Ratio (D)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)

# Generate solute concentrations in aqueous phase (x-axis)
c_s_range = np.linspace(0.01, 1.0, 100)  # Solute concentrations in aqueous phase
c_o_range = calculate_distribution(c_s_range, D)  # Corresponding concentrations in organic phase

# Plotting the tie-line diagram (c_s vs. c_o)
st.header("Tie-Line Diagram: Solute Distribution between Phases")
fig, ax = plt.subplots()

# Plot solute distribution in both phases
ax.plot(c_s_range, c_o_range, label="Tie Line", color='blue')

# Add labels and title
ax.set_xlabel("Solute Concentration in Aqueous Phase (mol/L)")
ax.set_ylabel("Solute Concentration in Organic Phase (mol/L)")
ax.set_title("Tie-Line Diagram: Liquid-Liquid Extraction")
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Output the calculated solute concentrations for the given initial concentration
c_o = calculate_distribution(c_s, D)
st.write(f"For an initial solute concentration of {c_s:.2f} mol/L in the aqueous phase, the solute concentration in the organic phase is {c_o:.2f} mol/L.")

# Conclusion and analysis
st.write("""
This tie-line diagram illustrates the equilibrium distribution of a solute between two immiscible phases (aqueous and organic).
The slope of the line depends on the distribution ratio (D), which is the ratio of solute concentration in the organic phase to that in the aqueous phase.
A higher D indicates that the solute prefers the organic phase, while a lower D indicates preference for the aqueous phase.
""")
```

```{code-cell} ipython3
# CSTR Calculation

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title and Description
st.title("CSTR and PFR Reactor Volume Calculator for First-Order Reactions")
st.write("""
This app calculates the reactor volume for both CSTR and PFR given the feed rate, rate constant, and target conversion.
You can also compare the performance of both reactors by observing the volume required to achieve the same conversion,
and plot the conversion profile for the PFR along the reactor length.
""")

# Input parameters
st.sidebar.header("Input Parameters")

# Feed rate (F_A0)
FA0 = st.sidebar.number_input("Feed rate (mol/s)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)

# Reaction rate constant (k)
k = st.sidebar.number_input("Reaction rate constant (1/s)", min_value=0.001, max_value=1.0, value=0.1, step=0.01)

# Conversion (X)
X = st.sidebar.slider("Conversion (X)", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

# CSTR Volume Calculation
if FA0 > 0 and k > 0:
    V_CSTR = FA0 * X / (k * (1 - X))
    st.write(f"**CSTR Volume required for {X * 100}% conversion:** {V_CSTR:.2f} m³")

# PFR Volume Calculation
if FA0 > 0 and k > 0:
    V_PFR = FA0 / k * np.log(1 / (1 - X))
    st.write(f"**PFR Volume required for {X * 100}% conversion:** {V_PFR:.2f} m³")

# Compare Reactor Volumes
if V_CSTR and V_PFR:
    st.write(f"### Performance Comparison:")
    st.write(f"The required volume for CSTR is {V_CSTR:.2f} m³, whereas for PFR it is {V_PFR:.2f} m³.")
    st.write("PFR requires less volume than CSTR for the same conversion.")

# PFR Conversion Profile
st.write("### PFR Conversion Profile:")
length = np.linspace(0, V_PFR, 100)
X_PFR_profile = 1 - np.exp(-k * length / FA0)

plt.figure()
plt.plot(length, X_PFR_profile, label="PFR Conversion")
plt.xlabel("Reactor Volume (m³)")
plt.ylabel("Conversion")
plt.title("PFR Conversion Profile")
plt.grid(True)
plt.legend()
st.pyplot(plt)

# CSTR Conversion vs Volume plot
st.write("### CSTR Conversion vs Reactor Volume:")
V_CSTR_range = np.linspace(0.1, V_CSTR, 100)
X_CSTR_range = 1 - (FA0 / (k * V_CSTR_range + FA0))

plt.figure()
plt.plot(V_CSTR_range, X_CSTR_range, label="CSTR Conversion")
plt.xlabel("Reactor Volume (m³)")
plt.ylabel("Conversion")
plt.title("CSTR Conversion vs Reactor Volume")
plt.grid(True)
plt.legend()
st.pyplot(plt)
```
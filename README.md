

# Student Alcohol Consumption and Academic Performance Analysis

## Project Overview
This project aims to analyze the relationship between alcohol consumption and academic performance among secondary school students. By exploring this dataset, we identify trends, correlations, and insights into how various factors like study time, parents' education, and living arrangements impact students' academic results and alcohol consumption habits.

## Dataset
The analysis is based on a dataset of secondary school students, which includes information such as:
- **Demographic Data**: Age, gender, family structure.
- **Parental Background**: Parents' education level, job status, and relationship status.
- **Academic Information**: Study time, absences, and final grades.
- **Alcohol Consumption**: Weekday and weekend alcohol consumption levels.

## Project Structure
- `STUDENT_PROJECT.ipynb`: Jupyter notebook containing the data analysis and visualization.
- `data/`: Directory containing the original dataset used for the analysis.

## Analysis Steps
1. **Data Cleaning and Preprocessing**:
   - Renamed columns for better readability.
   - Converted gender values for consistency.
   - Checked for null values and handled duplicates.
   
2. **Descriptive Statistics**:
   - Used summary statistics (`describe()`) to understand the distribution of numerical and categorical data.
   - Detailed dataset information using `info()` for data types and memory usage.
   
3. **Filtering and Insights**:
   - Filtered students based on parental living arrangements (`Pstatus`).
   - Identified students whose parents had no formal education.
   - Extracted details of students with the highest number of absences.
   - Identified female students who spend the maximum study time.

4. **Key Findings**:
   - Analysis of the correlation between alcohol consumption and academic performance.
   - Identification of the impact of parental education on student grades.
   - Insights into the difference in study time between students with high and low alcohol consumption.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
   - `pandas` for data manipulation and analysis.
   - `matplotlib` and `seaborn` for data visualization.
   - `numpy` for numerical operations.

## Usage
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd student-alcohol-analysis
   ```
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```
3. Run the Jupyter notebook `STUDENT_PROJECT.ipynb` to view the analysis:
   ```bash
   jupyter notebook STUDENT_PROJECT.ipynb
   ```

## Results
- Detailed analysis of how alcohol consumption varies among students based on their demographic and academic background.
- Visualizations that highlight key patterns and relationships in the data.
- Findings can be used for school policy recommendations and interventions for students at risk.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please submit a pull request.

## License
This project is licensed under the MIT License.


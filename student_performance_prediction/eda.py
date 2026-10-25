
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student-mat.csv", sep=";")

# Chart 1: Distribution of final grades
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=20, kde=True)

plt.title("Distribution of Final Student Grades")
plt.xlabel("Final Grade (0 to 20)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("grade_distribution.png")
plt.show()

# Chart 2: Study time and final grades
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="studytime", y="G3")

plt.title("Study Time vs Final Grade")
plt.xlabel("Weekly Study Time Category")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("studytime_vs_grade.png")
plt.show()

# Chart 3: Absences and final grades
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="absences", y="G3")

plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("absences_vs_grade.png")
plt.show()

print("All charts created successfully!")
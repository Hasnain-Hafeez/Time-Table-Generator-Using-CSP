from constraint import Problem
import random
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QGridLayout, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
from PyQt5.QtWidgets import QFileDialog, QPushButton, QMessageBox, QCheckBox
import sys
import pandas as pd

ADDED = []

def generate_variable_names(variables, suffixes):
    new_variables = []
    for var in variables:
        for suffix in suffixes:
            new_var = var + suffix
            new_variables.append(new_var)
    return new_variables

class TT:
    def __init__(self, v, l, d, dl, state=False):
        self.problem = Problem()
        self.variables = v
        self.domain = d
        self.labs = l
        self.domainL = dl
        self.state = state
        self._setup_problem()
        
    def _setup_problem(self):
        # Add domain to theory and add basic constraints
        for v in self.variables:
            self.problem.addVariables(v, self.domain)
            self.individual_class_constraints(v)

        # Avoid room clashes among different classes
        v1, v2, v3, v4 = self.variables
        for a in v1:
            for b in v2:
                self.problem.addConstraint(self.different_room, (a,b))
            for c in v3:
                self.problem.addConstraint(self.different_room, (a,c))
            for d in v4:
                self.problem.addConstraint(self.different_room, (a,d))

        for b in v2:
            for c in v3:
                self.problem.addConstraint(self.different_room, (b,c))
            for d in v4:
                self.problem.addConstraint(self.different_room, (b,d))

        for c in v3:
            for d in v4:
                self.problem.addConstraint(self.different_room, (c,d))
                
        # This portion is for lab
        
        # Add lab domain to lab variables
        self.dedicatedLabDomains()

        for labs in self.labs:
            for lab in labs:
                if lab not in ADDED:
                    self.problem.addVariable(lab, self.domainL)
        l1, l2, l3, l4 = self.labs
        
        # Avoid clash with theory e.g. BCE1 theory and BCE1 lab
        for l in l1:
            for v in v1:
                self.problem.addConstraint(self.clash_avoidance, (l, v))
                if l[:-7] == v[:-1]:
                    self.problem.addConstraint(self.different_day, (l, v))
        for l in l2:
            for v in v2:
                self.problem.addConstraint(self.clash_avoidance, (l, v))
                if l[:-7] == v[:-1]:
                    self.problem.addConstraint(self.different_day, (l, v))
        for l in l3:
            for v in v3:
                self.problem.addConstraint(self.clash_avoidance, (l, v))
                if l[:-7] == v[:-1]:
                    self.problem.addConstraint(self.different_day, (l, v))
        for l in l4:
            for v in v4:
                self.problem.addConstraint(self.clash_avoidance, (l, v))
                if l[:-7] == v[:-1]:
                    self.problem.addConstraint(self.different_day, (l, v))

        # Avoid day slot clashes e.g. BCE1 lab1 and BCE1 lab2
        for lab in self.labs:
            for i in range(len(lab)):
                for j in range(i + 1, len(lab)):
                    prefix_i = lab[i][:-7]  # Extract prefix of the first element
                    prefix_j = lab[j][:-7]  # Extract prefix of the second element
                    if prefix_i != prefix_j:  # Check if prefixes are different
                        self.problem.addConstraint(self.clash_avoidance, (lab[i], lab[j]))

            # Add lab slots adjacently i.e. both lab slots of lab1 should be adjacent
            m = 0
            while m < len(lab):
                lectures = 2
                while lectures > 1:
                    for n in range(m + 1, m + lectures):
                        self.problem.addConstraint(self.adjacent_lab_slots, (lab[m],lab[n]))
                    m += 1
                    lectures -= 1
                m += 1

        # Different rooms for labs e.g. BCE1 lab and BCE3 lab
        for a in l1:
            for b in l2:
                self.problem.addConstraint(self.different_room, (a,b))
            for c in l3:
                self.problem.addConstraint(self.different_room, (a,c))
            for d in l4:
                self.problem.addConstraint(self.different_room, (a,d))

        for b in l2:
            for c in l3:
                self.problem.addConstraint(self.different_room, (b,c))
            for d in l4:
                self.problem.addConstraint(self.different_room, (b,d))

        for c in l3:
            for d in l4:
                self.problem.addConstraint(self.different_room, (c,d))

        # Stop lectures spread over less than 4 days
        self.distributeLectures(self.variables, self.labs)
        
    def distributeLectures(self, variables, labvariables):
        for var, lab in zip(variables, labvariables):
            lectures = var + lab
            self.problem.addConstraint(self.min_days_with_lecture, lectures)

    def min_days_with_lecture(self, *args):
        """
        args: list of (day, slot, room) for all variables in one class
        Ensures that at least 4 different days have >=1 lecture
        """
        days_covered = set(day for day, slot, room in args)
        return len(days_covered) >= 4

    def individual_class_constraints(self, variables):     
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self.problem.addConstraint(self.clash_avoidance, (variables[i], variables[j]))
                if variables[i][:-1] == variables[j][:-1]:
                    self.problem.addConstraint(self.different_day, (variables[i], variables[j]))
    
    def dedicatedLabDomains(self):
        for i in [1,3,5,7]:
            INDICES = df.index[df[f'DedicatedLab{i}'].notna()].tolist()
            for index in INDICES:
                #print(df['DedicatedLab3'][index], df['Courses3'][index])
                domains = [(day, slot, df[f'DedicatedLab{i}'][index]) for day in days for slot in slots if slot not in ['S5', 'S6']]
                if self.state:
                    for _ in range(10):
                        random.shuffle(domains)
                self.problem.addVariable(df[f'Labs{i}'][index] + ' - LABa', domains)
                self.problem.addVariable(df[f'Labs{i}'][index] + ' - LABb', domains)
                ADDED.append(df[f'Labs{i}'][index] + ' - LABa') 
                ADDED.append(df[f'Labs{i}'][index] + ' - LABb') 

    def adjacent_lab_slots(self, p1, p2):
        return p1[0] == p2[0] and abs(int(p1[1][-1]) - int(p2[1][-1])) == 1 and p1[2] == p2[2] and self.pair_slots(p1, p2)
    
    def pair_slots(self, p1, p2):
        pairs = [('S1', 'S2'), ('S2', 'S1'), ('S3', 'S4'), ('S4', 'S3'), ('S5', 'S6'), ('S6', 'S5')]
        if (p1[1], p2[1]) in pairs:
            return True
        return False
  
    def clash_avoidance(self, p1, p2):
        return p1[0] != p2[0] or p1[1] != p2[1]
        
    def different_day(self, instance1, instance2):
        return instance1[0] != instance2[0]
        
    def different_room(self, instance1, instance2):
        return instance1 != instance2

    def r_solution(self):
        return self.problem.getSolution()
    
#  Various factors for domain
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
slots = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
rooms = ['Z-{}'.format(num) for num in range(101, 103)]
roomsL = ['COMPUTER LAB {}'.format(num) for num in range(1, 3)]

# take course names (variables) from excel sheet
df = pd.read_excel('courses.xlsx')
v1 = df['Courses1'].dropna().tolist()
v2 = df['Courses3'].dropna().tolist()
v3 = df['Courses5'].dropna().tolist()
v4 = df['Courses7'].dropna().tolist()
l1 = df['Labs1'].dropna().add(' - LAB').tolist()
l2 = df['Labs3'].dropna().add(' - LAB').tolist()
l3 = df['Labs5'].dropna().add(' - LAB').tolist()
l4 = df['Labs7'].dropna().add(' - LAB').tolist()

# Represent no of classes with suffixes
suffixes = ['a', 'b']

# Generate new variables with no of classes
v1 = generate_variable_names(v1, suffixes)
v2 = generate_variable_names(v2, suffixes)
v3 = generate_variable_names(v3, suffixes)
v4 = generate_variable_names(v4, suffixes)
l1 = generate_variable_names(l1, suffixes)
l2 = generate_variable_names(l2, suffixes)
l3 = generate_variable_names(l3, suffixes)
l4 = generate_variable_names(l4, suffixes)

# Group different classes together
variables = [v1, v2, v3, v4]
lab = [l1, l2, l3, l4]

# Graphics
class ScheduleApp(QWidget):
    def __init__(self):
        super().__init__()

        self.schedule = {}
        self.setWindowTitle("Class Schedule App")
        self.setGeometry(100, 100, 1920, 1000)  # Set the initial size of the window
        self.setStyleSheet("font-size: 16pt; font-weight: bold;")  # Set the background color
        self.setStyleSheet("""
            QTableWidget {
                background-color: #F9F6EE; /* table background */
                color: #252525; /* text color */
                gridline-color: gray;  /* optional, makes grid lines visible */
                font-size: 10pt; font-weight: bold;
            }
            QHeaderView::section {
                background-color: #222;  /* header background */
                color: #F9F6EE;  /* header text color */
                font-size: 10pt; font-weight: bold;
            }
        """)
        self.enable_randomization = False
        self.resize(1024, 768)           # default starting size
        self.setMinimumSize(800, 600)    # cannot shrink below 800x600
        #self.setMaximumSize(1920, 1080)  # cannot expand beyond 1920x1080

        # Create widgets
        self.class_label = QLabel("Select Class To See Schedule:")
        self.class_dropdown = QComboBox()
        self.class_dropdown.addItems(["BCE1", "BCE3", "BCE5", "BCE7"])
        self.class_dropdown.setEnabled(False)  # Disable the dropdown initially
        #self.class_dropdown.setCurrentIndex(-1)  # No selection initially
        self.class_dropdown.currentIndexChanged.connect(self.show_schedule)
        self.class_dropdown.setStyleSheet("background-color: white;")
        self.class_dropdown.setFixedWidth(80)
        self.class_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # Solution selector (disabled until "Generate" is clicked)
        self.solution_label = QLabel("Select Solution:")
        self.solution_dropdown = QComboBox()
        self.solution_dropdown.setEnabled(False)
        self.solution_dropdown.currentIndexChanged.connect(self.show_selected_solution)
        self.solution_dropdown.setStyleSheet("background-color: white;")
        self.solution_dropdown.setFixedWidth(80)
        self.solution_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # Disable initially
        self.solution_dropdown.setEnabled(False)
        self.solution_label.setEnabled(False)

        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_schedules)  # Change to call the new method
        self.generate_button.clicked.connect(self.show_schedule)  # Show the schedule after generating
        self.generate_button.clicked.connect(lambda: self.class_dropdown.setEnabled(True))
        self.generate_button.setStyleSheet("background-color: white;")
        self.generate_button.setFixedWidth(100)
        
        self.num_solutions_label = QLabel("Number of Solutions:")
        self.num_solutions_spin = QSpinBox()
        self.num_solutions_spin.setRange(1, 20)      # choose your max
        self.num_solutions_spin.setValue(1)          # default is 5
        # set color white for background of the spinbox
        self.num_solutions_spin.setStyleSheet("background-color: white;")
        # make spinbox width smaller
        self.num_solutions_spin.setFixedWidth(40)
        self.num_solutions_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.num_solutions_spin.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # Disable initially
        self.num_solutions_spin.setEnabled(False)
        self.num_solutions_label.setEnabled(False)

        
        self.schedule_table = QTableWidget()
        self.schedule_table.setColumnCount(len(slots) + 1)  # Add one column for "Days"
        self.schedule_table.setHorizontalHeaderLabels(["Days"] + slots)  # Add "Days" column header
        self.schedule_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.schedule_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.schedule_table.verticalHeader().setVisible(False)
        
        #Set row height for the header row (column names)
        header_row_height = 73  # Adjust the height as needed
        self.schedule_table.horizontalHeader().setMinimumHeight(header_row_height)

        self.save_button = QPushButton("Save Solutions")
        self.save_button.clicked.connect(self.save_solutions_to_excel)
        self.save_button.setEnabled(False)  # Enabled only after generating schedules
        
        self.randomization_checkbox = QCheckBox("Enable Randomization")
        self.randomization_checkbox.setChecked(False)  # default disabled

        # Connect signal
        self.randomization_checkbox.stateChanged.connect(self.toggle_randomization_controls)

        # Create layout
        layout = QGridLayout()
        layout.addWidget(self.class_label, 0, 0)
        layout.addWidget(self.class_dropdown, 0, 1)
        layout.addWidget(self.solution_label, 0, 2)
        layout.addWidget(self.solution_dropdown, 0, 3)
        layout.addWidget(self.randomization_checkbox, 0, 4)
        layout.addWidget(self.num_solutions_label, 0, 5)
        layout.addWidget(self.num_solutions_spin, 0, 6)
        layout.addWidget(self.save_button, 0, 7)

        layout.addWidget(self.generate_button, 0, 8)
        layout.addWidget(self.schedule_table, 1, 0, 1, 9)
        
        self.setLayout(layout)

        # Center the window on the screen
        self.center()

        # Additional setup for the schedule table
        self.schedule_table.verticalHeader().setVisible(False)  # Hide the vertical header (row numbers)
        
        # Set column width for "Days" column
        column_width_days = 269 # Change this value as needed
        self.schedule_table.setColumnWidth(0, column_width_days)

        # Set column width for other columns
        for j in range(1, len(slots) + 1):
            column_width = 269 # Change this value as needed
            self.schedule_table.setColumnWidth(j, column_width)

        # Initialize the domains
        self.domain = [(day, slot, room) for day in days for slot in slots for room in rooms]
        list.reverse(self.domain)
        self.domainL = [(day, slot, room) for day in days for slot in slots for room in roomsL]
        list.reverse(self.domainL)
   
    def generate_random_domains(self):
        domain, domainL = self.domain.copy(), self.domainL.copy()
        if self.enable_randomization:
            for _ in range(10):
                random.shuffle(domain)
                random.shuffle(domainL)
        return domain, domainL
            
    def center(self):
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().screenGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

    def generate_schedules(self):
        self.generate_button.setEnabled(False)
        self.schedule_table.setRowCount(0)
        self.schedule_table.setColumnCount(len(slots) + 1)
        self.schedule.clear()

        # We'll store multiple solutions
        self.all_solutions = []

        for class_name, class_vars_labs in zip(["BCE1", "BCE3", "BCE5", "BCE7"], [v1 + l1, v2 + l2, v3 + l3, v4 + l4]):
            schedule_list = []
            solutions = []
            N = 1
            if not self.enable_randomization:
                # Normal ONE solution mode
                solver = TT(variables, lab, self.domain, self.domainL, self.enable_randomization)
                solutions.append(solver.r_solution())
            else:
                # Get N solutions for this class
                N = self.num_solutions_spin.value()
                domain, domainL = self.generate_random_domains()
                for _ in range(N):
                    solver = TT(variables, lab, domain, domainL, self.enable_randomization)
                    solutions.append(solver.r_solution())
                self.solution_dropdown.setEnabled(True)
                self.generate_button.setEnabled(True)
            
            for sol in solutions:
                schedule = {day: {slot: "" for slot in slots} for day in days}
                for variable in class_vars_labs:
                    day, slot, room = sol[variable]
                    schedule[day][slot] = f"{variable[:-1]}\n({room})"
                schedule_list.append(schedule)

            self.all_solutions.append(schedule_list)
            # Store only the first solution for initial display
            self.schedule[class_name] = schedule_list[0]

        # Populate the solution dropdown
        self.solution_dropdown.clear()
        for i in range(N):
            self.solution_dropdown.addItem(f"Solution {i+1}")
        
        # Create the message box
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)          # set icon type: Information, Warning, Critical, Question
        msg.setWindowTitle("Success")                    # window title
        msg.setText("Schedule(s) created successfully!")          # main message text
        msg.setStandardButtons(QMessageBox.Ok)        # buttons (Ok by default)
        msg.exec_()                                   # show the message box
        self.save_button.setEnabled(True)

    def toggle_randomization_controls(self, state):
        enabled = (state == Qt.Checked)

        # GUI behavior
        self.solution_dropdown.setEnabled(enabled)
        self.num_solutions_spin.setEnabled(enabled)

        # Optional: visually gray-out label text
        self.solution_label.setEnabled(enabled)
        self.num_solutions_label.setEnabled(enabled)

        # BACKEND state variable
        self.enable_randomization = enabled
        self.generate_button.setEnabled(True)

    def show_selected_solution(self):
        selected_solution_index = self.solution_dropdown.currentIndex()
        selected_class = self.class_dropdown.currentText()
        class_index = ["BCE1", "BCE3", "BCE5", "BCE7"].index(selected_class)

        # Fetch the selected solution for that class
        schedule = self.all_solutions[class_index][selected_solution_index]
        self.display_schedule_table(schedule)
    
    def show_schedule(self):
        selected_class = self.class_dropdown.currentText()
        schedule = self.schedule[selected_class]
        self.display_schedule_table(schedule)

    def display_schedule_table(self, schedule):
        self.schedule_table.setRowCount(len(days))

        # Initialize the "Days" column in the schedule table
        for i, day in enumerate(days):
            item = QTableWidgetItem(day)
            item.setTextAlignment(Qt.AlignCenter)  # Set text alignment
            self.schedule_table.setItem(i, 0, item)
    
        for i, day in enumerate(days):
            for j, slot in enumerate(slots):
                item = QTableWidgetItem(schedule[day][slot])
                item.setTextAlignment(Qt.AlignCenter)  # Set text alignment
                self.schedule_table.setItem(i, j + 1, item)
                
        # Fix the typo in the following line
        for i in range(self.schedule_table.rowCount()):
            self.schedule_table.setRowHeight(i, 155)  # Adjust the row height as needed

    def get_col_widths_by_longest_line(self, dataframe):
        """
        For each column (and the index), compute the width based on the longest single
        line in any cell (split by newline). This avoids long second/third lines
        inflating the required column width.
        Returns list: [index_width, col1_width, col2_width, ...]
        """
        # index width: longest line in any index value or the index name
        idx_values = [str(v) for v in dataframe.index.values]
        idx_longest = 0
        for v in idx_values:
            for line in v.splitlines() or [""]:
                idx_longest = max(idx_longest, len(line))
        idx_longest = max(idx_longest, len(str(dataframe.index.name or "")))

        col_widths = []
        for col in dataframe.columns:
            maxw = len(str(col))
            for cell in dataframe[col].astype(str).values:
                # split on newlines and take the longest visible line
                for line in cell.splitlines() or [""]:
                    maxw = max(maxw, len(line))
            col_widths.append(maxw)

        return [idx_longest] + col_widths
    
    def save_solutions_to_excel(self):
        if not hasattr(self, "all_solutions") or not self.all_solutions:
            QMessageBox.warning(self, "Warning", "No solutions to save!")
            return
        try:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Solutions", "", "Excel Files (*.xlsx)")
            if not filename:
                return

            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                workbook = writer.book

                # Base centered format
                base_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': True})

                # Create formats with different fill colors for each class
                class_formats = {
                    "BCE1": workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': True, 'bg_color': '#FFC7CE', 'border': 1}),
                    "BCE3": workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': True, 'bg_color': '#C6EFCE', 'border': 1}),
                    "BCE5": workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': True, 'bg_color': '#FFEB9C', 'border': 1}),
                    "BCE7": workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': True, 'bg_color': '#9DC3E6', 'border': 1}),
                }

                for sol_idx in range(len(self.all_solutions[0])):  
                    sheet_name = f"Solution {sol_idx+1}"
                    all_classes_df = pd.DataFrame()

                    for class_name, class_schedules in zip(["BCE1", "BCE3", "BCE5", "BCE7"], self.all_solutions):
                        schedule = class_schedules[sol_idx]  # Pick this solution
                        df_schedule = pd.DataFrame(schedule)
                        df_schedule.index.name = "Slots"
                        df_schedule.insert(0, "Class", class_name)
                        all_classes_df = pd.concat([all_classes_df, df_schedule])

                    # Write to Excel
                    all_classes_df.to_excel(writer, sheet_name=sheet_name, index=True)
                    worksheet = writer.sheets[sheet_name]

                    # Apply column formatting
                    for idx, col in enumerate(all_classes_df.columns):
                        for row_idx, val in enumerate(all_classes_df[col], start=1):  # start=1 because row 0 is header
                            class_name = all_classes_df.iloc[row_idx-1, 0]  # First column has class name
                            cell_format = class_formats.get(class_name, base_format)
                            worksheet.write(row_idx, idx+1, val, cell_format)  # idx+1 because index column

                    # Index column formatting
                    worksheet.set_column(0, 0, 10, base_format)

                    # Set default row height
                    worksheet.set_default_row(40)

                    # compute widths based on longest line per cell
                    col_widths = self.get_col_widths_by_longest_line(all_classes_df)

                    # apply widths and center_format (index is column 0)
                    for col_idx, width in enumerate(col_widths):
                        # small padding to avoid truncation
                        worksheet.set_column(col_idx, col_idx, width + 2, base_format)

            QMessageBox.information(self, "Saved", f"All solutions saved to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving: {str(e)}")


# ------------------- Run Application -------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScheduleApp()
    window.show()
    sys.exit(app.exec_())

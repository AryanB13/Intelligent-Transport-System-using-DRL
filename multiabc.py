import multiprocessing
import subprocess
# Define the list of Python files to run
file_list = ["conference_trainingN7.py", "conference_trainingN8.py", "conference_trainingN9.py", "conference_trainingN10.py", "conference_trainingN12.py"]

# Define a function to run a file
def run_file(file):
    subprocess.call(["python3", file])

# Create a process for each file
processes = []
for file in file_list:
    process = multiprocessing.Process(target=run_file, args=(file,))
    processes.append(process)
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()
    
    
    
    #exception
    
    #multi threading

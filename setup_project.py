import os

def create_project_structure(base_path="nepal_inflation_forecast"):
    structure = {
        "app": ["__init__.py", "pages/dashboard.py", "pages/data_analysis.py", 
                "pages/model_training.py", "pages/forecasting.py", 
                "pages/model_performance.py"],
        "data": ["raw/", "processed/"],
        "src": ["data/", "features/", "models/", "utils/"],
        "notebooks": [],
        "reports": [],
        "": ["requirements.txt", "app.py", "README.md"]  # Files at the root level
    }

    for folder, files in structure.items():
        # Create the folder
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        # Create files within the folder
        for file in files:
            file_path = os.path.join(base_path, folder, file)
            if file.endswith('/'):  # Create subdirectories
                os.makedirs(file_path, exist_ok=True)
            else:  # Create files
                with open(file_path, 'w') as f:
                    pass  # Create an empty file

    print(f"Project structure created at: {os.path.abspath(base_path)}")

# Call the function to create the structure
create_project_structure()

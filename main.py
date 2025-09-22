# -----------------------------------------------------------
# Patient Management API using FastAPI
# -----------------------------------------------------------
# Features:
# 1. "/"       → Home endpoint
# 2. "/about"  → About this API
# 3. "/view"   → View all patient data
# 4. "/patient/{patient_id}" → View single patient by ID
# 5. "/sort"   → Sort patients by height, weight, or BMI
#
# Note:
# - Data is loaded from data.json file
# - Works for both dict-based and list-based JSON structures
# -----------------------------------------------------------

from fastapi import FastAPI, Path, HTTPException, Query
import json

# -----------------------------------------------------------
# Initialize the FastAPI app
# -----------------------------------------------------------
app = FastAPI(title="Patient Management API", version="1.0")


# -----------------------------------------------------------
# Utility: Load patient data from JSON file
# -----------------------------------------------------------
def load_data():
    """Load patient data from a JSON file."""
    with open("data.json", "r") as f:
        data = json.load(f)
    return data


# -----------------------------------------------------------
# ROUTE 1: Root (Home) endpoint
# -----------------------------------------------------------
@app.get("/")
def hello():
    """Return a welcome message for the API."""
    return {"message": "Patient Management API"}


# -----------------------------------------------------------
# ROUTE 2: About endpoint
# -----------------------------------------------------------
@app.get("/about")
def about():
    """Return information about the API."""
    return {"message": "Fully Functional Patient API"}


# -----------------------------------------------------------
# ROUTE 3: View all patients
# -----------------------------------------------------------
@app.get("/view")
def view():
    """Return all patient records from the dataset."""
    return load_data()


# -----------------------------------------------------------
# ROUTE 4: View patient by ID
# -----------------------------------------------------------
@app.get("/patient/{patient_id}")
def view_patient(
    patient_id: str = Path(
        ..., description="ID of the patient", example="P001"
    )
):
    """
    Retrieve a single patient by their ID.
    - Works with both dict-based JSON (id as key) and list-based JSON (id field inside objects).
    """
    data = load_data()
    
    # Case 1: JSON is a dict where keys are patient IDs
    if isinstance(data, dict) and patient_id in data:
        return data[patient_id]
    
    # Case 2: JSON is a list where each patient is a dict
    if isinstance(data, list):
        for patient in data:
            if str(patient.get("id")) == patient_id:
                return patient
    
    # Case 3: Not found
    raise HTTPException(status_code=404, detail="Patient not found")


# -----------------------------------------------------------
# ROUTE 5: Sort patients
# -----------------------------------------------------------
@app.get("/sort")
def sort_patients(
    sort_by: str = Query(..., description="Sort on the basis of height, weight or bmi"),
    order: str = Query("asc", description="Sort order: asc or desc")
):
    """
    Sort patients based on a selected field (height, weight, or bmi).
    - sort_by: field name to sort on
    - order: asc (default) or desc
    """
    valid_fields = ["height", "weight", "bmi"]
    
    # Validate sort field
    if sort_by not in valid_fields:
        raise HTTPException(
            status_code=400, detail=f"Invalid field. Select from {valid_fields}"
        )
    
    # Validate order
    if order not in ["asc", "desc"]:
        raise HTTPException(
            status_code=400, detail="Invalid order. Select between asc and desc"
        )
    
    # Load data
    data = load_data()
    
    # Normalize data into list
    if isinstance(data, dict):
        patients = list(data.values())
    elif isinstance(data, list):
        patients = data
    else:
        raise HTTPException(status_code=500, detail="Invalid data format")
    
    # Determine sort order (reverse=True means descending)
    reverse_order = True if order == "desc" else False
    
    # Sort patients by the chosen field
    sorted_data = sorted(
        patients,
        key=lambda x: x.get(sort_by, 0),  # use 0 if field missing
        reverse=reverse_order
    )
    
    return sorted_data
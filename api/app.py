from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return{
        "message":"Welcome to customer churn prediction API"
    }

@app.get("/square/{num}")
def square(num:int):
    return{
        "number":num,
        "square":num*num
    }

from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int

@app.post("/student")
def create_student(student: Student):
    return {
        "message": f"{student.name} added successfully"
    }
from pydantic import BaseModel,EmailStr, Field
from typing import Optional
class student(BaseModel):
    name:str = 'Gaurang'
    age: Optional[int]= None
    email:EmailStr
    cgpa: float = Field (gt=0, lt=10)
new_student = {'age':'20','email':'abc@gmail.com','cgpa':'5'}
student = student(**new_student)
print(student)
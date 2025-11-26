from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Triage System",
    description="AI-powered patient triage and department recommendation system",
    version="1.0.0"
)

# Input model
class PatientInfo(BaseModel):
    gender: str = Field(..., description="Patient's gender (e.g., male, female)")
    age: int = Field(..., ge=0, le=150, description="Patient's age")
    symptoms: List[str] = Field(..., min_items=1, description="List of symptoms")

# Output model
class DepartmentRecommendation(BaseModel):
    recommended_department: str = Field(..., description="Recommended hospital department")

# Initialize LLM
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_tokens=500
    )

# Create prompt template with advanced prompt engineering
def create_triage_prompt():
    template = """You are an expert medical triage AI assistant helping hospital staff route patients to the appropriate department.

Your task is to analyze patient information and recommend the most suitable hospital department.

PATIENT INFORMATION:
- Gender: {gender}
- Age: {age} years old
- Symptoms: {symptoms}

AVAILABLE DEPARTMENTS:
1. Emergency Medicine - Life-threatening conditions, severe trauma, acute cardiac events
2. Cardiology - Heart-related issues, chest pain, palpitations, irregular heartbeat
3. Neurology - Brain and nervous system issues, headaches, dizziness, seizures, stroke symptoms, difficulty walking
4. Orthopedics - Bone, joint, and muscle problems, fractures, sprains
5. Gastroenterology - Digestive system issues, abdominal pain, nausea, vomiting, diarrhea
6. Pulmonology - Respiratory issues, breathing difficulties, cough, asthma
7. Internal Medicine - General medical conditions, fever, fatigue, multiple symptoms
8. Pediatrics - Children and adolescents (under 18 years)
9. Geriatrics - Elderly patients (over 65 years) with complex conditions
10. Dermatology - Skin conditions, rashes, lesions
11. Ophthalmology - Eye problems, vision issues
12. ENT (Ear, Nose, Throat) - Ear, nose, throat issues, hearing problems
13. Urology - Urinary system issues, kidney problems
14. Gynecology - Women's reproductive health issues
15. Psychiatry - Mental health concerns, anxiety, depression

INSTRUCTIONS:
1. Analyze the symptoms carefully, considering the patient's age and gender
2. Identify the primary medical concern
3. Consider symptom combinations that might indicate specific conditions
4. Recommend the SINGLE most appropriate department
5. Return ONLY the department name, nothing else

IMPORTANT CONSIDERATIONS:
- Symptoms like "pusing" (dizziness), "mual" (nausea), "sulit berjalan" (difficulty walking) together suggest neurological issues
- Chest pain + shortness of breath → likely Cardiology or Emergency
- Multiple vague symptoms in elderly → consider Geriatrics or Internal Medicine
- Severe or life-threatening symptoms → Emergency Medicine
- Children under 18 → consider Pediatrics unless specialist needed

{format_instructions}

Respond with ONLY the JSON output, no additional text or markdown formatting."""

    return ChatPromptTemplate.from_template(template)

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Hospital Triage System API",
        "status": "operational",
        "endpoints": {
            "POST /recommend": "Get department recommendation for patient",
            "GET /health": "Check API health status"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Main recommendation endpoint
@app.post("/recommend", response_model=DepartmentRecommendation)
async def recommend_department(patient: PatientInfo):
    """
    Recommend appropriate hospital department based on patient information.
    
    Args:
        patient: PatientInfo object containing gender, age, and symptoms
        
    Returns:
        DepartmentRecommendation with recommended department, reasoning, and urgency level
    """
    try:
        # Initialize LLM and parser
        llm = get_llm()
        parser = PydanticOutputParser(pydantic_object=DepartmentRecommendation)
        
        # Create prompt
        prompt = create_triage_prompt()
        
        # Format symptoms as a readable string
        symptoms_str = ", ".join(patient.symptoms)
        
        # Create the chain
        chain = prompt | llm | parser
        
        # Get recommendation
        result = chain.invoke({
            "gender": patient.gender,
            "age": patient.age,
            "symptoms": symptoms_str,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing recommendation: {str(e)}"
        )

# Example endpoint to demonstrate usage
@app.get("/example")
async def get_example():
    return {
        "example_request": {
            "gender": "female",
            "age": 62,
            "symptoms": ["pusing", "mual", "sulit berjalan"]
        },
        "example_response": {
            "recommended_department": "Neurology",
            "reasoning": "Combination of dizziness, nausea, and difficulty walking suggests potential neurological issues that require specialist evaluation.",
            "urgency_level": "High"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
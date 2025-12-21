
import httpx
import asyncio

async def test_upload_persona_pdf():
    """
    페르소나 PDF 업로드 API를 테스트합니다.
    """
    url = "http://127.0.0.1:8000/api/v1/personas/upload"
    company_id = 1
    pdf_path = "/home/ec2-user/flex/server/docs/2024년-상반기-3급-신입사원-채용-직무소개서.pdf"

    try:
        with open(pdf_path, "rb") as f:
            files = {"pdf_file": (pdf_path.split('/')[-1], f, "application/pdf")}
            data = {"company_id": str(company_id)}
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                print(f"Sending request to {url}...")
                response = await client.post(url, data=data, files=files)

        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 201:
            print("✓ Persona PDF uploaded successfully!")
            response_data = response.json()
            print("\n--- Persona ---")
            print(f"Name: {response_data['persona']['persona_name']}")
            print(f"Archetype: {response_data['persona']['archetype']}")
            print(f"Description: {response_data['persona']['description']}")
            
            print("\n--- Questions ---")
            for i, q in enumerate(response_data['questions']):
                print(f"{i+1}. {q['question_text']} (Type: {q['question_type']})")

        else:
            print(f"✗ Error: {response.text}")

    except FileNotFoundError:
        print(f"✗ Error: PDF file not found at {pdf_path}")
    except httpx.RequestError as e:
        print(f"✗ Error connecting to the server: {e}")
    except Exception as e:
        print(f"✗ An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(test_upload_persona_pdf())

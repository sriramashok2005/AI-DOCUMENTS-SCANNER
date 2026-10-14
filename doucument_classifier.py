def classify_document(text: str) -> str:
    text = text.lower()

    if "invoice" in text or "total amount" in text:
        return "Invoice"

    elif "leave policy" in text or "employee policy" in text:
        return "Employee Policy"
    
    elif "troubleshooting" in text or "error solution" in text:
            return "Troubleshooting Guide"

    elif "product features" in text or "installation guide" in text:
         return "Product Documentation"
    else:
         return"Unknown Document"
import os
import csv
import json
import random
import logging
import pandas as pd
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sample_data(data_path=None):
    """
    Load sample data for training or testing
    
    Args:
        data_path: Path to the data file
        
    Returns:
        X: List of document texts
        y: List of document class labels
    """
    if data_path is None:
        data_path = os.path.join(config.SAMPLE_DIR, "sample_data.csv")
    
    if not os.path.exists(data_path):
        logger.info(f"Sample data not found at {data_path}, creating it")
        create_sample_data(data_path)
    
    logger.info(f"Loading sample data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        X = df['text'].tolist()
        y = df['label'].tolist()
        
        logger.info(f"Loaded {len(X)} samples")
        
        return X, y
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        raise

def create_sample_data(output_path=None, num_samples=100):
    """
    Create sample data for training or testing
    
    Args:
        output_path: Path to save the data
        num_samples: Number of samples to create
        
    Returns:
        Path to the created data file
    """
    if output_path is None:
        output_path = os.path.join(config.SAMPLE_DIR, "sample_data.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Creating {num_samples} sample data records at {output_path}")
    
    # Sample text for each document class
    class_templates = {
        "invoice": [
            "INVOICE #: {invoice_number}\nDate: {date}\nDue Date: {due_date}\nBill To: {customer}\nAmount Due: ${amount}",
            "Invoice\nInvoice Number: {invoice_number}\nInvoice Date: {date}\nCustomer: {customer}\nTotal: ${amount}",
            "BILLING INVOICE\nInvoice #: {invoice_number}\nIssue Date: {date}\nPayment Due: {due_date}\nCustomer ID: {customer}\nTotal Amount: ${amount}",
            "FACTURE #: {invoice_number}\nDate: {date}\nDate d'échéance: {due_date}\nClient: {customer}\nMontant: ${amount}\nTotal TVA: ${tax}",
            "Facture\nNuméro de Facture: {invoice_number}\nDate de Facture: {date}\nClient: {customer}\nTotal: ${amount}\nTVA: ${tax}",
            "فاتورة\nرقم الفاتورة: {invoice_number}\nتاريخ: {date}\nالعميل: {customer}\nالمبلغ: ${amount}\nضريبة القيمة المضافة: ${tax}",
            "فاتورة رقم: {invoice_number}\nتاريخ الإصدار: {date}\nتاريخ الاستحقاق: {due_date}\nالعميل: {customer}\nالمبلغ المستحق: ${amount}",
            "Facture N°: {invoice_number}\nDu: {date}\nAu nom de: {customer}\nMontant HT: ${subtotal}\nTVA: ${tax}\nTotal TTC: ${amount}",
            "N° Facture: {invoice_number}\nDate Facture: {date}\nClient: {customer}\nMontant Total Hors Taxes: ${subtotal}\nMontant TVA: ${tax}\nMontant TTC: ${amount}"
        ],
        "contract": [
            "CONTRACT AGREEMENT\nBetween {party1} and {party2}\nEffective Date: {date}\nTerm: {term}\nThis agreement is made on {date} between {party1} and {party2}...",
            "SERVICES CONTRACT\nThis Contract dated {date} is between {party1} ('Provider') and {party2} ('Client')\nTerm of Service: {term}\nScope of Work: {scope}",
            "CONSULTING AGREEMENT\nThis Consulting Agreement effective {date} is between {party1} and {party2}\nServices: {scope}\nCompensation: ${amount}"
        ],
        "id_card": [
            "IDENTIFICATION CARD\nName: {name}\nID Number: {id_number}\nDate of Birth: {dob}\nIssue Date: {date}\nExpiration: {expiration}",
            "EMPLOYEE ID\n{name}\nEmployee Number: {id_number}\nDepartment: {department}\nIssued: {date}",
            "SECURITY BADGE\nName: {name}\nAccess Level: {level}\nID: {id_number}\nValid Until: {expiration}"
        ],
        "resume": [
            "RESUME\n{name}\nEmail: {email}\nPhone: {phone}\nEducation: {education}\nExperience: {experience}",
            "CURRICULUM VITAE\n{name}\nContact: {email}, {phone}\nQualifications: {education}\nWork History: {experience}",
            "PROFESSIONAL RESUME\nName: {name}\nContact Information: {email} | {phone}\nSkills: {skills}\nProfessional Experience: {experience}"
        ],
        "certificate": [
            "CERTIFICATE OF COMPLETION\nThis certifies that {name} has successfully completed {course} on {date}. Certification ID: {id_number}",
            "ACHIEVEMENT AWARD\nAwarded to {name} for excellence in {course} on this day {date}. Certificate Number: {id_number}",
            "CERTIFICATION\nThis document certifies that {name} has met all requirements for {course} as of {date}. Credential ID: {id_number}"
        ],
        "receipt": [
            "RECEIPT\nTransaction #: {transaction_id}\nDate: {date}\nItems: {items}\nTotal: ${amount}\nThank you for your purchase!",
            "PURCHASE RECEIPT\nStore: {store}\nDate: {date}\nProduct(s): {items}\nAmount Paid: ${amount}\nPayment Method: {payment_method}",
            "SALES RECEIPT\nReceipt #: {transaction_id}\nSale Date: {date}\nDescription: {items}\nSubtotal: ${subtotal}\nTax: ${tax}\nTotal: ${amount}"
        ],
        "report": [
            "QUARTERLY REPORT\nPeriod: {quarter}\nPrepared by: {name}\nDate: {date}\nPerformance Summary: {summary}\nFinancial Results: {results}",
            "ANNUAL REPORT\nFiscal Year: {year}\nCompany: {company}\nRevenue: ${revenue}\nExpenses: ${expenses}\nNet Income: ${income}",
            "PROJECT REPORT\nProject Name: {project}\nSubmission Date: {date}\nTeam Members: {team}\nObjectives: {objectives}\nOutcomes: {outcomes}"
        ],
        "letter": [
            "Dear {recipient},\n\nI am writing to {purpose}. As we discussed {topic}, I wanted to follow up on {details}.\n\nSincerely,\n{sender}",
            "FORMAL LETTER\nFrom: {sender}\nTo: {recipient}\nDate: {date}\n\nSubject: {subject}\n\nDear {recipient},\n\nIn reference to {subject}, we would like to inform you that {details}.\n\nRegards,\n{sender}",
            "COVER LETTER\nFrom: {name}\nContact: {email}, {phone}\nDate: {date}\n\nDear Hiring Manager,\n\nI am writing to express my interest in {position} at {company}. My background in {experience} makes me an ideal candidate.\n\nSincerely,\n{name}"
        ],
        "form": [
            "APPLICATION FORM\nApplicant Name: {name}\nDate of Birth: {dob}\nAddress: {address}\nPhone: {phone}\nEmail: {email}\nSignature: _________\nDate: {date}",
            "REGISTRATION FORM\nRegistrant: {name}\nID Number: {id_number}\nEvent: {event}\nDate: {date}\nPayment Status: {status}",
            "CONSENT FORM\nName: {name}\nDate: {date}\nI, {name}, hereby consent to {consent_purpose}.\nContact Information: {phone}, {email}\nSignature: _________"
        ]
    }
    
    # Generate random values for templates
    def random_date():
        return f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(2020, 2023)}"
    
    def random_name():
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def random_company():
        prefixes = ["Global", "Advanced", "Premier", "Elite", "Innovative", "Trusted", "Strategic"]
        nouns = ["Systems", "Solutions", "Technologies", "Enterprises", "Industries", "Consultants", "Associates"]
        return f"{random.choice(prefixes)} {random.choice(nouns)}"
    
    # Generate sample data
    data = []
    for i in range(num_samples):
        # Choose random class
        doc_class = random.choice(config.DOCUMENT_CLASSES)
        
        # Skip classes not in our templates
        if doc_class not in class_templates:
            continue
        
        # Choose random template for this class
        template = random.choice(class_templates[doc_class])
        
        # Generate text from template with random values
        text = template.format(
            invoice_number=f"INV-{random.randint(1000, 9999)}",
            date=random_date(),
            due_date=random_date(),
            customer=random_name(),
            amount=f"{random.randint(100, 10000)}.{random.randint(0, 99):02d}",
            party1=random_name(),
            party2=random_company(),
            term=f"{random.randint(1, 12)} months",
            scope=f"Provide {random.choice(['consulting', 'development', 'design', 'marketing'])} services",
            name=random_name(),
            id_number=f"ID-{random.randint(10000, 99999)}",
            dob=random_date(),
            expiration=random_date(),
            department=random.choice(["IT", "HR", "Finance", "Marketing", "Sales", "Operations"]),
            level=random.choice(["Basic", "Standard", "Advanced", "Administrative", "Executive"]),
            email=f"{random_name().lower().replace(' ', '.')}@example.com",
            phone=f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            education=f"{random.choice(['Bachelor', 'Master', 'PhD'])} in {random.choice(['Computer Science', 'Business', 'Engineering', 'Arts'])}",
            experience=f"{random.randint(1, 10)} years of experience in {random.choice(['software development', 'project management', 'data analysis', 'design'])}",
            skills=", ".join(random.sample(["Python", "Java", "JavaScript", "Management", "Communication", "Leadership", "Design", "Marketing"], 3)),
            course=f"{random.choice(['Advanced', 'Fundamental', 'Practical'])} {random.choice(['Programming', 'Management', 'Leadership', 'Design'])}",
            transaction_id=f"TXN-{random.randint(1000, 9999)}",
            items=f"{random.randint(1, 5)} x {random.choice(['Product A', 'Service B', 'Item C', 'Subscription D'])}",
            store=f"{random_company()} Store",
            payment_method=random.choice(["Credit Card", "Cash", "Check", "PayPal", "Bank Transfer"]),
            subtotal=f"{random.randint(100, 1000)}.{random.randint(0, 99):02d}",
            tax=f"{random.randint(10, 100)}.{random.randint(0, 99):02d}",
            quarter=f"Q{random.randint(1, 4)} {random.randint(2020, 2023)}",
            summary=f"Performance {random.choice(['exceeded', 'met', 'approached'])} expectations",
            results=f"Revenue {random.choice(['increased', 'decreased', 'stabilized'])} by {random.randint(1, 20)}%",
            year=str(random.randint(2020, 2023)),
            company=random_company(),
            revenue=f"{random.randint(100000, 9999999)}",
            expenses=f"{random.randint(50000, 5000000)}",
            income=f"{random.randint(10000, 1000000)}",
            project=f"Project {random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Omega'])}",
            team=", ".join(random.sample([random_name(), random_name(), random_name(), random_name()], random.randint(2, 4))),
            objectives=f"To {random.choice(['develop', 'implement', 'design', 'optimize'])} {random.choice(['system', 'process', 'solution', 'framework'])}",
            outcomes=f"Successfully {random.choice(['completed', 'delivered', 'achieved', 'exceeded'])} project goals",
            recipient=random_name(),
            purpose=f"discuss {random.choice(['the project', 'our agreement', 'the meeting', 'your application'])}",
            topic=random.choice(["in our last meeting", "over the phone", "in our email exchange", "during the conference"]),
            details=random.choice(["the next steps", "the timeline", "the budget", "the requirements"]),
            sender=random_name(),
            subject=f"Re: {random.choice(['Your Inquiry', 'Project Update', 'Meeting Follow-up', 'Application Status'])}",
            position=f"{random.choice(['Senior', 'Junior', 'Lead'])} {random.choice(['Developer', 'Designer', 'Manager', 'Analyst'])}",
            address=f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Park'])} {random.choice(['St', 'Ave', 'Blvd', 'Rd'])}, {random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])}",
            event=f"{random.choice(['Annual Conference', 'Workshop', 'Seminar', 'Training', 'Meeting'])}",
            status=random.choice(["Paid", "Pending", "Cancelled", "Refunded"]),
            consent_purpose=random.choice(["participate in the study", "receive communications", "share my data", "be photographed during the event"])
        )
        
        data.append({
            "text": text,
            "label": doc_class
        })
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(data)
    
    logger.info(f"Created sample data with {len(data)} records")
    
    return output_path 

def augment_with_additional_data(X, y, additional_data_path=None):
    """
    Augment training data with additional examples
    
    Args:
        X: List of document texts
        y: List of document class labels
        additional_data_path: Path to additional data CSV
        
    Returns:
        X_augmented: Augmented list of document texts
        y_augmented: Augmented list of document class labels
    """
    if additional_data_path and os.path.exists(additional_data_path):
        logger.info(f"Augmenting with additional data from {additional_data_path}")
        
        try:
            df = pd.read_csv(additional_data_path)
            additional_X = df['text'].tolist()
            additional_y = df['label'].tolist()
            
            # Count original samples per class
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            logger.info(f"Original class distribution: {class_counts}")
            
            # Ensure we have at least 2 examples of each class
            min_required = 2
            for class_name in config.DOCUMENT_CLASSES:
                if class_name not in class_counts or class_counts[class_name] < min_required:
                    # We need to ensure there are at least min_required samples
                    needed = min_required - class_counts.get(class_name, 0)
                    logger.info(f"Class {class_name} needs {needed} more samples to have at least {min_required}")
                    
                    # If we don't have enough, duplicate an existing one
                    existing_idx = [i for i, label in enumerate(y) if label == class_name]
                    if existing_idx:
                        # Duplicate an existing sample
                        for _ in range(needed):
                            idx = existing_idx[0]
                            X.append(X[idx])
                            y.append(class_name)
                            logger.info(f"Added duplicate for class {class_name}")
                    else:
                        # If no existing sample, create a basic one
                        for _ in range(needed):
                            X.append(f"Example document of class {class_name}")
                            y.append(class_name)
                            logger.info(f"Added placeholder for class {class_name}")
            
            # Now add the additional data
            X_augmented = X + additional_X
            y_augmented = y + additional_y
            
            logger.info(f"Added {len(additional_X)} samples, new total: {len(X_augmented)}")
            
            return X_augmented, y_augmented
        except Exception as e:
            logger.error(f"Error loading additional data: {str(e)}")
            return X, y
    else:
        return X, y 
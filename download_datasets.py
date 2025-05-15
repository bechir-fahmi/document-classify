import os
import csv
import random
import requests
import zipfile
import io
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Create directories if they don't exist
os.makedirs('data/samples/downloaded', exist_ok=True)
os.makedirs('data/samples/synthetic', exist_ok=True)

# List of datasets to download
DATASETS = [
    {
        'name': 'sroie',
        'url': 'https://github.com/zzzDavid/ICDAR-2019-SROIE/raw/master/data/train.zip',
        'type': 'receipt',
        'lang': 'en'
    },
    {
        'name': 'invoices',
        'url': 'https://github.com/rossumai/public-invoices/raw/master/invoices.zip', 
        'type': 'invoice',
        'lang': 'mixed'
    }
]

# Template generator for synthetic data
TEMPLATES = {
    'purchase_order': [
        "BON DE COMMANDE\nNuméro: {po_num}\nDate: {date}\nFournisseur: {supplier}\nArticles: {items}\nTotal: {total}€\nDélai de livraison: {delivery} jours",
        "PURCHASE ORDER\nPO Number: {po_num}\nDate: {date}\nVendor: {supplier}\nItems: {items}\nTotal: ${total}\nDelivery terms: {delivery} days"
    ],
    'quote': [
        "DEVIS\nN° Devis: {quote_num}\nDate: {date}\nClient: {client}\nDésignation: {description}\nMontant HT: {amount}€\nTVA: {tax}€\nTotal TTC: {total}€\nValidité: {validity} jours",
        "QUOTATION\nQuote #: {quote_num}\nDate: {date}\nCustomer: {client}\nDescription: {description}\nSubtotal: ${amount}\nTax: ${tax}\nTotal: ${total}\nValid until: {validity} days"
    ],
    'delivery_note': [
        "BON DE LIVRAISON\nBL N°: {dn_num}\nDate: {date}\nClient: {client}\nCommande N°: {order_ref}\nArticles livrés: {items}\nTransporteur: {carrier}",
        "DELIVERY NOTE\nDelivery #: {dn_num}\nDate: {date}\nCustomer: {client}\nOrder reference: {order_ref}\nItems delivered: {items}\nCarrier: {carrier}"
    ],
    'bank_statement': [
        "RELEVÉ BANCAIRE\nCompte N°: {account}\nTitulaire: {holder}\nPériode: {period}\nSolde initial: {init_balance}€\nTotal débits: {debits}€\nTotal crédits: {credits}€\nSolde final: {final_balance}€",
        "BANK STATEMENT\nAccount #: {account}\nAccount holder: {holder}\nPeriod: {period}\nOpening balance: ${init_balance}\nTotal debits: ${debits}\nTotal credits: ${credits}\nClosing balance: ${final_balance}"
    ],
    'expense_report': [
        "NOTE DE FRAIS\nEmployé: {employee}\nDépartement: {department}\nDate: {date}\nType de dépense: {expense_type}\nMontant: {amount}€\nJustificatif: {receipts}",
        "EXPENSE REPORT\nEmployee: {employee}\nDepartment: {department}\nDate: {date}\nExpense type: {expense_type}\nAmount: ${amount}\nReceipts: {receipts}"
    ],
    'payslip': [
        "BULLETIN DE PAIE\nEmployeur: {employer}\nEmployé: {employee}\nPériode: {period}\nSalaire brut: {gross}€\nCotisations: {deductions}€\nNet à payer: {net}€\nDate de paiement: {payment_date}",
        "PAYSLIP\nEmployer: {employer}\nEmployee: {employee}\nPeriod: {period}\nGross salary: ${gross}\nDeductions: ${deductions}\nNet pay: ${net}\nPayment date: {payment_date}"
    ]
}

def download_dataset(dataset):
    """Download and extract a dataset"""
    print(f"Downloading {dataset['name']} dataset...")
    try:
        response = requests.get(dataset['url'], stream=True)
        if response.status_code == 200:
            # Extract the zip file
            z = zipfile.ZipFile(io.BytesIO(response.content))
            extract_path = f"data/samples/downloaded/{dataset['name']}"
            os.makedirs(extract_path, exist_ok=True)
            z.extractall(extract_path)
            print(f"Successfully downloaded and extracted {dataset['name']} dataset")
            return True
        else:
            print(f"Failed to download {dataset['name']} dataset. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {dataset['name']} dataset: {str(e)}")
        return False

def generate_synthetic_data(doc_type, count=100):
    """Generate synthetic data for a document type"""
    print(f"Generating {count} synthetic {doc_type} documents...")
    
    synthetic_data = []
    templates = TEMPLATES.get(doc_type, [])
    
    if not templates:
        print(f"No templates available for {doc_type}")
        return synthetic_data
    
    for i in range(count):
        template = random.choice(templates)
        
        # Generate random data based on document type
        if doc_type == 'purchase_order':
            data = {
                'po_num': f"PO-{random.randint(1000, 9999)}",
                'date': f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                'supplier': f"Supplier {random.randint(1, 100)}",
                'items': f"{random.randint(1, 20)} x Item {random.randint(1, 50)}",
                'total': f"{random.randint(100, 10000)}.{random.randint(0, 99):02d}",
                'delivery': str(random.randint(1, 30))
            }
        elif doc_type == 'quote':
            amount = random.randint(500, 5000) + random.randint(0, 99)/100
            tax = amount * 0.2
            total = amount + tax
            data = {
                'quote_num': f"QT-{random.randint(1000, 9999)}",
                'date': f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                'client': f"Client {random.randint(1, 100)}",
                'description': f"Service {random.randint(1, 50)}",
                'amount': f"{amount:.2f}",
                'tax': f"{tax:.2f}",
                'total': f"{total:.2f}",
                'validity': str(random.randint(15, 90))
            }
        elif doc_type == 'delivery_note':
            data = {
                'dn_num': f"DN-{random.randint(1000, 9999)}",
                'date': f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                'client': f"Client {random.randint(1, 100)}",
                'order_ref': f"PO-{random.randint(1000, 9999)}",
                'items': f"{random.randint(1, 20)} x Item {random.randint(1, 50)}",
                'carrier': f"Carrier {random.randint(1, 10)}"
            }
        elif doc_type == 'bank_statement':
            init_balance = random.randint(1000, 10000) + random.randint(0, 99)/100
            debits = random.randint(100, 2000) + random.randint(0, 99)/100
            credits = random.randint(100, 3000) + random.randint(0, 99)/100
            final_balance = init_balance - debits + credits
            data = {
                'account': f"ACC-{random.randint(10000, 99999)}",
                'holder': f"Name {random.randint(1, 100)}",
                'period': f"01/{random.randint(1, 12)}/2023 - {random.randint(25, 30)}/{random.randint(1, 12)}/2023",
                'init_balance': f"{init_balance:.2f}",
                'debits': f"{debits:.2f}",
                'credits': f"{credits:.2f}",
                'final_balance': f"{final_balance:.2f}"
            }
        elif doc_type == 'expense_report':
            data = {
                'employee': f"Employee {random.randint(1, 100)}",
                'department': random.choice(["Sales", "Marketing", "IT", "HR", "Finance"]),
                'date': f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                'expense_type': random.choice(["Travel", "Meals", "Accommodation", "Office Supplies", "Other"]),
                'amount': f"{random.randint(50, 500)}.{random.randint(0, 99):02d}",
                'receipts': random.choice(["Receipt scan", "Ticket + Invoice", "Multiple receipts"])
            }
        elif doc_type == 'payslip':
            gross = random.randint(2000, 5000) + random.randint(0, 99)/100
            deductions = gross * random.uniform(0.2, 0.4)
            net = gross - deductions
            data = {
                'employer': f"Company {random.randint(1, 50)}",
                'employee': f"Employee {random.randint(1, 100)}",
                'period': f"{random.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])} 2023",
                'gross': f"{gross:.2f}",
                'deductions': f"{deductions:.2f}",
                'net': f"{net:.2f}",
                'payment_date': f"{random.randint(25, 30)}/{random.randint(1, 12)}/2023"
            }
        else:
            data = {}
        
        # Format the template with random data
        try:
            text = template.format(**data)
            synthetic_data.append({"text": text, "label": doc_type})
        except KeyError as e:
            print(f"Error generating {doc_type} document: {str(e)}")
    
    return synthetic_data

def process_dataset(dataset_path, doc_type):
    """Process downloaded dataset and convert to our format"""
    processed_data = []
    # Implementation depends on the specific dataset format
    # This is a placeholder - real implementation would parse the specific dataset
    
    # For demonstration, just return a placeholder
    return processed_data

def save_to_csv(data, filename):
    """Save data to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} records to {filename}")

def merge_csv_files(output_file='data/samples/enhanced_commercial_data.csv'):
    """Merge all CSV files in the samples directory"""
    dfs = []
    
    # Read existing data
    for file in Path('data/samples').glob('*.csv'):
        try:
            df = pd.read_csv(file, encoding='utf-8')
            if 'text' in df.columns and 'label' in df.columns:
                dfs.append(df)
                print(f"Added {len(df)} rows from {file}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    # Read synthetic data
    for file in Path('data/samples/synthetic').glob('*.csv'):
        try:
            df = pd.read_csv(file, encoding='utf-8')
            dfs.append(df)
            print(f"Added {len(df)} rows from {file}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    # Merge and save
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Merged dataset saved to {output_file} with {len(merged_df)} records")
    else:
        print("No data to merge")

def main():
    # Download datasets
    for dataset in DATASETS:
        download_dataset(dataset)
    
    # Generate synthetic data for each document type
    for doc_type in TEMPLATES.keys():
        synthetic_data = generate_synthetic_data(doc_type, count=100)
        if synthetic_data:
            save_to_csv(synthetic_data, f"data/samples/synthetic/{doc_type}_synthetic.csv")
    
    # Merge all data into a single file
    merge_csv_files()

if __name__ == "__main__":
    main() 
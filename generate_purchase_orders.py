import os
import csv
import random
from datetime import datetime, timedelta

# Create directory if it doesn't exist
os.makedirs('data/samples/enhanced', exist_ok=True)

# PO Templates in different languages and formats
TEMPLATES = {
    'fr_standard': """BON DE COMMANDE
N° BC: {po_num}
Date: {date}
Fournisseur: {supplier}
{supplier_address}

Client: {customer}
{customer_address}

Articles:
{items}

Sous-total: {subtotal}€
TVA ({tax_rate}%): {tax}€
Total: {total}€

Délai de livraison: {delivery} jours
Mode de paiement: {payment_method}

{signature_line}
""",
    'fr_simple': """BON DE COMMANDE
Commande N°: {po_num}
Date: {date}
Fournisseur: {supplier}
Client: {customer}

{items}

Total HT: {subtotal}€
Total TTC: {total}€
""",
    'fr_formal': """BON DE COMMANDE
RÉFÉRENCE: {po_num}
ÉMIS LE: {date}

FOURNISSEUR:
{supplier}
{supplier_address}

DESTINATAIRE:
{customer}
{customer_address}

DÉSIGNATION:
{items}

CONDITIONS:
Livraison: {delivery} jours
Paiement: {payment_method}

MONTANT:
Total HT: {subtotal}€
TVA {tax_rate}%: {tax}€
Total TTC: {total}€

Signature: ________________
""",
    'en_standard': """PURCHASE ORDER
PO Number: {po_num}
Date: {date}
Vendor: {supplier}
{supplier_address}

Bill To: {customer}
{customer_address}

Items:
{items}

Subtotal: ${subtotal}
Tax ({tax_rate}%): ${tax}
Total: ${total}

Delivery terms: {delivery} days
Payment method: {payment_method}

{signature_line}
""",
    'en_simple': """PURCHASE ORDER
Order #: {po_num}
Date: {date}
Vendor: {supplier}
Customer: {customer}

{items}

Subtotal: ${subtotal}
Total: ${total}
""",
    'ar_standard': """طلب شراء
رقم الطلب: {po_num}
التاريخ: {date}
المورد: {supplier}
{supplier_address}

العميل: {customer}
{customer_address}

المواد:
{items}

المجموع الفرعي: {subtotal}
الضريبة ({tax_rate}%): {tax}
المجموع الكلي: {total}

مدة التسليم: {delivery} أيام
طريقة الدفع: {payment_method}

{signature_line}
"""
}

# Sample data pools
COMPANIES = [
    "Tech Solutions", "Global Supplies", "Office Pro", "Industrial Equipment", 
    "Martin SARL", "Dupont Entreprises", "Michelin", "Carrefour", "Orange",
    "Total Energies", "Bureau Vallée", "Schmidt Groupe", "Leroy Merlin",
    "الشركة العربية", "مؤسسة التجارة", "شركة المستقبل", "التقنية الحديثة"
]

ADDRESSES = [
    "123 Rue de Paris, 75001 Paris, France",
    "45 Avenue des Champs-Élysées, 75008 Paris, France",
    "78 Boulevard Haussmann, 75009 Paris, France",
    "56 Rue de Lyon, 69001 Lyon, France",
    "123 Main St, New York, NY 10001, USA",
    "456 Market St, San Francisco, CA 94105, USA",
    "789 Business Ave, Chicago, IL 60601, USA",
    "شارع الملك فهد، الرياض، المملكة العربية السعودية",
    "شارع الشيخ زايد، دبي، الإمارات العربية المتحدة"
]

PRODUCTS = [
    "Laptop Dell XPS 15", "Serveur HP ProLiant", "Imprimante HP LaserJet",
    "Scanner Epson", "Écran Dell 27\"", "Clavier Logitech", "Souris Microsoft",
    "Téléphone IP Cisco", "Routeur Cisco", "Switch Netgear", "Câble HDMI 2m",
    "Office Desk", "Office Chair", "Filing Cabinet", "Conference Table", 
    "Whiteboard", "Paper Shredder", "Coffee Machine", "Water Dispenser",
    "حاسوب محمول", "طابعة ليزر", "خادم شبكة", "جهاز تخزين", "شاشة عرض"
]

PAYMENT_METHODS = [
    "Virement bancaire", "Carte bancaire", "Chèque", "Prélèvement automatique",
    "Bank Transfer", "Credit Card", "Check", "Cash on Delivery",
    "شيك", "تحويل بنكي", "بطاقة ائتمان"
]

SIGNATURE_LINES = [
    "Signature et cachet de l'entreprise",
    "Signature et tampon", 
    "Signature autorisée",
    "Authorized Signature", 
    "Company Stamp and Signature",
    "توقيع معتمد",
    "ختم الشركة والتوقيع"
]

def generate_items(count, lang='fr'):
    """Generate a random list of items"""
    items_text = ""
    total = 0
    
    for i in range(count):
        product = random.choice(PRODUCTS)
        quantity = random.randint(1, 10)
        price = round(random.uniform(10, 500), 2)
        item_total = quantity * price
        total += item_total
        
        if lang == 'fr':
            items_text += f"{quantity} x {product} - Prix unitaire: {price:.2f}€ - Total: {item_total:.2f}€\n"
        elif lang == 'en':
            items_text += f"{quantity} x {product} - Unit price: ${price:.2f} - Total: ${item_total:.2f}\n"
        elif lang == 'ar':
            items_text += f"{product} x {quantity} - السعر: {price:.2f} - المجموع: {total:.2f}\n"
    
    return items_text, total

def generate_date(format_type='fr'):
    """Generate a random date from the last 6 months"""
    today = datetime.now()
    days_ago = random.randint(0, 180)
    random_date = today - timedelta(days=days_ago)
    
    if format_type == 'fr':
        return random_date.strftime('%d/%m/%Y')
    elif format_type == 'en':
        return random_date.strftime('%m/%d/%Y')
    elif format_type == 'ar':
        return random_date.strftime('%d/%m/%Y')

def generate_purchase_order(template_key):
    """Generate a purchase order using the specified template"""
    template = TEMPLATES[template_key]
    lang = template_key.split('_')[0]
    
    # Generate common data
    po_num = f"PO-{random.randint(1000, 9999)}"
    supplier = random.choice(COMPANIES)
    supplier_address = random.choice(ADDRESSES)
    customer = random.choice(COMPANIES)
    while customer == supplier:  # Ensure different companies
        customer = random.choice(COMPANIES)
    customer_address = random.choice(ADDRESSES)
    
    # Generate items and calculate totals
    items_count = random.randint(2, 6)
    items_text, subtotal = generate_items(items_count, lang)
    tax_rate = random.choice([5, 10, 20])
    tax = round(subtotal * tax_rate / 100, 2)
    total = subtotal + tax
    
    # Other fields
    date = generate_date('fr' if lang == 'fr' else 'en' if lang == 'en' else 'ar')
    delivery = random.randint(3, 30)
    payment_method = random.choice(PAYMENT_METHODS)
    signature_line = random.choice(SIGNATURE_LINES)
    
    # Format the template with data
    po_text = template.format(
        po_num=po_num,
        date=date,
        supplier=supplier,
        supplier_address=supplier_address,
        customer=customer,
        customer_address=customer_address,
        items=items_text,
        subtotal=f"{subtotal:.2f}",
        tax_rate=tax_rate,
        tax=f"{tax:.2f}",
        total=f"{total:.2f}",
        delivery=delivery,
        payment_method=payment_method,
        signature_line=signature_line
    )
    
    return po_text

def save_to_csv(data, filename):
    """Save data to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} records to {filename}")

def main():
    print("Generating purchase order examples...")
    
    purchase_orders = []
    
    # Generate purchase orders using all templates
    for template_key in TEMPLATES.keys():
        # Generate 50 examples for each template
        for _ in range(50):
            po_text = generate_purchase_order(template_key)
            purchase_orders.append({
                "text": po_text,
                "label": "purchase_order"
            })
    
    # Save the data
    save_to_csv(purchase_orders, "data/samples/enhanced/purchase_orders.csv")
    
    print(f"Generated {len(purchase_orders)} purchase order examples")

if __name__ == "__main__":
    main() 
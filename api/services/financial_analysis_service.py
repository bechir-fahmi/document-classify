"""
Financial Analysis Service - Single Responsibility Principle
Handles financial document analysis and bilan generation
"""
import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from api.interfaces.document_classifier import IFinancialAnalyzer, ITextExtractor

logger = logging.getLogger(__name__)


class FinancialAnalysisService:
    """Service for financial analysis - follows SRP"""
    
    def __init__(
        self,
        financial_analyzer: IFinancialAnalyzer,
        text_extractor: ITextExtractor,
        classification_service: 'DocumentClassificationService'
    ):
        self._financial_analyzer = financial_analyzer
        self._text_extractor = text_extractor
        self._classification_service = classification_service
    
    def analyze_document_financial(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a document for financial information
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Financial analysis result
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        try:
            # Extract text
            text = self._text_extractor.extract_text(file_path)
            
            # Classify document to determine type
            classification_result = self._classification_service.classify_document(
                file_path, upload_to_cloud=False
            )
            
            final_doc_type = classification_result["final_prediction"]
            
            # Analyze financial information
            financial_data = self._financial_analyzer.analyze_financial_data(
                text, final_doc_type, document_id
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "document_id": document_id,
                "financial_analysis": financial_data,
                "document_classification": {
                    "rule_based_prediction": classification_result["rule_based_prediction"],
                    "model_prediction": classification_result["model_prediction"],
                    "model_confidence": classification_result["model_confidence"],
                    "final_prediction": final_doc_type,
                    "confidence_scores": classification_result["confidence_scores"]
                },
                "document_embedding": classification_result["document_embedding"],
                "embedding_model": classification_result["embedding_model"],
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in financial analysis: {str(e)}")
            raise
    
    def generate_financial_bilan(
        self, 
        file_paths: List[str], 
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate financial bilan from multiple documents
        
        Args:
            file_paths: List of document file paths
            period_days: Analysis period in days
            
        Returns:
            Financial bilan report
        """
        start_time = time.time()
        transactions = []
        processed_files = 0
        
        logger.info(f"Processing {len(file_paths)} files for financial bilan")
        
        for file_path in file_paths:
            try:
                # Analyze each document
                result = self.analyze_document_financial(file_path)
                
                # Extract transaction data
                financial_data = result["financial_analysis"]
                if financial_data and financial_data.get("amount", 0) > 0:
                    transactions.append(financial_data)
                    processed_files += 1
                    
                    logger.info(
                        f"Processed {file_path}: {result['document_classification']['final_prediction']}, "
                        f"Amount: {financial_data.get('amount', 0)} {financial_data.get('currency', 'EUR')}"
                    )
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        if not transactions:
            raise ValueError("No valid financial documents could be processed")
        
        # Generate bilan report
        bilan = self._generate_bilan_report(transactions, period_days)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Generated bilan from {processed_files} documents in {processing_time:.1f}ms")
        
        return {
            **bilan,
            "processing_time_ms": processing_time,
            "processed_documents": processed_files
        }
    
    def generate_bilan_from_files_directly(
        self, 
        downloaded_files: List[Dict[str, Any]], 
        business_info: Dict[str, Any], 
        period_days: int
    ) -> Dict[str, Any]:
        """
        Generate bilan directly from downloaded files using Groq AI
        This maintains compatibility with the original implementation
        """
        try:
            # Extract texts from files
            extracted_texts = []
            
            for file_info in downloaded_files:
                file_path = file_info['file_path']
                original_doc = file_info['original_doc']
                
                try:
                    # Extract text from file
                    text = self._text_extractor.extract_text(file_path)
                    
                    if text and len(text.strip()) > 10:  # Only include files with meaningful content
                        extracted_texts.append({
                            "filename": original_doc.get('filename', 'unknown'),
                            "document_type": original_doc.get('document_type', 'unknown'),
                            "text": text[:3000]  # Limit text length for API
                        })
                        logger.info(f"Extracted text from: {original_doc.get('filename', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"Error extracting text from {file_path}: {str(e)}")
                    continue
            
            if not extracted_texts:
                raise ValueError("No processable documents found with extractable text")
            
            logger.info(f"Processing {len(extracted_texts)} documents for bilan generation")
            
            # Use Groq financial analyzer for bilan generation
            bilan_result = self._generate_groq_bilan(extracted_texts, business_info, period_days)
            
            return bilan_result
            
        except Exception as e:
            logger.error(f"Error in direct file bilan generation: {str(e)}")
            raise ValueError(f"Failed to generate bilan from files: {str(e)}")
    
    def _generate_groq_bilan(
        self, 
        extracted_texts: List[Dict[str, Any]], 
        business_info: Dict[str, Any], 
        period_days: int
    ) -> Dict[str, Any]:
        """Generate bilan using Groq AI"""
        try:
            # Prepare text context
            documents_context = ""
            for i, doc in enumerate(extracted_texts, 1):
                documents_context += f"\n--- Document {i}: {doc['filename']} (Type: {doc['document_type']}) ---\n"
                documents_context += f"{doc['text']}\n"
                documents_context += "-" * 80 + "\n"
            
            # Create comprehensive prompt for Tunisian bilan
            prompt = f"""Analyze these {len(extracted_texts)} financial documents and generate a complete Tunisian accounting bilan following the Plan Comptable Tunisien.

Business Information:
- Company: {business_info.get('name', 'N/A')}
- Period: {business_info.get('period_start', 'N/A')} to {business_info.get('period_end', 'N/A')}
- Analysis Period: Last {period_days} days

Documents to analyze:
{documents_context}

Generate EXACTLY this JSON structure:
{{
    "bilan_comptable": {{
        "actif": {{
            "actif_non_courant": {{
                "immobilisations_corporelles": 0,
                "immobilisations_incorporelles": 0,
                "immobilisations_financieres": 0,
                "total_actif_non_courant": 0
            }},
            "actif_courant": {{
                "stocks_et_en_cours": 0,
                "clients_et_comptes_rattaches": 0,
                "autres_creances": 0,
                "disponibilites": 0,
                "total_actif_courant": 0
            }},
            "total_actif": 0
        }},
        "passif": {{
            "capitaux_propres": {{
                "capital_social": 0,
                "reserves": 0,
                "resultat_net_exercice": 0,
                "total_capitaux_propres": 0
            }},
            "passif_non_courant": {{
                "emprunts_dettes_financieres_lt": 0,
                "provisions_lt": 0,
                "total_passif_non_courant": 0
            }},
            "passif_courant": {{
                "fournisseurs_et_comptes_rattaches": 0,
                "dettes_fiscales_et_sociales": 0,
                "autres_dettes_ct": 0,
                "total_passif_courant": 0
            }},
            "total_passif": 0
        }}
    }},
    "compte_de_resultat": {{
        "produits_exploitation": {{
            "chiffre_affaires": 0,
            "production_immobilisee": 0,
            "subventions_exploitation": 0,
            "autres_produits_exploitation": 0,
            "total_produits_exploitation": 0
        }},
        "charges_exploitation": {{
            "achats_consommes": 0,
            "charges_personnel": 0,
            "dotations_amortissements": 0,
            "autres_charges_exploitation": 0,
            "total_charges_exploitation": 0
        }},
        "resultat_exploitation": 0,
        "resultat_financier": 0,
        "resultat_exceptionnel": 0,
        "resultat_avant_impot": 0,
        "impot_sur_benefices": 0,
        "resultat_net": 0
    }},
    "ratios_financiers": {{
        "marge_brute_percent": 0.0,
        "marge_nette_percent": 0.0,
        "rentabilite_actif_percent": 0.0,
        "liquidite_generale": 0.0,
        "autonomie_financiere_percent": 0.0
    }},
    "analyse_financiere": {{
        "points_forts": [],
        "points_faibles": [],
        "recommandations": []
    }},
    "details_transactions": []
}}

INSTRUCTIONS:
1. Analyze each document and extract financial data
2. Classify according to Tunisian accounting standards
3. Calculate totals and ratios accurately
4. Return ONLY valid JSON, no additional text
"""
            
            # Use Groq financial analyzer
            try:
                from utils.groq_utils import client as groq_client
                
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a Tunisian certified accountant. Analyze documents to generate accurate bilans following Plan Comptable Tunisien."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                groq_response = response.choices[0].message.content
                logger.info(f"Groq bilan response received: {len(groq_response)} characters")
                
                # Parse the JSON response
                import json
                try:
                    bilan_data = json.loads(groq_response)
                    return bilan_data
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', groq_response, re.DOTALL)
                    if json_match:
                        bilan_data = json.loads(json_match.group())
                        return bilan_data
                    else:
                        raise ValueError("Could not parse JSON from Groq response")
                        
            except Exception as e:
                logger.error(f"Error calling Groq API: {str(e)}")
                # Fallback to simplified bilan
                return self._generate_fallback_bilan(extracted_texts, business_info, period_days)
                
        except Exception as e:
            logger.error(f"Error generating Groq bilan: {str(e)}")
            return self._generate_fallback_bilan(extracted_texts, business_info, period_days)
    
    def _generate_fallback_bilan(
        self, 
        extracted_texts: List[Dict[str, Any]], 
        business_info: Dict[str, Any], 
        period_days: int
    ) -> Dict[str, Any]:
        """Generate a fallback bilan when Groq is not available"""
        return {
            "bilan_comptable": {
                "actif": {
                    "actif_non_courant": {
                        "immobilisations_corporelles": 0,
                        "immobilisations_incorporelles": 0,
                        "immobilisations_financieres": 0,
                        "total_actif_non_courant": 0
                    },
                    "actif_courant": {
                        "stocks_et_en_cours": 0,
                        "clients_et_comptes_rattaches": 0,
                        "autres_creances": 0,
                        "disponibilites": 0,
                        "total_actif_courant": 0
                    },
                    "total_actif": 0
                },
                "passif": {
                    "capitaux_propres": {
                        "capital_social": 0,
                        "reserves": 0,
                        "resultat_net_exercice": 0,
                        "total_capitaux_propres": 0
                    },
                    "passif_non_courant": {
                        "emprunts_dettes_financieres_lt": 0,
                        "provisions_lt": 0,
                        "total_passif_non_courant": 0
                    },
                    "passif_courant": {
                        "fournisseurs_et_comptes_rattaches": 0,
                        "dettes_fiscales_et_sociales": 0,
                        "autres_dettes_ct": 0,
                        "total_passif_courant": 0
                    },
                    "total_passif": 0
                }
            },
            "compte_de_resultat": {
                "produits_exploitation": {
                    "chiffre_affaires": 0,
                    "total_produits_exploitation": 0
                },
                "charges_exploitation": {
                    "achats_consommes": 0,
                    "charges_personnel": 0,
                    "total_charges_exploitation": 0
                },
                "resultat_exploitation": 0,
                "resultat_net": 0
            },
            "ratios_financiers": {
                "marge_brute_percent": 0.0,
                "marge_nette_percent": 0.0,
                "rentabilite_actif_percent": 0.0,
                "liquidite_generale": 0.0,
                "autonomie_financiere_percent": 0.0
            },
            "analyse_financiere": {
                "points_forts": ["Documents analysés avec succès"],
                "points_faibles": ["Données limitées disponibles"],
                "recommandations": ["Fournir plus de documents pour une analyse complète"]
            },
            "details_transactions": [],
            "processed_documents": len(extracted_texts),
            "business_info": business_info,
            "period_days": period_days,
            "generated_at": datetime.now().isoformat(),
            "note": "Bilan généré en mode de secours - données limitées"
        }

    def _generate_bilan_report(self, transactions: List[Dict[str, Any]], period_days: int) -> Dict[str, Any]:
        """Generate bilan report from transactions (simplified version)"""
        total_amount = sum(t.get("amount", 0) for t in transactions)
        currencies = set(t.get("currency", "EUR") for t in transactions)
        
        # Group by category
        categories = {}
        for transaction in transactions:
            category = transaction.get("category", "other")
            if category not in categories:
                categories[category] = {"count": 0, "total": 0}
            categories[category]["count"] += 1
            categories[category]["total"] += transaction.get("amount", 0)
        
        return {
            "period": {
                "days": period_days,
                "start_date": (datetime.now().replace(day=1)).isoformat(),
                "end_date": datetime.now().isoformat()
            },
            "summary": {
                "total_amount": total_amount,
                "transaction_count": len(transactions),
                "currencies": list(currencies)
            },
            "currency_breakdown": {
                currency: {
                    "total": sum(t.get("amount", 0) for t in transactions if t.get("currency") == currency),
                    "count": len([t for t in transactions if t.get("currency") == currency])
                }
                for currency in currencies
            },
            "category_breakdown": categories,
            "recommendations": self._generate_recommendations(transactions),
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, transactions: List[Dict[str, Any]]) -> List[str]:
        """Generate financial recommendations based on transactions"""
        recommendations = []
        
        if len(transactions) > 10:
            recommendations.append("Consider implementing automated expense tracking")
        
        # Check for high-value transactions
        high_value_count = len([t for t in transactions if t.get("amount", 0) > 1000])
        if high_value_count > 0:
            recommendations.append(f"Review {high_value_count} high-value transactions for accuracy")
        
        return recommendations
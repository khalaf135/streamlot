"""
UAE SME Onboarding - Document Extraction Schemas
=================================================
Based on the UAE Document Requirements table.

Usage pattern:
    1. Call CLASSIFIER_SYSTEM on the OCR'd text -> detects doc_type
    2. Call EXTRACTION_SYSTEM with SCHEMAS[doc_type] -> returns filled JSON
    3. Validate / store / route to human review if confidence == "low"
"""

# ============================================================
# SYSTEM PROMPTS
# ============================================================

CLASSIFIER_SYSTEM = (
    "You are a UAE legal document classifier. Given OCR text from a document, "
    "identify the document type. Return ONLY valid JSON:\n"
    '{"doc_type": "<type>", "confidence": "high|medium|low", "reason": "<short reason>"}\n\n'
    "Valid doc_type values:\n"
    "  trade_license, certificate_of_incorporation, moa, aoa, power_of_attorney,\n"
    "  establishment_card, passport, emirates_id, uae_visa, proof_of_address,\n"
    "  individual_kyc_form, ejari, utility_bill, invoice_contract, company_profile,\n"
    "  chamber_of_commerce, bank_statement, audited_financials, business_plan,\n"
    "  source_of_funds, vat_certificate, unknown"
)


EXTRACTION_SYSTEM = (
    "You are an expert legal assistant extracting fields from UAE company documents.\n\n"
    "RULES:\n"
    "1. Use ONLY the provided context. Do not infer or use outside knowledge.\n"
    "2. Extract values VERBATIM as they appear. Do not rephrase, expand abbreviations, "
    "   or substitute text from footers, watermarks, or logos.\n"
    "3. For bilingual fields (EN/AR), fill both if both exist in the document; "
    "   otherwise set the missing language to null.\n"
    "4. Keep dates in the exact format shown in the document (do NOT reformat).\n"
    "5. For money, include the currency exactly as written (e.g., '55,000.00 AED').\n"
    "6. Multi-value fields (activities, shareholders) must be returned as arrays "
    "   following the schema.\n"
    "7. Missing or unreadable fields -> null. Do not guess.\n"
    "8. Return ONLY valid JSON matching the provided schema. No preamble, no markdown.\n"
    "9. Add a top-level 'extraction_confidence' field: 'high' | 'medium' | 'low'.\n"
    "   Use 'low' if OCR quality seems poor or many fields are null.\n"
)


# ============================================================
# REUSABLE SUB-SCHEMAS (referenced below)
# ============================================================

PERSON_SCHEMA = {
    "full_name_en": "string or null",
    "full_name_ar": "string or null",
    "nationality": "string or null",
    "role": "string or null",          # e.g. 'Manager', 'Shareholder', 'UBO'
    "id_number": "string or null",     # passport/EID/internal person no.
    "ownership_percentage": "string or null"  # e.g. '100.00%'
}

ADDRESS_SCHEMA = {
    "full_address": "string or null",
    "building": "string or null",
    "unit_or_office": "string or null",
    "area": "string or null",
    "emirate": "string or null",
    "po_box": "string or null",
    "makani_no": "string or null"
}

CONTACT_SCHEMA = {
    "phone": "string or null",
    "mobile": "string or null",
    "fax": "string or null",
    "email": "string or null",
    "website": "string or null"
}


# ============================================================
# PER-DOCUMENT SCHEMAS
# ============================================================

SCHEMAS = {

    # --- LEGAL ---------------------------------------------------------------

    "trade_license": {
        "doc_type": "trade_license",
        "extraction_confidence": "high | medium | low",
        "license_no": None,
        "main_license_no": None,
        "legal_name_en": None,
        "legal_name_ar": None,
        "trade_name_en": None,
        "trade_name_ar": None,
        "legal_type": None,                         # LLC-SO, LLC, Sole Est., etc.
        "license_category": None,                   # e.g. DED Dubai
        "issuing_authority": None,                  # DED-Dubai, DMCC, ADGM...
        "issue_date": None,
        "expiry_date": None,
        "register_no": None,                        # Commercial register no.
        "chamber_no": None,                         # DCCI / Chamber
        "duns_no": None,                            # D&B D-U-N-S
        "capital": {
            "nominated": None,
            "paid": None,
            "no_of_shares": None,
            "currency": None
        },
        "activities": [                             # list of strings, verbatim
            # "Information Technology Network Services", ...
        ],
        "shareholders": [PERSON_SCHEMA],            # each entry follows PERSON_SCHEMA
        "managers": [PERSON_SCHEMA],
        "address": ADDRESS_SCHEMA,
        "contact": CONTACT_SCHEMA,
        "branches": [
            {"license_no": None, "business_name_en": None, "business_name_ar": None}
        ]
    },

    "certificate_of_incorporation": {
        "doc_type": "certificate_of_incorporation",
        "extraction_confidence": "high | medium | low",
        "registration_no": None,
        "company_name_en": None,
        "company_name_ar": None,
        "incorporation_date": None,
        "jurisdiction": None,                       # e.g. DIFC, ADGM, JAFZA
        "legal_type": None,
        "registered_office": ADDRESS_SCHEMA,
        "issuing_authority": None
    },

    "moa": {
        "doc_type": "moa",
        "extraction_confidence": "high | medium | low",
        "company_name_en": None,
        "company_name_ar": None,
        "signing_date": None,
        "notarization_date": None,
        "capital": {
            "total": None,
            "paid": None,
            "no_of_shares": None,
            "share_value": None,
            "currency": None
        },
        "shareholders": [PERSON_SCHEMA],
        "business_activities": [],                  # list of verbatim strings
        "registered_office": ADDRESS_SCHEMA
    },

    "aoa": {
        "doc_type": "aoa",
        "extraction_confidence": "high | medium | low",
        "company_name_en": None,
        "company_name_ar": None,
        "signing_date": None,
        "board_composition": None,                  # free text / list
        "board_powers": [],                         # list of strings
        "roles": [
            {"role_title": None, "holder_name": None, "powers": None}
        ],
        "governance_notes": None
    },

    "power_of_attorney": {
        "doc_type": "power_of_attorney",
        "extraction_confidence": "high | medium | low",
        "principal": PERSON_SCHEMA,
        "agent": PERSON_SCHEMA,
        "authority_scope": None,                    # verbatim block
        "issue_date": None,
        "expiry_date": None,
        "is_notarized": None,                       # true/false/null
        "notary_public": None,
        "jurisdiction": None
    },

    "establishment_card": {
        "doc_type": "establishment_card",
        "extraction_confidence": "high | medium | low",
        "company_name_en": None,
        "company_name_ar": None,
        "immigration_no": None,
        "unified_no": None,
        "issue_date": None,
        "expiry_date": None,
        "issuing_authority": None                   # ICP / GDRFA
    },

    # --- KYC ----------------------------------------------------------------

    "passport": {
        "doc_type": "passport",
        "extraction_confidence": "high | medium | low",
        "full_name": None,
        "full_name_ar": None,
        "passport_no": None,
        "nationality": None,
        "date_of_birth": None,
        "place_of_birth": None,
        "gender": None,
        "issue_date": None,
        "expiry_date": None,
        "place_of_issue": None,
        "issuing_authority": None
    },

    "emirates_id": {
        "doc_type": "emirates_id",
        "extraction_confidence": "high | medium | low",
        "emirates_id_no": None,                     # 784-YYYY-XXXXXXX-X
        "full_name_en": None,
        "full_name_ar": None,
        "nationality": None,
        "date_of_birth": None,
        "gender": None,
        "issue_date": None,
        "expiry_date": None,
        "card_number": None                         # physical card no.
    },

    "uae_visa": {
        "doc_type": "uae_visa",
        "extraction_confidence": "high | medium | low",
        "full_name": None,
        "visa_type": None,                          # Employment, Investor, Residence
        "uid_no": None,
        "visa_file_no": None,
        "sponsor_name": None,
        "sponsor_no": None,
        "profession": None,
        "issue_date": None,
        "expiry_date": None,
        "place_of_issue": None,
        "entry_stamp_date": None
    },

    "proof_of_address": {
        "doc_type": "proof_of_address",
        "extraction_confidence": "high | medium | low",
        "source_document": None,                    # DEWA bill, tenancy, bank letter...
        "full_name": None,
        "address": ADDRESS_SCHEMA,
        "issue_date": None
    },

    "individual_kyc_form": {
        "doc_type": "individual_kyc_form",
        "extraction_confidence": "high | medium | low",
        "full_name": None,
        "date_of_birth": None,
        "nationality": None,
        "id_type": None,
        "id_number": None,
        "residential_address": ADDRESS_SCHEMA,
        "occupation": None,
        "employer": None,
        "annual_income_range": None,
        "source_of_funds": None,                    # salary, investments, inheritance
        "source_of_wealth": None,
        "is_pep": None,                             # Politically Exposed Person
        "pep_details": None,
        "risk_profile": None,                       # low/medium/high
        "form_date": None
    },

    # --- BUSINESS PROOF ------------------------------------------------------

    "ejari": {
        "doc_type": "ejari",
        "extraction_confidence": "high | medium | low",
        "contract_no": None,
        "registration_date": None,
        "owner": {
            "name_en": None,
            "name_ar": None,
            "owner_no": None,
            "nationality": None
        },
        "lessor": {                                 # property mgmt company
            "name_en": None,
            "name_ar": None,
            "license_no": None,
            "license_issuer": None
        },
        "tenant": {
            "name_en": None,
            "name_ar": None,
            "tenant_no": None,
            "license_no": None,
            "license_expiry": None,
            "license_issuer": None
        },
        "lease_terms": {
            "start_date": None,
            "end_date": None,
            "grace_start_date": None,
            "grace_end_date": None,
            "contract_amount": None,                # e.g. "55,000.00 AED"
            "actual_contract_amount": None,
            "annual_amount": None,
            "actual_annual_amount": None,
            "discount": None,
            "security_deposit": None
        },
        "property": {
            "building_name": None,
            "plot_number": None,
            "land_area": None,
            "land_dm_no": None,
            "makani_no": None,
            "property_no": None,
            "type": None,                           # Office / Shop / etc.
            "sub_type": None,
            "usage": None,                          # Commercial / Residential
            "size_sqm": None,
            "dewa_premise_no": None
        },
        "fees": {
            "receipt_no": None,
            "rent_registration_fee": None,
            "ejari_license_fee": None,
            "knowledge_fee": None,
            "innovation_fee": None,
            "vat": None,
            "total_fees": None
        }
    },

    "utility_bill": {
        "doc_type": "utility_bill",
        "extraction_confidence": "high | medium | low",
        "provider": None,                           # DEWA, Etisalat, du, SEWA, ADDC
        "account_holder_name": None,
        "account_number": None,
        "premise_no": None,
        "service_address": ADDRESS_SCHEMA,
        "bill_date": None,
        "billing_period_from": None,
        "billing_period_to": None,
        "amount_due": None,
        "due_date": None
    },

    "invoice_contract": {
        "doc_type": "invoice_contract",
        "extraction_confidence": "high | medium | low",
        "document_kind": None,                      # 'invoice' or 'contract'
        "invoice_or_contract_no": None,
        "issue_date": None,
        "issuer_name": None,
        "client_name": None,
        "client_trn": None,
        "services_description": None,
        "line_items": [
            {"description": None, "quantity": None, "unit_price": None, "total": None}
        ],
        "subtotal": None,
        "vat_amount": None,
        "total_amount": None,
        "currency": None,
        "payment_terms": None
    },

    "company_profile": {
        "doc_type": "company_profile",
        "extraction_confidence": "high | medium | low",
        "company_name_en": None,
        "company_name_ar": None,
        "founded_year": None,
        "industry": None,
        "services": [],                             # list of strings
        "geography": [],                            # countries/regions served
        "website": None,
        "key_clients": [],
        "team_size": None
    },

    "chamber_of_commerce": {
        "doc_type": "chamber_of_commerce",
        "extraction_confidence": "high | medium | low",
        "chamber_name": None,                       # DCCI, Abu Dhabi Chamber...
        "registration_no": None,
        "member_name_en": None,
        "member_name_ar": None,
        "membership_category": None,
        "issue_date": None,
        "expiry_date": None
    },

    # --- FINANCIAL / BANKING ------------------------------------------------

    "bank_statement": {
        "doc_type": "bank_statement",
        "extraction_confidence": "high | medium | low",
        "bank_name": None,
        "account_holder": None,
        "account_number": None,
        "iban": None,
        "currency": None,
        "statement_period_from": None,
        "statement_period_to": None,
        "opening_balance": None,
        "closing_balance": None,
        "average_balance": None,
        "total_credits": None,
        "total_debits": None,
        "number_of_transactions": None,
        "top_counterparties": [
            {"name": None, "total_amount": None, "transaction_count": None}
        ]
    },

    "audited_financials": {
        "doc_type": "audited_financials",
        "extraction_confidence": "high | medium | low",
        "company_name": None,
        "fiscal_year_end": None,
        "auditor_name": None,
        "audit_opinion": None,                      # Unqualified / Qualified / etc.
        "currency": None,
        "revenue": None,
        "gross_profit": None,
        "net_profit": None,
        "ebitda": None,
        "total_assets": None,
        "total_liabilities": None,
        "total_equity": None,
        "cash_and_equivalents": None
    },

    "business_plan": {
        "doc_type": "business_plan",
        "extraction_confidence": "high | medium | low",
        "company_name": None,
        "plan_period": None,
        "expected_annual_turnover": None,
        "revenue_streams": [],
        "target_markets": [],
        "key_products_services": [],
        "projected_headcount": None,
        "funding_required": None
    },

    "source_of_funds": {
        "doc_type": "source_of_funds",
        "extraction_confidence": "high | medium | low",
        "declarant_name": None,
        "declaration_date": None,
        "income_source": None,                      # salary, business, inheritance
        "funding_type": None,                       # initial capital, ongoing
        "amount": None,
        "currency": None,
        "supporting_documents_mentioned": []
    },

    "vat_certificate": {
        "doc_type": "vat_certificate",
        "extraction_confidence": "high | medium | low",
        "trn_no": None,                             # Tax Registration Number
        "company_name_en": None,
        "company_name_ar": None,
        "registration_date": None,
        "effective_date": None,
        "tax_period": None,                         # monthly / quarterly
        "issuing_authority": None                   # FTA
    }
}


# ============================================================
# HELPER: build the user prompt for a given doc
# ============================================================

def build_extraction_prompt(doc_type: str, ocr_text: str) -> str:
    import json
    if doc_type not in SCHEMAS:
        raise ValueError(f"Unknown doc_type: {doc_type}")
    schema_str = json.dumps(SCHEMAS[doc_type], indent=2, ensure_ascii=False)
    return (
        f"Document type: {doc_type}\n\n"
        f"Schema to fill (return JSON with this exact structure):\n"
        f"{schema_str}\n\n"
        f"--- DOCUMENT OCR TEXT ---\n"
        f"{ocr_text}\n"
        f"--- END ---\n\n"
        f"Return the filled JSON only."
    )
